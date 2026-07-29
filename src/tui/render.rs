//! Frame rendering for the ratatui monitor UI.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::charts::{self, ChartGranularity};
use crate::formatting::format_y_axis_value;
use crate::process_usage::{ProcessUsageSnapshot, process_usage_text};
use crate::tool::Tool;
use crate::tui::commands::{HelpView, help_topics};
use crate::tui::data::{ChartData, Dashboard};
use crate::tui::input::InputLine;
use crate::tui::palette::{ACCENT, DIM};
use crate::tui::table::draw_table;
use crate::{AppState, IntegrityStatus};

pub struct Ui<'a> {
    pub dash: &'a Dashboard,
    pub state: &'a AppState,
    pub input: &'a InputLine,
    pub notice: Option<&'a str>,
    pub sync_status: Option<&'a str>,
    pub process_usage: Option<ProcessUsageSnapshot>,
    pub refresh_in: std::time::Duration,
    pub help: Option<HelpView>,
}

const MIN_WIDTH: u16 = 68;
const MIN_HEIGHT: u16 = 22;

pub fn draw(frame: &mut Frame, ui: &Ui) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        draw_too_small(frame, area);
        return;
    }

    let [header, body, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(10),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_header(frame, header, ui);

    if !ui.dash.has_visible_data {
        let message = if !ui.dash.window_complete {
            "Loading usage history...".to_string()
        } else {
            match ui.dash.session_id.as_deref() {
                Some(session_id) => format!("No usage data found for session {session_id}."),
                None => "No usage data found from any tool.".to_string(),
            }
        };
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL)),
            body,
        );
    } else {
        // Table gets what it needs (rows + chrome), charts take the rest,
        // and the split flips in the charts' favor when height is tight.
        let table_want = ui.dash.rows.len() as u16 + 6;
        let min_charts = 12 * ui.dash.charts.len() as u16;
        let table_height = table_want.min(body.height.saturating_sub(min_charts).max(8));
        let [table_area, charts_area] =
            Layout::vertical([Constraint::Length(table_height), Constraint::Min(6)]).areas(body);
        draw_table(frame, table_area, ui.dash);
        draw_charts(frame, charts_area, ui);
    }

    draw_footer(frame, footer, ui);

    if let Some(view) = ui.help {
        draw_help(frame, area, view);
    }
}

fn draw_too_small(frame: &mut Frame, area: Rect) {
    let text = vec![
        Line::from("Terminal too small"),
        Line::from(format!(
            "current {}x{}, need at least {}x{}",
            area.width, area.height, MIN_WIDTH, MIN_HEIGHT
        )),
    ];
    frame.render_widget(Paragraph::new(text).alignment(Alignment::Center), area);
}

fn draw_header(frame: &mut Frame, area: Rect, ui: &Ui) {
    let [tabs_area, status_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);

    let mut spans: Vec<Span> = vec![
        Span::styled(" ai-usage ", Style::default().fg(ACCENT).bold()),
        Span::styled(
            format!("v{} ", env!("CARGO_PKG_VERSION")),
            Style::default().fg(DIM),
        ),
    ];
    for tool in Tool::ROTATION {
        let label = format!(" {} ", tool.display_name());
        if tool.key() == ui.state.tool {
            spans.push(Span::styled(
                label,
                Style::default().fg(Color::Black).bg(ACCENT).bold(),
            ));
        } else {
            spans.push(Span::styled(label, Style::default().fg(DIM)));
        }
        spans.push(Span::raw(" "));
    }
    let full_usage = process_usage_text(ui.process_usage.as_deref(), false);
    let compact_usage = process_usage_text(ui.process_usage.as_deref(), true);
    let usage_text =
        if spans_width(&spans) + full_usage.chars().count() + 2 <= tabs_area.width as usize {
            full_usage
        } else {
            compact_usage
        };
    let usage_width = usage_text.chars().count().min(tabs_area.width as usize) as u16;
    let [tabs_left, usage_right] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(usage_width)]).areas(tabs_area);
    frame.render_widget(Paragraph::new(Line::from(spans)), tabs_left);
    frame.render_widget(
        Paragraph::new(Span::styled(
            usage_text,
            Style::default().fg(Color::Indexed(81)),
        ))
        .alignment(Alignment::Right),
        usage_right,
    );

    let mut left: Vec<Span> = vec![
        Span::styled(
            format!(" {}", ui.dash.window_label),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            format!("  |  host: {}", crate::host_label(ui.state.host.as_deref())),
            Style::default().fg(DIM),
        ),
    ];
    if let Some(headline) = &ui.dash.headline {
        left.push(Span::styled(
            format!("  |  {}", headline),
            Style::default().fg(Color::White),
        ));
    }
    if let Some(session_id) = &ui.dash.session_id {
        left.push(Span::styled(
            format!("  |  session: {session_id}"),
            Style::default().fg(ACCENT),
        ));
    }

    // Leading spacer keeps a visible gap when the left side is truncated.
    let mut full_right: Vec<Span> = vec![Span::raw("  ")];
    if let Some(sync) = ui.sync_status {
        full_right.push(Span::styled(
            format!("{}  |  ", sync),
            Style::default().fg(DIM),
        ));
    }
    if !ui.dash.window_complete {
        full_right.push(Span::styled(
            "history: loading  |  ",
            Style::default().fg(Color::Indexed(143)),
        ));
    }
    let (integrity_text, integrity_color) = integrity_status_display(ui.state.integrity_status);
    full_right.push(Span::styled(
        integrity_text,
        Style::default().fg(integrity_color),
    ));
    let countdown = crate::formatting::format_countdown(ui.refresh_in);
    full_right.push(Span::styled(
        format!("  |  refresh in {countdown} "),
        Style::default().fg(DIM),
    ));

    let right_spans = if spans_width(&full_right) <= status_area.width as usize {
        full_right
    } else {
        let (integrity_text, integrity_color) = integrity_status_compact(ui.state.integrity_status);
        let refresh_text = format!("refresh:{countdown}");
        let core_width = integrity_text.chars().count() + refresh_text.chars().count() + 3;
        let mut occupied = core_width;
        let mut compact: Vec<Span> = Vec::new();

        if let Some(sync) = ui.sync_status {
            let width = sync.chars().count() + 3;
            if occupied + width <= status_area.width as usize {
                compact.push(Span::styled(sync, Style::default().fg(DIM)));
                compact.push(Span::raw(" | "));
                occupied += width;
            }
        }
        if !ui.dash.window_complete {
            let history = "history:load";
            let width = history.len() + 3;
            if occupied + width <= status_area.width as usize {
                compact.push(Span::styled(
                    history,
                    Style::default().fg(Color::Indexed(143)),
                ));
                compact.push(Span::raw(" | "));
            }
        }
        compact.push(Span::styled(
            integrity_text,
            Style::default().fg(integrity_color),
        ));
        compact.push(Span::raw(" | "));
        compact.push(Span::styled(refresh_text, Style::default().fg(DIM)));
        compact
    };

    // The right side keeps its full width; the left side truncates into
    // whatever remains.
    let right_width = spans_width(&right_spans).min(status_area.width as usize) as u16;
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(right_width)])
            .areas(status_area);
    frame.render_widget(Paragraph::new(Line::from(left)), left_area);
    frame.render_widget(
        Paragraph::new(Line::from(right_spans)).alignment(Alignment::Right),
        right_area,
    );
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|span| span.content.chars().count()).sum()
}

fn integrity_status_display(status: IntegrityStatus) -> (String, Color) {
    match status {
        IntegrityStatus::Unavailable => ("integrity: unavailable".to_string(), DIM),
        IntegrityStatus::Pending => ("integrity: pending".to_string(), Color::Indexed(143)),
        IntegrityStatus::Checking { percent } => (
            format!("integrity: checking {percent}%"),
            Color::Indexed(143),
        ),
        IntegrityStatus::Checked { duration } => (
            format!("integrity: ok ({:.1}s)", duration.as_secs_f64()),
            Color::Indexed(108),
        ),
        IntegrityStatus::Failed => ("integrity: FAILED".to_string(), Color::Indexed(203)),
    }
}

fn integrity_status_compact(status: IntegrityStatus) -> (String, Color) {
    match status {
        IntegrityStatus::Unavailable => ("integrity:n/a".to_string(), DIM),
        IntegrityStatus::Pending => ("integrity:pending".to_string(), Color::Indexed(143)),
        IntegrityStatus::Checking { percent } => {
            (format!("integrity:{percent}%"), Color::Indexed(143))
        }
        IntegrityStatus::Checked { .. } => ("integrity:ok".to_string(), Color::Indexed(108)),
        IntegrityStatus::Failed => ("integrity:FAIL".to_string(), Color::Indexed(203)),
    }
}

fn draw_charts(frame: &mut Frame, area: Rect, ui: &Ui) {
    let charts = &ui.dash.charts;
    if charts.is_empty() {
        return;
    }
    let constraints: Vec<Constraint> = charts
        .iter()
        .map(|_| Constraint::Ratio(1, charts.len() as u32))
        .collect();
    let chunks = Layout::vertical(constraints).split(area);
    let last = charts.len() - 1;
    for (i, (chart, chunk)) in charts.iter().zip(chunks.iter()).enumerate() {
        let bottom = (i == last).then_some(ui.dash.span_label.as_str());
        draw_chart(frame, *chunk, chart, bottom);
    }
}

const SEPARATOR_COLOR: Color = Color::Indexed(238);
/// Left gutter for the y-axis labels: 5-char value + tick mark + space.
const Y_GUTTER: u16 = 7;
/// Cubic segments per source interval. This rounds visual corners while
/// keeping the underlying chart buckets and values unchanged.
const CURVE_SUBDIVISIONS: usize = 4;

/// Interpolate uniformly spaced chart samples with a monotone cubic curve.
/// The result passes through every original point and is clamped to each
/// segment's endpoints, so a visual smoothing pass cannot invent a higher
/// peak or a negative token count.
fn smoothed_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let slopes: Vec<f64> = points
        .windows(2)
        .map(|pair| pair[1].1 - pair[0].1)
        .collect();
    let mut tangents = Vec::with_capacity(points.len());
    tangents.push(slopes[0]);
    for pair in slopes.windows(2) {
        let (previous, next) = (pair[0], pair[1]);
        let tangent = if previous * next <= 0.0 {
            0.0
        } else {
            2.0 * previous * next / (previous + next)
        };
        tangents.push(tangent);
    }
    tangents.push(*slopes.last().unwrap_or(&0.0));

    let mut smoothed = Vec::with_capacity((points.len() - 1) * CURVE_SUBDIVISIONS + 1);
    smoothed.push(points[0]);
    for index in 0..points.len() - 1 {
        let (x0, y0) = points[index];
        let (x1, y1) = points[index + 1];
        for step in 1..=CURVE_SUBDIVISIONS {
            if step == CURVE_SUBDIVISIONS {
                smoothed.push((x1, y1));
                continue;
            }
            let t = step as f64 / CURVE_SUBDIVISIONS as f64;
            let t2 = t * t;
            let t3 = t2 * t;
            let y = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
                + (t3 - 2.0 * t2 + t) * tangents[index]
                + (-2.0 * t3 + 3.0 * t2) * y1
                + (t3 - t2) * tangents[index + 1];
            smoothed.push((x0 + (x1 - x0) * t, y.clamp(y0.min(y1), y0.max(y1))));
        }
    }
    smoothed
}

/// Write `text` into `buf` left-anchored at `start`, but only when the target
/// span (plus a one-cell gap on both sides) is still empty, so overlapping
/// labels are dropped instead of colliding.
fn overlay_at(buf: &mut [char], start: usize, text: &str) {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() || buf.is_empty() {
        return;
    }
    let start = start.min(buf.len().saturating_sub(chars.len()));
    let end = start + chars.len();
    if end > buf.len() {
        return;
    }
    let guard_start = start.saturating_sub(1);
    let guard_end = (end + 1).min(buf.len());
    if buf[guard_start..guard_end].iter().any(|c| *c != ' ') {
        return;
    }
    buf[start..end].copy_from_slice(&chars);
}

/// Place a segment's head/date label pair centered at `mid`, writing both
/// rows only when the combined span is free on the head row. Widest segments
/// are placed first by the caller, so a narrow partial edge segment drops its
/// own label instead of stealing the row from a full segment.
fn place_segment_pair(head: &mut [char], date: &mut [char], mid: usize, h: &str, d: &str) {
    let h_chars: Vec<char> = h.chars().collect();
    let d_chars: Vec<char> = d.chars().collect();
    let span = h_chars.len().max(d_chars.len());
    if span == 0 || head.len() < span {
        return;
    }
    let start = mid.saturating_sub(span / 2).min(head.len() - span);
    let guard_start = start.saturating_sub(1);
    let guard_end = (start + span + 1).min(head.len());
    if head[guard_start..guard_end].iter().any(|c| *c != ' ') {
        return;
    }
    let h_start = start + (span - h_chars.len()) / 2;
    head[h_start..h_start + h_chars.len()].copy_from_slice(&h_chars);
    let d_start = start + (span - d_chars.len()) / 2;
    date[d_start..d_start + d_chars.len()].copy_from_slice(&d_chars);
}

/// Render one axis-aligned text row: `indent` blank columns (the y gutter),
/// then the prebuilt character row.
fn render_axis_row(frame: &mut Frame, area: Rect, indent: u16, row: Vec<char>, color: Color) {
    let text: String = row.into_iter().collect();
    let line = Line::from(vec![
        Span::raw(" ".repeat(indent as usize)),
        Span::styled(text, Style::default().fg(color)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_chart(frame: &mut Frame, area: Rect, chart: &ChartData, bottom_label: Option<&str>) {
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(
            format!(" {} ", chart.title),
            Style::default().fg(ACCENT).bold(),
        ));
    if let Some(label) = bottom_label {
        block = block.title_bottom(
            Line::from(Span::styled(
                format!(" {} ", label),
                Style::default().fg(DIM),
            ))
            .alignment(Alignment::Right),
        );
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if chart.len < 2 || chart.series.is_empty() {
        frame.render_widget(
            Paragraph::new("Not enough data points for chart.").alignment(Alignment::Center),
            inner,
        );
        return;
    }
    if inner.width < Y_GUTTER + 12 || inner.height < 6 {
        return;
    }

    let show_x_labels = bottom_label.is_some();
    // The Month header line already names the year-month, so the per-segment
    // date line is redundant and skipped (same rule as the plain chart).
    let date_row = chart.granularity != ChartGranularity::Month;

    let mut constraints = vec![Constraint::Length(1)];
    if date_row {
        constraints.push(Constraint::Length(1));
    }
    constraints.push(Constraint::Min(3));
    constraints.push(Constraint::Length(1));
    if show_x_labels {
        constraints.push(Constraint::Length(1));
    }
    constraints.push(Constraint::Length(1));
    let rows = Layout::vertical(constraints).split(inner);
    let mut row_iter = rows.iter().copied();
    let head_area = row_iter.next().unwrap();
    let date_area = if date_row { row_iter.next() } else { None };
    let plot_area = row_iter.next().unwrap();
    let axis_area = row_iter.next().unwrap();
    let xlabel_area = if show_x_labels { row_iter.next() } else { None };
    let legend_area = row_iter.next().unwrap();

    let [y_area, canvas_area] =
        Layout::horizontal([Constraint::Length(Y_GUTTER), Constraint::Min(4)]).areas(plot_area);
    let width = canvas_area.width as usize;
    let len = chart.len;
    let col_of = |x: f64| {
        (((x + 0.5) / len as f64) * width as f64)
            .round()
            .clamp(0.0, width.saturating_sub(1) as f64) as usize
    };

    // Y axis: a value label on every row, densest useful tick spacing.
    let top = (chart.max_y * 1.05).max(1.0);
    let rows_n = plot_area.height.max(2) as usize;
    let y_lines: Vec<Line> = (0..rows_n)
        .map(|r| {
            let value = top * (rows_n - 1 - r) as f64 / (rows_n - 1) as f64;
            Line::from(Span::styled(
                format!("{} \u{2524}", format_y_axis_value(value)),
                Style::default().fg(DIM),
            ))
        })
        .collect();
    frame.render_widget(Paragraph::new(y_lines), y_area);

    // Plot: granularity separators first, series lines drawn over them.
    let canvas = Canvas::default()
        .marker(symbols::Marker::Braille)
        .x_bounds([-0.5, len as f64 - 0.5])
        .y_bounds([0.0, top])
        .paint(|ctx| {
            for seg in chart.segments.iter().skip(1) {
                let x = seg.start as f64 - 0.5;
                ctx.draw(&CanvasLine {
                    x1: x,
                    y1: 0.0,
                    x2: x,
                    y2: top,
                    color: SEPARATOR_COLOR,
                });
            }
            ctx.layer();
            for series in &chart.series {
                let color = Color::Indexed(series.color);
                let points = smoothed_points(&series.points);
                for pair in points.windows(2) {
                    ctx.draw(&CanvasLine {
                        x1: pair[0].0,
                        y1: pair[0].1,
                        x2: pair[1].0,
                        y2: pair[1].1,
                        color,
                    });
                }
            }
        });
    frame.render_widget(canvas, canvas_area);

    // Segment header: "Wed : 5.45B" over " 07 / 22", centered per segment.
    // The compact/skip decision looks at inner segments only, so a narrow
    // partial segment at either window edge never suppresses the whole row.
    let seg_cells = |s: &crate::tui::data::Segment| {
        ((s.end - s.start + 1) as f64 / len as f64 * width as f64) as usize
    };
    let inner_min = if chart.segments.len() > 2 {
        chart.segments[1..chart.segments.len() - 1]
            .iter()
            .map(seg_cells)
            .min()
            .unwrap_or(0)
    } else {
        chart.segments.iter().map(seg_cells).min().unwrap_or(0)
    };
    if inner_min >= 7 {
        let compact = inner_min < 13;
        let mut head = vec![' '; width];
        let mut date = vec![' '; width];
        // Widest segments first: full segments keep their labels, narrow
        // edge segments only render when there is room left.
        let mut order: Vec<usize> = (0..chart.segments.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(seg_cells(&chart.segments[i])));
        for &i in &order {
            let seg = &chart.segments[i];
            let (h, d) = charts::segment_label(&seg.anchor, chart.granularity, seg.total, compact);
            let mid = col_of((seg.start + seg.end) as f64 / 2.0);
            place_segment_pair(&mut head, &mut date, mid, h.trim(), d.trim());
        }
        render_axis_row(frame, head_area, Y_GUTTER, head, Color::White);
        if let Some(area) = date_area {
            render_axis_row(frame, area, Y_GUTTER, date, DIM);
        }
    }

    // Axis rule with a tick under every granularity separator.
    let mut axis = vec!['\u{2500}'; width];
    for seg in chart.segments.iter().skip(1) {
        axis[col_of(seg.start as f64 - 0.5)] = '\u{2534}';
    }
    let axis_line = Line::from(vec![
        Span::raw(" ".repeat(Y_GUTTER as usize - 1)),
        Span::styled(
            format!("\u{2514}{}", axis.into_iter().collect::<String>()),
            Style::default().fg(DIM),
        ),
    ]);
    frame.render_widget(Paragraph::new(axis_line), axis_area);

    // X tick labels at wall-clock-aligned positions.
    if let Some(area) = xlabel_area {
        let mut row = vec![' '; width];
        for (idx, label) in &chart.x_ticks {
            overlay_at(&mut row, col_of(*idx as f64), label);
        }
        render_axis_row(frame, area, Y_GUTTER, row, DIM);
    }

    draw_chart_legend(frame, legend_area, chart, show_x_labels);
}

/// One legend row per chart: colored line markers with series names, plus the
/// window-navigation hint right-aligned on the bottom chart.
fn draw_chart_legend(frame: &mut Frame, area: Rect, chart: &ChartData, with_hint: bool) {
    const HINT: &str = "PgUp/PgDn: page | Left/Right: move | +/-: zoom";
    let hint_width = if with_hint {
        HINT.chars().count() as u16 + 2
    } else {
        0
    };
    let [legend_area, hint_area] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(hint_width)]).areas(area);

    let budget = legend_area.width as usize;
    let mut used = Y_GUTTER as usize;
    let mut spans: Vec<Span> = vec![Span::raw(" ".repeat(Y_GUTTER as usize))];
    let legend_items = chart_legend_items(chart);
    for (i, (series, share)) in legend_items.iter().enumerate() {
        let label = share.map_or_else(
            || series.name.clone(),
            |share| format!("{} ({share}%)", series.name),
        );
        let item_width = 3 + label.chars().count() + 2;
        if used + item_width + 4 > budget {
            spans.push(Span::styled(
                format!("+{}", legend_items.len() - i),
                Style::default().fg(DIM),
            ));
            break;
        }
        spans.push(Span::styled(
            "\u{2500}\u{2500} ",
            Style::default().fg(Color::Indexed(series.color)),
        ));
        spans.push(Span::raw(format!("{label}  ")));
        used += item_width;
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), legend_area);
    if with_hint {
        frame.render_widget(
            Paragraph::new(Span::styled(HINT, Style::default().fg(DIM)))
                .alignment(Alignment::Right),
            hint_area,
        );
    }
}

fn chart_legend_items(chart: &ChartData) -> Vec<(&crate::tui::data::Series, Option<u8>)> {
    let Some(all) = chart.series.iter().find(|series| series.name == "All") else {
        return chart.series.iter().map(|series| (series, None)).collect();
    };
    let mut tools: Vec<(&crate::tui::data::Series, f64)> = chart
        .series
        .iter()
        .filter(|series| series.name != "All")
        .map(|series| (series, series.points.iter().map(|point| point.1).sum()))
        .collect();
    tools.sort_by(|(left, left_total), (right, right_total)| {
        right_total
            .total_cmp(left_total)
            .then_with(|| left.name.cmp(&right.name))
    });
    let total: f64 = tools.iter().map(|(_, value)| value).sum();
    let mut items: Vec<_> = tools
        .into_iter()
        .map(|(series, value)| {
            let share = if total > 0.0 {
                (value * 100.0 / total).round().clamp(0.0, 100.0) as u8
            } else {
                0
            };
            (series, Some(share))
        })
        .collect();
    items.push((all, None));
    items
}

fn draw_footer(frame: &mut Frame, area: Rect, ui: &Ui) {
    let prompt = Span::styled("> ", Style::default().fg(ACCENT).bold());
    let line = if !ui.input.is_empty() {
        Line::from(vec![prompt, Span::raw(ui.input.snapshot().to_string())])
    } else if let Some(notice) = ui.notice {
        Line::from(vec![
            prompt,
            Span::styled(notice.to_string(), Style::default().fg(Color::Indexed(179))),
        ])
    } else {
        Line::from(vec![
            prompt,
            Span::styled(
                format!(
                    "ai-usage by SihaoLiu, v{}, refresh in {}, enter h or help for usage",
                    env!("CARGO_PKG_VERSION"),
                    crate::formatting::format_countdown(ui.refresh_in)
                ),
                Style::default().fg(Color::Indexed(240)),
            ),
        ])
    };
    frame.render_widget(Paragraph::new(line), area);
    if ui.input.is_empty() || ui.input.cursor_chars() > 0 {
        let x = area.x + 2 + ui.input.cursor_chars() as u16;
        frame.set_cursor_position(Position::new(x.min(area.right().saturating_sub(1)), area.y));
    }
}

fn draw_help(frame: &mut Frame, area: Rect, view: HelpView) {
    let (title, lines, footer) = match view {
        HelpView::Index => {
            let lines: Vec<Line> = help_topics()
                .iter()
                .map(|topic| {
                    Line::from(vec![
                        Span::styled(
                            format!("  {:<24}", topic.invocation),
                            Style::default().fg(ACCENT),
                        ),
                        Span::raw(topic.summary),
                    ])
                })
                .collect();
            (
                " Commands ".to_string(),
                lines,
                " h <topic> for details (e.g. h view)  |  Esc: close ",
            )
        }
        HelpView::Topic(idx) => {
            let topic = &help_topics()[idx.min(help_topics().len() - 1)];
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("  Command:  ", Style::default().fg(DIM)),
                    Span::styled(topic.invocation, Style::default().fg(ACCENT)),
                ]),
                Line::from(""),
            ];
            lines.extend(
                topic
                    .detail
                    .iter()
                    .map(|text| Line::from(vec![Span::raw("  "), Span::raw(*text)])),
            );
            (
                format!(" help: {} ", topic.name),
                lines,
                " Esc: back to index  |  h <topic>: jump ",
            )
        }
    };

    let content_width = lines
        .iter()
        .map(|l| l.width() as u16)
        .max()
        .unwrap_or(0)
        .max(footer.chars().count() as u16)
        + 4;
    let height = (lines.len() as u16 + 4).min(area.height.saturating_sub(2));
    let width = content_width.clamp(46, area.width.saturating_sub(4));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(title, Style::default().fg(ACCENT).bold()))
                .title_bottom(Line::from(Span::styled(footer, Style::default().fg(DIM)))),
        ),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AllPricing, SubscriptionFees};
    use crate::model_id::Vendor;
    use crate::process_usage::{ProcessUsage, ProcessUsageDisplay};
    use crate::table_view::{DataRow, DisplayRow, RowMetrics, TableView};
    use crate::time_utils::TimeWindow;
    use crate::tui::data::{ChartData, Segment, Series};
    use chrono::TimeZone;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::collections::HashMap;

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    fn row_metrics(cache_hit: i64) -> RowMetrics {
        RowMetrics {
            count: 1,
            cache_hit,
            prefill: 1_390_000,
            decoding: 480_000,
            cache_hit_cost: 10.0,
            prefill_cost: 2.0,
            decoding_cost: 3.0,
        }
    }

    fn data_row(
        label: &str,
        vendor: Vendor,
        vendor_label: &str,
        metrics: RowMetrics,
    ) -> DisplayRow {
        DisplayRow::Data(Box::new(DataRow {
            vendor,
            vendor_label: vendor_label.to_string(),
            model_label: label.to_string(),
            model_raw: label.to_ascii_lowercase().replace(' ', "-"),
            harness_label: "Claude Code".to_string(),
            harness_short: "CC".to_string(),
            metrics,
        }))
    }

    fn dashboard(view: TableView, rows: Vec<DisplayRow>, totals: RowMetrics) -> Dashboard {
        Dashboard {
            tool: Tool::All,
            view,
            window_label: String::new(),
            window_complete: true,
            has_visible_data: true,
            session_id: None,
            model_stats: Vec::new(),
            rows,
            totals,
            summary: Default::default(),
            insight: None,
            headline: None,
            span_label: String::new(),
            charts: Vec::new(),
        }
    }

    fn render_table(width: u16, height: u16, dash: &Dashboard) -> Terminal<TestBackend> {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_table(frame, frame.area(), dash))
            .expect("draw");
        terminal
    }

    fn line_containing(text: &str, needle: &str) -> String {
        text.lines()
            .find(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("missing {needle:?} in:\n{text}"))
            .to_string()
    }

    fn line_index_containing(text: &str, needle: &str) -> u16 {
        text.lines()
            .position(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("missing {needle:?} in:\n{text}")) as u16
    }

    fn char_index(text: &str, needle: &str) -> usize {
        let byte = text
            .find(needle)
            .unwrap_or_else(|| panic!("missing {needle:?} in {text:?}"));
        text[..byte].chars().count()
    }

    fn char_positions(text: &str, needle: char) -> Vec<usize> {
        text.chars()
            .enumerate()
            .filter_map(|(index, ch)| (ch == needle).then_some(index))
            .collect()
    }

    #[test]
    fn process_usage_is_right_aligned_on_the_title_line() {
        let dash = dashboard(TableView::Flat, Vec::new(), RowMetrics::default());
        let state = AppState {
            tool: "all".to_string(),
            table_view: TableView::Flat,
            host: None,
            session_id: None,
            local_host_id: None,
            days: 3,
            time_window: TimeWindow::rolling_days(3),
            monitor_interval: 3600,
            pricing: AllPricing::load_raw().finalize(),
            subscription_fees: SubscriptionFees::default(),
            fee_env_path: std::path::PathBuf::from(".fee.env"),
            version_cache: HashMap::new(),
            all_tool_prompt: None,
            raw_cache: None,
            raw_cache_last_used_at: None,
            raw_refresh: None,
            integrity_status: IntegrityStatus::Checked {
                duration: std::time::Duration::from_millis(8_500),
            },
            integrity_started_at: None,
        };
        let input = InputLine::new();
        let ui = Ui {
            dash: &dash,
            state: &state,
            input: &input,
            notice: None,
            sync_status: Some("Sync: checked just now"),
            process_usage: Some(std::sync::Arc::new(ProcessUsageDisplay::new(
                ProcessUsage {
                    cpu_percent: 12.34,
                    memory_bytes: 1_331_438_182,
                },
            ))),
            refresh_in: std::time::Duration::from_secs(3_596),
            help: None,
        };
        let backend = TestBackend::new(240, 2);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| draw_header(frame, frame.area(), &ui))
            .expect("draw");

        let text = buffer_text(&terminal);
        let title = line_containing(&text, "ai-usage CPU: 12.3%");
        let status = line_containing(&text, "integrity: ok (8.5s)");
        assert!(title.contains(env!("CARGO_PKG_VERSION")), "{title}");
        assert!(
            title.ends_with("ai-usage CPU: 12.3%  |  Mem: 1.24 GiB"),
            "{title}"
        );
        assert!(status.contains("Sync: checked just now"), "{status}");
        assert!(status.contains("refresh in 00:59:56"), "{status}");
        assert!(!status.contains("CPU"), "{status}");
        assert!(!status.contains("Mem"), "{status}");

        let backend = TestBackend::new(68, 2);
        let mut terminal = Terminal::new(backend).expect("narrow terminal");
        terminal
            .draw(|frame| draw_header(frame, frame.area(), &ui))
            .expect("draw narrow header");

        let text = buffer_text(&terminal);
        let title = line_containing(&text, "ai-usage CPU:12.3%");
        let status = line_containing(&text, "integrity:ok");
        assert!(title.ends_with("ai-usage CPU:12.3% | Mem:1.24G"), "{title}");
        assert!(text.contains("integrity:ok"), "{text}");
        assert!(text.contains("refresh:00:59:56"), "{text}");
        assert!(!status.contains("CPU"), "{status}");
        assert!(!status.contains("Mem"), "{status}");
    }

    #[test]
    fn wide_table_uses_exact_values_and_fixed_percentage_slots() {
        let alpha = RowMetrics {
            count: 418,
            cache_hit: 9_730_000_000,
            prefill: 1_390_000,
            decoding: 480_000,
            cache_hit_cost: 156.94,
            prefill_cost: 4.25,
            decoding_cost: 1.75,
        };
        let beta = RowMetrics {
            count: 1_540,
            cache_hit: 385_000_000,
            prefill: 11_600_000,
            decoding: 1_680_000,
            cache_hit_cost: 921.89,
            prefill_cost: 12.0,
            decoding_cost: 8.0,
        };
        let mut totals = alpha;
        totals.add(&beta);
        let dash = dashboard(
            TableView::Flat,
            vec![
                data_row("Alpha", Vendor::Anthropic, "Anthropic", alpha),
                data_row("Beta", Vendor::Anthropic, "", beta),
            ],
            totals,
        );

        let terminal = render_table(360, 8, &dash);
        let text = buffer_text(&terminal);
        let alpha_line = line_containing(&text, "Alpha");
        let beta_line = line_containing(&text, "Beta");
        let total_line = line_containing(&text, "TOTAL");

        assert!(alpha_line.contains("9,730,000,000"), "{alpha_line}");
        assert!(beta_line.contains("385,000,000"), "{beta_line}");
        assert_eq!(
            char_positions(&alpha_line, '↑'),
            char_positions(&beta_line, '↑')
        );
        assert_eq!(
            char_positions(&alpha_line, '%'),
            char_positions(&beta_line, '%')
        );

        let cache_arrow = char_positions(&alpha_line, '↑')[1];
        let total_value = "10,115,000,000";
        let total_value_end = char_index(&total_line, total_value) + total_value.len() - 1;
        assert_eq!(
            total_value_end + 2,
            cache_arrow,
            "{total_line}\n{alpha_line}"
        );
    }

    #[test]
    fn compact_units_share_one_column_and_have_scale_colors() {
        let cases = [
            ("Scale T", 1_234_000_000_000, 'T', Color::Indexed(177)),
            ("Scale B", 9_730_000_000, 'B', Color::Indexed(214)),
            ("Scale M", 385_000_000, 'M', Color::Indexed(81)),
            ("Scale K", 752_000, 'K', Color::Indexed(108)),
        ];
        let mut totals = RowMetrics::default();
        let rows = cases
            .iter()
            .enumerate()
            .map(|(index, (label, value, _, _))| {
                let metrics = row_metrics(*value);
                totals.add(&metrics);
                data_row(
                    label,
                    Vendor::Anthropic,
                    if index == 0 { "Anthropic" } else { "" },
                    metrics,
                )
            })
            .collect();
        let dash = dashboard(TableView::Flat, rows, totals);

        let terminal = render_table(170, 9, &dash);
        let text = buffer_text(&terminal);
        let buffer = terminal.backend().buffer();
        let mut unit_column = None;

        for (label, _, unit, color) in cases {
            let line = line_containing(&text, label);
            let y = line_index_containing(&text, label);
            let cache_arrow = char_positions(&line, '↑')[1];
            let unit_x = cache_arrow - 2;
            assert_eq!(line.chars().nth(unit_x), Some(unit), "{line}");
            assert_eq!(buffer[(unit_x as u16, y)].fg, color, "{line}");
            assert_eq!(*unit_column.get_or_insert(unit_x), unit_x, "{line}");
        }
    }

    #[test]
    fn table_rows_have_distinct_visual_hierarchy() {
        let alpha = row_metrics(9_730_000_000);
        let beta = row_metrics(385_000_000);
        let gamma = row_metrics(752_000);
        let mut anthropic_total = alpha;
        anthropic_total.add(&beta);
        let mut totals = anthropic_total;
        totals.add(&gamma);
        let dash = dashboard(
            TableView::Vendor,
            vec![
                DisplayRow::GroupHeader {
                    vendor: "Anthropic".to_string(),
                },
                data_row("Alpha", Vendor::Anthropic, "", alpha),
                data_row("Beta", Vendor::Anthropic, "", beta),
                DisplayRow::Subtotal {
                    vendor: "Anthropic".to_string(),
                    metrics: anthropic_total,
                },
                DisplayRow::GroupHeader {
                    vendor: "OpenAI".to_string(),
                },
                data_row("Gamma", Vendor::OpenAI, "", gamma),
            ],
            totals,
        );

        let terminal = render_table(180, 12, &dash);
        let text = buffer_text(&terminal);
        let buffer = terminal.backend().buffer();
        let background_at = |needle: &str| {
            let line = line_containing(&text, needle);
            let x = char_index(&line, needle) as u16;
            let y = line_index_containing(&text, needle);
            buffer[(x, y)].bg
        };
        let header_line = text
            .lines()
            .find(|line| line.contains("Vendor") && line.contains("Cache Hit"))
            .expect("table header");
        let header_x = char_index(header_line, "Vendor") as u16;
        let header_y = text
            .lines()
            .position(|line| line == header_line)
            .expect("table header index") as u16;

        assert_eq!(buffer[(header_x, header_y)].bg, Color::Indexed(236));
        assert_eq!(background_at("Anthropic"), Color::Indexed(235));
        assert_eq!(background_at("Beta"), Color::Indexed(233));
        line_containing(&text, "Anthropic total");
        let subtotal_y = line_index_containing(&text, "Beta") + 1;
        assert_eq!(buffer[(1, subtotal_y)].bg, Color::Indexed(234));
        assert_eq!(background_at("OpenAI"), Color::Indexed(235));
        assert_eq!(background_at("TOTAL"), Color::Indexed(237));
    }

    #[test]
    fn narrow_table_titles_and_summary_use_complete_fields() {
        let metrics = row_metrics(9_730_000_000);
        let mut dash = dashboard(
            TableView::Flat,
            vec![data_row("Alpha", Vendor::Anthropic, "Anthropic", metrics)],
            metrics,
        );
        dash.summary.daily = 3_430.09;
        dash.summary.weekly = 24_010.63;
        dash.summary.monthly = 102_902.70;
        dash.summary.savings = 102_502.70;
        dash.summary.subscription_rate = 0.003;

        for width in [68, 72, 80] {
            let terminal = render_table(width, 7, &dash);
            let text = buffer_text(&terminal);
            let top = text.lines().next().expect("top border");
            let bottom = line_containing(&text, "Daily");

            assert!(top.contains("Usage / API Cost"), "{width}: {top}");
            assert!(top.contains("↑ share of column"), "{width}: {top}");
            assert!(!top.contains("share of row"), "{width}: {top}");
            assert!(bottom.contains("Daily $3.43K"), "{width}: {bottom}");
            assert!(bottom.contains("Weekly $24.01K"), "{width}: {bottom}");
            assert!(bottom.contains("Monthly $102.9K"), "{width}: {bottom}");
            assert!(bottom.contains("Saving $102.5K"), "{width}: {bottom}");
            assert!(!bottom.contains("/ MTok"), "{width}: {bottom}");
        }
    }

    #[test]
    fn wide_table_bounds_model_id_and_expands_metric_columns() {
        let dash = Dashboard {
            tool: Tool::All,
            view: TableView::Flat,
            window_label: String::new(),
            window_complete: true,
            has_visible_data: true,
            session_id: None,
            model_stats: Vec::new(),
            rows: vec![DisplayRow::Data(Box::new(DataRow {
                vendor: Vendor::Anthropic,
                vendor_label: "Anthropic".to_string(),
                model_label: "Opus 4.8".to_string(),
                model_raw: "claude-opus-4-8".to_string(),
                harness_label: "Claude Code".to_string(),
                harness_short: "CC".to_string(),
                metrics: RowMetrics::default(),
            }))],
            totals: RowMetrics::default(),
            summary: Default::default(),
            insight: None,
            headline: None,
            span_label: String::new(),
            charts: Vec::new(),
        };
        let backend = TestBackend::new(240, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_table(frame, frame.area(), &dash))
            .expect("draw");
        let text = buffer_text(&terminal);
        let header = text
            .lines()
            .find(|line| line.contains("Model Id") && line.contains("Cache Hit"))
            .expect("table header");
        let model_id = header.find("Model Id").expect("Model Id column");
        let harness = header.find("Harness").expect("Harness column");
        let messages = header.find("Msgs").expect("Msgs column");
        let cache_hit = header.find("Cache Hit").expect("Cache Hit column");

        assert!(
            harness - model_id <= 24,
            "Model Id absorbed the wide-table surplus:\n{header}"
        );
        assert!(
            cache_hit - messages >= 18,
            "metric columns did not receive the wide-table surplus:\n{header}"
        );
    }

    /// Regression: a 24-day window at Week granularity ends mid-window with a
    /// one-bucket Monday segment. The narrow edge segment must not suppress
    /// the whole segment-header row.
    #[test]
    fn narrow_edge_segment_keeps_header_of_full_segments() {
        let monday = |day: u32| {
            chrono::Local
                .with_ymd_and_hms(2026, 6, day, 0, 0, 0)
                .unwrap()
        };
        // 73 buckets of 8h across 2026-06-05 .. 2026-06-29 (Mondays: 8/15/22/29).
        let len = 73;
        let points: Vec<(f64, f64)> = (0..len).map(|i| (i as f64, 1.0)).collect();
        let chart = ChartData {
            title: "test".to_string(),
            series: vec![Series {
                name: "All".to_string(),
                color: 226,
                points,
            }],
            max_y: 2.0,
            len,
            granularity: ChartGranularity::Week,
            segments: vec![
                Segment {
                    start: 0,
                    end: 8,
                    total: 9.0,
                    anchor: monday(1),
                },
                Segment {
                    start: 9,
                    end: 29,
                    total: 21.0,
                    anchor: monday(8),
                },
                Segment {
                    start: 30,
                    end: 50,
                    total: 21.0,
                    anchor: monday(15),
                },
                Segment {
                    start: 51,
                    end: 71,
                    total: 21.0,
                    anchor: monday(22),
                },
                // The one-bucket partial Monday at the window's right edge.
                Segment {
                    start: 72,
                    end: 72,
                    total: 1.0,
                    anchor: monday(29),
                },
            ],
            x_ticks: vec![(0, "06/05".to_string()), (36, "06/17".to_string())],
        };

        let backend = TestBackend::new(170, 24);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_chart(frame, frame.area(), &chart, Some("span")))
            .expect("draw");
        let text = buffer_text(&terminal);

        // Full weeks keep their header labels and totals.
        assert!(text.contains("Wk 24"), "missing week header:\n{}", text);
        assert!(text.contains("Wk 25"), "missing week header:\n{}", text);
        assert!(text.contains(" : 21"), "missing segment total:\n{}", text);
        // Separator ticks under each week boundary on the axis rule.
        assert!(text.contains('\u{2534}'), "missing axis tick:\n{}", text);
        // Dense y labels: one tick mark per plot row.
        assert!(
            text.matches('\u{2524}').count() >= 10,
            "sparse y ticks:\n{}",
            text
        );
    }

    #[test]
    fn comparison_legend_shows_descending_consumption_shares() {
        let points = |value| vec![(0.0, value), (1.0, value)];
        let chart = ChartData {
            title: "Total Token Consumption by Tool".to_string(),
            series: vec![
                Series {
                    name: "Codex".to_string(),
                    color: 39,
                    points: points(45.0),
                },
                Series {
                    name: "Claude Code".to_string(),
                    color: 173,
                    points: points(55.0),
                },
                Series {
                    name: "Kimi Code".to_string(),
                    color: 49,
                    points: points(400.0),
                },
                Series {
                    name: "All".to_string(),
                    color: 226,
                    points: points(500.0),
                },
            ],
            max_y: 500.0,
            len: 2,
            granularity: ChartGranularity::Hour,
            segments: Vec::new(),
            x_ticks: Vec::new(),
        };
        let backend = TestBackend::new(180, 12);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| draw_chart(frame, frame.area(), &chart, None))
            .expect("draw");

        let text = buffer_text(&terminal);
        let legend = line_containing(&text, "Kimi Code");
        let kimi = char_index(&legend, "Kimi Code (80%)");
        let claude = char_index(&legend, "Claude Code (11%)");
        let codex = char_index(&legend, "Codex (9%)");
        let all = char_index(&legend, "All");
        assert!(kimi < claude && claude < codex && codex < all, "{legend}");
        assert!(!legend.contains("All ("), "{legend}");
    }

    #[test]
    fn integrity_header_displays_checking_percentage() {
        let (text, _) = integrity_status_display(IntegrityStatus::Checking { percent: 37 });

        assert_eq!(text, "integrity: checking 37%");
    }

    #[test]
    fn smoothed_points_round_corners_without_moving_or_overshooting_samples() {
        let points = vec![(3.0, 0.0), (2.0, 10.0), (1.0, 0.0), (0.0, 4.0)];
        let smoothed = smoothed_points(&points);

        assert!(smoothed.len() > points.len());
        assert_eq!(smoothed.first(), points.first());
        assert_eq!(smoothed.last(), points.last());
        assert!(smoothed.iter().all(|(_, y)| (0.0..=10.0).contains(y)));
        assert!(smoothed.iter().any(|(_, y)| *y > 0.0 && *y < 10.0));
    }
}
