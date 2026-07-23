//! Frame rendering for the ratatui monitor UI.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table};

use crate::charts::{self, ChartGranularity};
use crate::formatting::{format_cost_per_mtok, format_number_compact, format_y_axis_value};
use crate::model_id::Vendor;
use crate::table_view::{DataRow, DisplayRow, RowMetrics, TableView};
use crate::tool::Tool;
use crate::tui::commands::{HelpView, help_topics};
use crate::tui::data::{ChartData, Dashboard};
use crate::tui::input::InputLine;
use crate::{AppState, IntegrityStatus};

pub struct Ui<'a> {
    pub dash: &'a Dashboard,
    pub state: &'a AppState,
    pub input: &'a InputLine,
    pub notice: Option<&'a str>,
    pub sync_status: Option<&'a str>,
    pub refresh_in: std::time::Duration,
    pub help: Option<HelpView>,
}

const MIN_WIDTH: u16 = 68;
const MIN_HEIGHT: u16 = 22;

const DIM: Color = Color::Indexed(245);
const ACCENT: Color = Color::Indexed(51);
const COL_PCT: Color = Color::Indexed(36);
const ROW_PCT: Color = Color::Indexed(179);

fn vendor_color(vendor: Vendor) -> Color {
    match vendor {
        Vendor::Anthropic => Color::Indexed(208),
        Vendor::OpenAI => Color::Indexed(255),
        Vendor::Google => Color::Indexed(39),
        Vendor::Moonshot => Color::Indexed(49),
        Vendor::Zhipu => Color::Indexed(135),
        Vendor::Unknown => Color::Indexed(245),
    }
}

pub fn draw(frame: &mut Frame, ui: &Ui) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        draw_too_small(frame, area);
        return;
    }

    let [header, body, footer] =
        Layout::vertical([Constraint::Length(2), Constraint::Min(10), Constraint::Length(1)])
            .areas(area);

    draw_header(frame, header, ui);

    if !ui.dash.has_source_data || !ui.dash.has_visible_data {
        let message = match ui.dash.session_id.as_deref() {
            Some(session_id) => format!("No usage data found for session {session_id}."),
            None => "No usage data found from any tool.".to_string(),
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
        draw_table(frame, table_area, ui);
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
    frame.render_widget(
        Paragraph::new(text).alignment(Alignment::Center),
        area,
    );
}

fn draw_header(frame: &mut Frame, area: Rect, ui: &Ui) {
    let [tabs_area, status_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);

    let mut spans: Vec<Span> = vec![
        Span::styled(" ai-usage ", Style::default().fg(ACCENT).bold()),
        Span::styled(format!("v{} ", env!("CARGO_PKG_VERSION")), Style::default().fg(DIM)),
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
    frame.render_widget(Paragraph::new(Line::from(spans)), tabs_area);

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
    let mut right_spans: Vec<Span> = vec![Span::raw("  ")];
    if let Some(sync) = ui.sync_status {
        right_spans.push(Span::styled(format!("{}  |  ", sync), Style::default().fg(DIM)));
    }
    let (integrity_text, integrity_color) = match ui.state.integrity_status {
        IntegrityStatus::Checking => ("integrity: checking".to_string(), Color::Indexed(143)),
        IntegrityStatus::Checked { duration } => (
            format!("integrity: ok ({:.1}s)", duration.as_secs_f64()),
            Color::Indexed(108),
        ),
        IntegrityStatus::Failed => ("integrity: FAILED".to_string(), Color::Indexed(203)),
    };
    right_spans.push(Span::styled(integrity_text, Style::default().fg(integrity_color)));
    right_spans.push(Span::styled(
        format!(
            "  |  refresh in {} ",
            crate::formatting::format_countdown(ui.refresh_in)
        ),
        Style::default().fg(DIM),
    ));

    // The right side keeps its full width; the left side truncates into
    // whatever remains.
    let right_width = right_spans
        .iter()
        .map(|s| s.content.chars().count() as u16)
        .sum::<u16>()
        .min(status_area.width);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(right_width)])
            .areas(status_area);
    frame.render_widget(Paragraph::new(Line::from(left)), left_area);
    frame.render_widget(
        Paragraph::new(Line::from(right_spans)).alignment(Alignment::Right),
        right_area,
    );
}

struct Cols {
    vendor: bool,
    /// Raw model id column (the full snapshot id), shown when very wide.
    raw: bool,
    harness: bool,
    harness_full: bool,
    strategy: bool,
    rate: bool,
}

fn visible_cols(width: u16, show_harness: bool) -> Cols {
    if width >= 150 {
        Cols {
            vendor: true,
            // The full layout needs 169 inner cells: 159 fixed/minimum
            // column cells plus ten one-cell gaps. Account for the border
            // around the table as well.
            raw: width >= 171,
            harness: show_harness,
            harness_full: show_harness,
            strategy: true,
            rate: true,
        }
    } else if width >= 138 {
        Cols {
            vendor: true,
            raw: false,
            harness: show_harness,
            harness_full: false,
            strategy: true,
            rate: false,
        }
    } else {
        Cols {
            vendor: false,
            raw: false,
            harness: false,
            harness_full: false,
            strategy: false,
            rate: false,
        }
    }
}

/// Reserve only the width a readable model label needs. The raw model id is
/// the flexible column, preventing an oversized Model column from crowding
/// the descriptive columns and numeric metrics on wide terminals.
fn model_column_width(longest_label: usize) -> u16 {
    (longest_label.saturating_add(2) as u16).clamp(14, 26)
}

fn dashboard_model_column_width(rows: &[DisplayRow]) -> u16 {
    let longest = rows
        .iter()
        .filter_map(|row| match row {
            DisplayRow::Data(data) => Some(data.model_label.chars().count()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    model_column_width(longest)
}

fn right(spans: Vec<Span<'static>>) -> Cell<'static> {
    Cell::from(Line::from(spans).alignment(Alignment::Right))
}

fn qty_cell(value: i64, col_total: i64, row_total: Option<i64>) -> Cell<'static> {
    let mut spans = vec![Span::raw(format_number_compact(value))];
    if col_total > 0 {
        let pct = value as f64 / col_total as f64 * 100.0;
        spans.push(Span::styled(
            format!(" \u{2191}{:.0}%", pct),
            Style::default().fg(COL_PCT),
        ));
    }
    if let Some(total) = row_total
        && total > 0
    {
        let pct = value as f64 / total as f64 * 100.0;
        spans.push(Span::styled(
            format!("\u{00B7}\u{2190}{:.0}%", pct),
            Style::default().fg(ROW_PCT),
        ));
    }
    right(spans)
}

fn cost_text(value: f64) -> String {
    if value.abs() >= 1_000.0 {
        format!("${:.1}K", value / 1_000.0)
    } else {
        format!("${:.2}", value)
    }
}

fn cost_cell(value: f64, total: f64) -> Cell<'static> {
    let mut spans = vec![Span::raw(cost_text(value))];
    if total > 0.0 {
        spans.push(Span::styled(
            format!(" \u{2191}{:.0}%", value / total * 100.0),
            Style::default().fg(COL_PCT),
        ));
    }
    right(spans)
}

/// Build the numeric cells for one row. `col_pct` adds the cyan share-of-
/// column percentages; the TOTAL row passes false since they are always 100%.
fn metric_cells(
    m: &RowMetrics,
    totals: &RowMetrics,
    cols: &Cols,
    col_pct: bool,
    cells: &mut Vec<Cell<'static>>,
) {
    let row_total = m.tokens();
    let col = |total: i64| if col_pct { total } else { 0 };
    cells.push(qty_cell(m.count, col(totals.count), None));
    if cols.strategy {
        cells.push(qty_cell(m.cache_hit, col(totals.cache_hit), Some(row_total)));
        cells.push(qty_cell(m.prefill, col(totals.prefill), Some(row_total)));
        cells.push(qty_cell(m.decoding, col(totals.decoding), Some(row_total)));
    }
    cells.push(qty_cell(row_total, col(totals.tokens()), None));
    cells.push(cost_cell(m.cost(), if col_pct { totals.cost() } else { 0.0 }));
    if cols.rate {
        cells.push(right(vec![Span::styled(
            format_cost_per_mtok(m.cost_per_mtok()),
            Style::default().fg(DIM),
        )]));
    }
}

fn model_cell_text(d: &DataRow, cols: &Cols, show_harness: bool, view: TableView) -> String {
    if !cols.vendor && show_harness && view != TableView::Model {
        format!("{}:{}", d.harness_short, d.model_label)
    } else {
        d.model_label.clone()
    }
}

fn harness_cell_text(d: &DataRow, cols: &Cols) -> String {
    if cols.harness_full {
        d.harness_label.clone()
    } else {
        d.harness_short.clone()
    }
}

fn draw_table(frame: &mut Frame, area: Rect, ui: &Ui) {
    let dash = ui.dash;
    let show_harness = dash.tool.is_all();
    let cols = visible_cols(area.width, show_harness);
    let totals = &dash.totals;
    let model_width = dashboard_model_column_width(&dash.rows);

    let mut header_cells: Vec<Cell> = Vec::new();
    let mut widths: Vec<Constraint> = Vec::new();
    if cols.vendor {
        header_cells.push(Cell::from("Vendor"));
        widths.push(Constraint::Length(9));
    }
    header_cells.push(Cell::from("Model"));
    widths.push(Constraint::Length(model_width));
    if cols.raw {
        header_cells.push(Cell::from("Model Id"));
        widths.push(Constraint::Min(20));
    }
    if cols.harness {
        header_cells.push(Cell::from("Harness"));
        widths.push(Constraint::Length(if cols.harness_full { 14 } else { 11 }));
    }
    header_cells.push(right(vec![Span::raw("Msgs")]));
    widths.push(Constraint::Length(11));
    if cols.strategy {
        for label in ["Cache Hit", "Prefill", "Decode"] {
            header_cells.push(right(vec![Span::raw(label)]));
            widths.push(Constraint::Length(16));
        }
    }
    header_cells.push(right(vec![Span::raw("Total")]));
    widths.push(Constraint::Length(11));
    header_cells.push(right(vec![Span::raw("Cost")]));
    widths.push(Constraint::Length(12));
    if cols.rate {
        header_cells.push(right(vec![Span::raw("$/MTok")]));
        widths.push(Constraint::Length(8));
    }
    let n_cols = header_cells.len();
    let header = Row::new(header_cells).style(Style::default().fg(DIM).add_modifier(Modifier::BOLD));

    let mut rows: Vec<Row> = Vec::new();
    for display_row in &dash.rows {
        match display_row {
            DisplayRow::GroupHeader { vendor } => {
                let color = vendor_color(vendor_by_name(vendor));
                let mut cells = vec![Cell::from(Span::styled(
                    vendor.clone(),
                    Style::default().fg(color).bold(),
                ))];
                cells.resize_with(n_cols, || Cell::from(""));
                rows.push(Row::new(cells));
            }
            DisplayRow::Data(d) => {
                let mut cells: Vec<Cell> = Vec::new();
                if cols.vendor {
                    cells.push(Cell::from(Span::styled(
                        d.vendor_label.clone(),
                        Style::default().fg(vendor_color(d.vendor)).bold(),
                    )));
                }
                cells.push(Cell::from(model_cell_text(d, &cols, show_harness, dash.view)));
                if cols.raw {
                    cells.push(Cell::from(Span::styled(
                        d.model_raw.clone(),
                        Style::default().fg(DIM),
                    )));
                }
                if cols.harness {
                    cells.push(Cell::from(Span::styled(
                        harness_cell_text(d, &cols),
                        Style::default().fg(DIM),
                    )));
                }
                metric_cells(&d.metrics, totals, &cols, true, &mut cells);
                rows.push(Row::new(cells));
            }
            DisplayRow::Subtotal { vendor, metrics } => {
                let mut cells: Vec<Cell> = Vec::new();
                let label = format!("{} total", vendor);
                if cols.vendor {
                    cells.push(Cell::from(""));
                    cells.push(Cell::from(Span::styled(label, Style::default().fg(DIM))));
                } else {
                    cells.push(Cell::from(Span::styled(label, Style::default().fg(DIM))));
                }
                if cols.raw {
                    cells.push(Cell::from(""));
                }
                if cols.harness {
                    cells.push(Cell::from(""));
                }
                metric_cells(metrics, totals, &cols, true, &mut cells);
                rows.push(Row::new(cells).style(Style::default().add_modifier(Modifier::DIM)));
            }
        }
    }

    let mut total_cells: Vec<Cell> = vec![Cell::from(Span::styled(
        "TOTAL",
        Style::default().bold(),
    ))];
    if cols.vendor {
        total_cells.push(Cell::from(""));
    }
    if cols.raw {
        total_cells.push(Cell::from(""));
    }
    if cols.harness {
        total_cells.push(Cell::from(""));
    }
    metric_cells(totals, totals, &cols, false, &mut total_cells);
    rows.push(Row::new(total_cells).style(Style::default().add_modifier(Modifier::BOLD)));

    let title = format!(
        " Usage / API Cost ({}) ",
        dash.view.description()
    );
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(title, Style::default().fg(ACCENT).bold()))
        .title_top(
            Line::from(vec![
                Span::styled(" \u{2191} share of column ", Style::default().fg(COL_PCT)),
                Span::styled("\u{00B7}", Style::default().fg(DIM)),
                Span::styled(" \u{2190} share of row ", Style::default().fg(ROW_PCT)),
            ])
            .alignment(Alignment::Right),
        );
    let summary = &dash.summary;
    let summary_line = format!(
        " Daily ${:.2} | Weekly ${:.2} | Monthly ${:.2} | Saving ${:.2} | {} / MTok ",
        summary.daily,
        summary.weekly,
        summary.monthly,
        summary.savings,
        format_cost_per_mtok(summary.subscription_rate),
    );
    let summary_len = summary_line.chars().count();
    block = block.title_bottom(Line::from(Span::styled(summary_line, Style::default().fg(DIM))));
    if let Some(insight) = &dash.insight {
        // Only when there is room next to the summary title.
        let insight_line = format!(" {} ", insight);
        if (area.width as usize) >= summary_len + insight_line.chars().count() + 4 {
            block = block.title_bottom(
                Line::from(Span::styled(insight_line, Style::default().fg(DIM)))
                    .alignment(Alignment::Right),
            );
        }
    }

    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .column_spacing(1);
    frame.render_widget(table, area);
}

fn vendor_by_name(name: &str) -> Vendor {
    match name {
        "Anthropic" => Vendor::Anthropic,
        "OpenAI" => Vendor::OpenAI,
        "Google" => Vendor::Google,
        "Moonshot" => Vendor::Moonshot,
        "Zhipu" => Vendor::Zhipu,
        _ => Vendor::Unknown,
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

    let slopes: Vec<f64> = points.windows(2).map(|pair| pair[1].1 - pair[0].1).collect();
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
    let seg_cells =
        |s: &crate::tui::data::Segment| ((s.end - s.start + 1) as f64 / len as f64 * width as f64) as usize;
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
            let (h, d) =
                charts::segment_label(&seg.anchor, chart.granularity, seg.total, compact);
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
    for (i, series) in chart.series.iter().enumerate() {
        let item_width = 3 + series.name.chars().count() + 2;
        if used + item_width + 4 > budget {
            spans.push(Span::styled(
                format!("+{}", chart.series.len() - i),
                Style::default().fg(DIM),
            ));
            break;
        }
        spans.push(Span::styled(
            "\u{2500}\u{2500} ",
            Style::default().fg(Color::Indexed(series.color)),
        ));
        spans.push(Span::raw(format!("{}  ", series.name)));
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
            lines.extend(topic.detail.iter().map(|text| {
                Line::from(vec![Span::raw("  "), Span::raw(*text)])
            }));
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
                .title_bottom(Line::from(Span::styled(
                    footer,
                    Style::default().fg(DIM),
                ))),
        ),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::data::{ChartData, Segment, Series};
    use chrono::TimeZone;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

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
                Segment { start: 0, end: 8, total: 9.0, anchor: monday(1) },
                Segment { start: 9, end: 29, total: 21.0, anchor: monday(8) },
                Segment { start: 30, end: 50, total: 21.0, anchor: monday(15) },
                Segment { start: 51, end: 71, total: 21.0, anchor: monday(22) },
                // The one-bucket partial Monday at the window's right edge.
                Segment { start: 72, end: 72, total: 1.0, anchor: monday(29) },
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
        assert!(text.matches('\u{2524}').count() >= 10, "sparse y ticks:\n{}", text);
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

    #[test]
    fn model_column_width_is_bounded_so_the_raw_id_gets_remaining_space() {
        assert_eq!(model_column_width(5), 14);
        assert_eq!(model_column_width(15), 17);
        assert_eq!(model_column_width(80), 26);
    }

    #[test]
    fn column_breakpoints_do_not_clip_metrics_or_the_raw_id() {
        // The table has a border, one-cell gaps, and fixed metric columns.
        // A 108-column terminal cannot fit the three cache-strategy columns
        // in addition to vendor, model, harness, and totals.
        assert!(!visible_cols(108, true).strategy);
        // The wide layout with the raw id needs one more column than 170
        // once the table border and inter-column gaps are included.
        assert!(!visible_cols(170, true).raw);
        assert!(visible_cols(171, true).raw);
    }
}
