//! Usage table layout and rendering for the monitor UI.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use crate::formatting::{format_cost_per_mtok, format_number};
use crate::model_id::Vendor;
use crate::table_view::{CostSummary, DataRow, DisplayRow, RowMetrics, TableView};
use crate::tui::data::Dashboard;
use crate::tui::palette::{
    ACCENT, COL_PCT, DIM, GROUP_BG, ROW_PCT, SCALE_B, SCALE_K, SCALE_M, SCALE_T, SUBTOTAL_BG,
    TABLE_HEADER_BG, TOTAL_BG, ZEBRA_BG, vendor_color,
};

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
            // The full layout occupies 168 cells at its minimum widths;
            // retain a small amount of breathing room before exposing it.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricKind {
    Messages,
    CacheHit,
    Prefill,
    Decode,
    Total,
    Cost,
    Rate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PercentFields {
    None,
    Column,
    ColumnAndRow,
}

impl PercentFields {
    fn column(self) -> bool {
        self != Self::None
    }

    fn row(self) -> bool {
        self == Self::ColumnAndRow
    }
}

impl MetricKind {
    fn label(self) -> &'static str {
        match self {
            Self::Messages => "Msgs",
            Self::CacheHit => "Cache Hit",
            Self::Prefill => "Prefill",
            Self::Decode => "Decode",
            Self::Total => "Total",
            Self::Cost => "Cost",
            Self::Rate => "$/MTok",
        }
    }

    fn min_width(self) -> u16 {
        match self {
            Self::Messages => 12,
            Self::CacheHit | Self::Prefill | Self::Decode => 17,
            Self::Total => 13,
            Self::Cost => 14,
            Self::Rate => 9,
        }
    }

    fn percent_fields(self) -> PercentFields {
        match self {
            Self::Messages | Self::Total | Self::Cost => PercentFields::Column,
            Self::CacheHit | Self::Prefill | Self::Decode => PercentFields::ColumnAndRow,
            Self::Rate => PercentFields::None,
        }
    }

    fn quantity(self, metrics: &RowMetrics) -> Option<i64> {
        match self {
            Self::Messages => Some(metrics.count),
            Self::CacheHit => Some(metrics.cache_hit),
            Self::Prefill => Some(metrics.prefill),
            Self::Decode => Some(metrics.decoding),
            Self::Total => Some(metrics.tokens()),
            Self::Cost | Self::Rate => None,
        }
    }

    fn numeric_value(self, metrics: &RowMetrics) -> f64 {
        self.quantity(metrics).map_or_else(
            || match self {
                Self::Cost => metrics.cost(),
                Self::Rate => metrics.cost_per_mtok(),
                _ => 0.0,
            },
            |value| value as f64,
        )
    }

    fn exact_text(self, metrics: &RowMetrics) -> String {
        match self {
            Self::Cost => format_cost_exact(metrics.cost()),
            Self::Rate => format_cost_per_mtok(metrics.cost_per_mtok()),
            _ => format_number(self.quantity(metrics).unwrap_or_default()),
        }
    }

    fn compact_width(self, unit_slot: bool) -> usize {
        match self {
            Self::Cost => 2 + usize::from(unit_slot),
            Self::Rate => 2,
            _ => 1 + usize::from(unit_slot),
        }
    }
}

fn metric_kinds(cols: &Cols) -> Vec<MetricKind> {
    let mut kinds = vec![MetricKind::Messages];
    if cols.strategy {
        kinds.extend([
            MetricKind::CacheHit,
            MetricKind::Prefill,
            MetricKind::Decode,
        ]);
    }
    kinds.extend([MetricKind::Total, MetricKind::Cost]);
    if cols.rate {
        kinds.push(MetricKind::Rate);
    }
    kinds
}

fn descriptive_column_count(cols: &Cols) -> usize {
    usize::from(cols.vendor) + 1 + usize::from(cols.raw) + usize::from(cols.harness)
}

/// Reserve only the width a readable model label needs so the descriptive
/// columns cannot crowd the numeric metrics on wide terminals.
fn model_column_width(longest_label: usize) -> u16 {
    (longest_label.saturating_add(2) as u16).clamp(14, 26)
}

fn dashboard_model_column_width(rows: &[DisplayRow]) -> u16 {
    let longest = rows
        .iter()
        .filter_map(|row| match row {
            DisplayRow::Data(data) => Some(data.model_label.chars().count()),
            DisplayRow::Subtotal { vendor, .. } => Some(vendor.chars().count() + " total".len()),
            DisplayRow::GroupHeader { .. } => None,
        })
        .max()
        .unwrap_or(0);
    model_column_width(longest)
}

/// Keep descriptive columns content-bound and share surplus width across the
/// numeric metrics, which are the cells that benefit from extra reading room.
fn table_column_widths(area_width: u16, rows: &[DisplayRow], cols: &Cols) -> Vec<u16> {
    let mut descriptive = Vec::new();
    if cols.vendor {
        descriptive.push(9);
    }
    descriptive.push(dashboard_model_column_width(rows));

    let raw_index = if cols.raw {
        let index = descriptive.len();
        descriptive.push(20);
        Some(index)
    } else {
        None
    };
    if cols.harness {
        descriptive.push(if cols.harness_full { 14 } else { 11 });
    }

    let mut metrics = metric_kinds(cols)
        .into_iter()
        .map(MetricKind::min_width)
        .collect::<Vec<_>>();

    let column_count = descriptive.len() + metrics.len();
    let cell_budget = area_width
        .saturating_sub(2)
        .saturating_sub(column_count.saturating_sub(1) as u16);

    if let Some(index) = raw_index {
        let longest = rows
            .iter()
            .filter_map(|row| match row {
                DisplayRow::Data(data) => Some(data.model_raw.chars().count()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let desired = longest.saturating_add(2).clamp(20, 36) as u16;
        let other_width = descriptive
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, width)| *width)
            .chain(metrics.iter().copied())
            .sum::<u16>();
        descriptive[index] = desired.min(cell_budget.saturating_sub(other_width));
    }

    let used = descriptive.iter().chain(&metrics).copied().sum::<u16>();
    let surplus = cell_budget.saturating_sub(used);
    if !metrics.is_empty() {
        let each = surplus / metrics.len() as u16;
        let remainder = surplus % metrics.len() as u16;
        for (index, width) in metrics.iter_mut().enumerate() {
            *width += each + u16::from((index as u16) < remainder);
        }
    }

    descriptive.extend(metrics);
    descriptive
}

fn display_row_metrics(row: &DisplayRow) -> Option<&RowMetrics> {
    match row {
        DisplayRow::Data(data) => Some(&data.metrics),
        DisplayRow::Subtotal { metrics, .. } => Some(metrics),
        DisplayRow::GroupHeader { .. } => None,
    }
}

fn format_cost_exact(value: f64) -> String {
    let sign = if value < 0.0 { "-" } else { "" };
    let fixed = format!("{:.2}", value.abs());
    let (whole, fraction) = fixed.split_once('.').unwrap_or((&fixed, "00"));
    let whole = whole.parse::<i64>().unwrap_or_default();
    format!("{sign}${}.{}", format_number(whole), fraction)
}

fn metric_exact_width(kind: MetricKind, dash: &Dashboard) -> usize {
    dash.rows
        .iter()
        .filter_map(display_row_metrics)
        .chain(std::iter::once(&dash.totals))
        .map(|metrics| kind.exact_text(metrics).chars().count())
        .max()
        .unwrap_or(0)
}

fn metric_max_abs(kind: MetricKind, dash: &Dashboard) -> f64 {
    dash.rows
        .iter()
        .filter_map(display_row_metrics)
        .chain(std::iter::once(&dash.totals))
        .map(|metrics| kind.numeric_value(metrics).abs())
        .fold(0.0, f64::max)
}

const COLUMN_PERCENT_WIDTH: usize = 6;
const ROW_PERCENT_WIDTH: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberMode {
    Exact,
    Compact { significant_digits: usize },
}

#[derive(Debug, Clone, Copy)]
struct NumericLayout {
    width: usize,
    value_width: usize,
    mode: NumberMode,
    column_percent: bool,
    row_percent: bool,
    unit_slot: bool,
}

impl NumericLayout {
    fn new(kind: MetricKind, width: u16, exact_width: usize, max_abs: f64) -> Self {
        let width = width as usize;
        let desired = kind.percent_fields();
        let unit_slot = max_abs >= 1_000.0;
        let mut column_percent = desired.column();
        let mut row_percent = desired.row();
        let percent_width = |column: bool, row: bool| {
            usize::from(column) * COLUMN_PERCENT_WIDTH + usize::from(row) * ROW_PERCENT_WIDTH
        };
        let full_percent_width = percent_width(column_percent, row_percent);

        if exact_width + full_percent_width <= width {
            return Self {
                width,
                value_width: width - full_percent_width,
                mode: NumberMode::Exact,
                column_percent,
                row_percent,
                unit_slot: false,
            };
        }

        loop {
            let reserved = percent_width(column_percent, row_percent);
            let value_width = width.saturating_sub(reserved);
            if value_width >= kind.compact_width(unit_slot) {
                let significant_digits = value_width
                    .saturating_sub(kind.compact_width(unit_slot))
                    .clamp(1, 6);
                return Self {
                    width,
                    value_width,
                    mode: NumberMode::Compact { significant_digits },
                    column_percent,
                    row_percent,
                    unit_slot,
                };
            }
            if row_percent {
                row_percent = false;
            } else if column_percent {
                column_percent = false;
            } else {
                return Self {
                    width,
                    value_width: width,
                    mode: NumberMode::Compact {
                        significant_digits: 1,
                    },
                    column_percent: false,
                    row_percent: false,
                    unit_slot,
                };
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MetricColumn {
    kind: MetricKind,
    layout: NumericLayout,
}

fn metric_columns(widths: &[u16], cols: &Cols, dash: &Dashboard) -> Vec<MetricColumn> {
    metric_kinds(cols)
        .into_iter()
        .zip(widths.iter().skip(descriptive_column_count(cols)).copied())
        .map(|(kind, width)| MetricColumn {
            kind,
            layout: NumericLayout::new(
                kind,
                width,
                metric_exact_width(kind, dash),
                metric_max_abs(kind, dash),
            ),
        })
        .collect()
}

fn scale(value: f64) -> (f64, Option<char>) {
    let abs = value.abs();
    if abs >= 1_000_000_000_000.0 {
        (abs / 1_000_000_000_000.0, Some('T'))
    } else if abs >= 1_000_000_000.0 {
        (abs / 1_000_000_000.0, Some('B'))
    } else if abs >= 1_000_000.0 {
        (abs / 1_000_000.0, Some('M'))
    } else if abs >= 1_000.0 {
        (abs / 1_000.0, Some('K'))
    } else {
        (abs, None)
    }
}

fn trim_fraction(mut text: String) -> String {
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
}

fn scaled_mantissa(value: f64, significant_digits: usize) -> String {
    let digits_before_decimal = if value >= 1.0 {
        value.log10().floor() as usize + 1
    } else {
        1
    };
    let decimals = significant_digits
        .saturating_sub(digits_before_decimal)
        .min(6);
    trim_fraction(format!("{value:.decimals$}"))
}

fn next_scale_unit(unit: char) -> char {
    match unit {
        'K' => 'M',
        'M' => 'B',
        'B' | 'T' => 'T',
        _ => unit,
    }
}

fn compact_scaled_parts(
    value: f64,
    significant_digits: usize,
    format_unscaled: impl Fn(f64) -> String,
) -> (String, Option<char>) {
    let (mut scaled, mut unit) = scale(value);
    loop {
        let mantissa = unit.map_or_else(
            || format_unscaled(scaled),
            |_| scaled_mantissa(scaled, significant_digits),
        );
        let rounded = mantissa.parse::<f64>().unwrap_or_default();
        if rounded < 1_000.0 || unit.is_none() || unit == Some('T') {
            return (mantissa, unit);
        }
        scaled /= 1_000.0;
        unit = unit.map(next_scale_unit);
    }
}

fn compact_quantity_parts(value: i64, significant_digits: usize) -> (String, Option<char>) {
    let sign = if value < 0 { "-" } else { "" };
    let (mantissa, unit) = compact_scaled_parts(value as f64, significant_digits, |scaled| {
        scaled.round().to_string()
    });
    (format!("{sign}{mantissa}"), unit)
}

fn compact_cost_parts(value: f64, significant_digits: usize) -> (String, Option<char>) {
    let sign = if value < 0.0 { "-" } else { "" };
    let (mantissa, unit) =
        compact_scaled_parts(value, significant_digits, |scaled| format!("{scaled:.2}"));
    (format!("{sign}${mantissa}"), unit)
}

fn scale_color(unit: char) -> Color {
    match unit {
        'K' => SCALE_K,
        'M' => SCALE_M,
        'B' => SCALE_B,
        'T' => SCALE_T,
        _ => DIM,
    }
}

fn exact_value_spans(text: String, width: usize, style: Style) -> Vec<Span<'static>> {
    let len = text.chars().count();
    vec![
        Span::raw(" ".repeat(width.saturating_sub(len))),
        Span::styled(text, style),
    ]
}

fn compact_value_spans(
    mut parts: impl FnMut(usize) -> (String, Option<char>),
    layout: &NumericLayout,
    significant_digits: usize,
    style: Style,
) -> Vec<Span<'static>> {
    let unit_width = usize::from(layout.unit_slot);
    let mantissa_width = layout.value_width.saturating_sub(unit_width);
    let mut digits = significant_digits;
    let (mut mantissa, mut unit) = parts(digits);
    while mantissa.chars().count() > mantissa_width && digits > 1 {
        digits -= 1;
        (mantissa, unit) = parts(digits);
    }
    let len = mantissa.chars().count();
    let mut spans = vec![
        Span::raw(" ".repeat(mantissa_width.saturating_sub(len))),
        Span::styled(mantissa, style),
    ];
    if layout.unit_slot {
        spans.push(match unit {
            Some(unit) => Span::styled(
                unit.to_string(),
                Style::default().fg(scale_color(unit)).bold(),
            ),
            None => Span::raw(" "),
        });
    }
    spans
}

fn percentage_text(arrow: char, value: f64) -> String {
    format!("{arrow}{:>3}%", format!("{value:.0}"))
}

fn numeric_cell(
    mut spans: Vec<Span<'static>>,
    layout: &NumericLayout,
    column_percent: Option<f64>,
    row_percent: Option<f64>,
) -> Cell<'static> {
    if layout.column_percent {
        spans.push(match column_percent {
            Some(value) => Span::styled(
                format!(" {}", percentage_text('\u{2191}', value)),
                Style::default().fg(COL_PCT),
            ),
            None => Span::raw(" ".repeat(COLUMN_PERCENT_WIDTH)),
        });
    }
    if layout.row_percent {
        match row_percent {
            Some(value) => {
                spans.push(Span::styled("\u{00B7}", Style::default().fg(DIM)));
                spans.push(Span::styled(
                    percentage_text('\u{2190}', value),
                    Style::default().fg(ROW_PCT),
                ));
            }
            None => spans.push(Span::raw(" ".repeat(ROW_PERCENT_WIDTH))),
        }
    }
    debug_assert_eq!(
        spans
            .iter()
            .map(|span| span.content.chars().count())
            .sum::<usize>(),
        layout.width
    );
    Cell::from(Line::from(spans))
}

fn quantity_cell(
    column: &MetricColumn,
    value: i64,
    column_percent: Option<f64>,
    row_percent: Option<f64>,
) -> Cell<'static> {
    let style = if value == 0 {
        Style::default().fg(DIM)
    } else {
        Style::default()
    };
    let spans = match column.layout.mode {
        NumberMode::Exact => {
            exact_value_spans(format_number(value), column.layout.value_width, style)
        }
        NumberMode::Compact { significant_digits } => compact_value_spans(
            |digits| compact_quantity_parts(value, digits),
            &column.layout,
            significant_digits,
            style,
        ),
    };
    numeric_cell(spans, &column.layout, column_percent, row_percent)
}

fn cost_cell(column: &MetricColumn, value: f64, column_percent: Option<f64>) -> Cell<'static> {
    let spans = match column.layout.mode {
        NumberMode::Exact => exact_value_spans(
            format_cost_exact(value),
            column.layout.value_width,
            Style::default(),
        ),
        NumberMode::Compact { significant_digits } => compact_value_spans(
            |digits| compact_cost_parts(value, digits),
            &column.layout,
            significant_digits,
            Style::default(),
        ),
    };
    numeric_cell(spans, &column.layout, column_percent, None)
}

fn rate_text_for_width(value: f64, width: usize) -> String {
    let exact = format_cost_per_mtok(value);
    if exact.chars().count() <= width {
        return exact;
    }
    for decimals in (0..=6).rev() {
        let candidate = format!("${value:.decimals$}");
        let rounded = candidate
            .strip_prefix('$')
            .and_then(|number| number.parse::<f64>().ok())
            .unwrap_or_default();
        if candidate.chars().count() <= width && (value == 0.0 || rounded != 0.0) {
            return candidate;
        }
    }
    let scientific = format!("${value:.0e}");
    scientific.chars().take(width).collect()
}

fn rate_cell(column: &MetricColumn, value: f64) -> Cell<'static> {
    let text = rate_text_for_width(value, column.layout.value_width);
    numeric_cell(
        exact_value_spans(text, column.layout.value_width, Style::default().fg(DIM)),
        &column.layout,
        None,
        None,
    )
}

fn metric_cells(
    m: &RowMetrics,
    totals: &RowMetrics,
    columns: &[MetricColumn],
    col_pct: bool,
    cells: &mut Vec<Cell<'static>>,
) {
    let row_total = m.tokens();
    for column in columns {
        let column_percent =
            |value: f64, total: f64| (col_pct && total > 0.0).then_some(value / total * 100.0);
        let row_percent =
            |value: i64| (row_total > 0).then_some(value as f64 / row_total as f64 * 100.0);
        let cell = match column.kind {
            MetricKind::Messages => quantity_cell(
                column,
                m.count,
                column_percent(m.count as f64, totals.count as f64),
                None,
            ),
            MetricKind::CacheHit => quantity_cell(
                column,
                m.cache_hit,
                column_percent(m.cache_hit as f64, totals.cache_hit as f64),
                row_percent(m.cache_hit),
            ),
            MetricKind::Prefill => quantity_cell(
                column,
                m.prefill,
                column_percent(m.prefill as f64, totals.prefill as f64),
                row_percent(m.prefill),
            ),
            MetricKind::Decode => quantity_cell(
                column,
                m.decoding,
                column_percent(m.decoding as f64, totals.decoding as f64),
                row_percent(m.decoding),
            ),
            MetricKind::Total => quantity_cell(
                column,
                row_total,
                column_percent(row_total as f64, totals.tokens() as f64),
                None,
            ),
            MetricKind::Cost => {
                cost_cell(column, m.cost(), column_percent(m.cost(), totals.cost()))
            }
            MetricKind::Rate => rate_cell(column, m.cost_per_mtok()),
        };
        cells.push(cell);
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

fn percentage_legend(columns: &[MetricColumn]) -> Vec<Span<'static>> {
    let has_column = columns.iter().any(|column| column.layout.column_percent);
    let has_row = columns.iter().any(|column| column.layout.row_percent);
    let mut spans = Vec::new();
    if has_column {
        spans.push(Span::styled(
            " \u{2191} share of column ",
            Style::default().fg(COL_PCT),
        ));
    }
    if has_row {
        if has_column {
            spans.push(Span::styled("\u{00B7}", Style::default().fg(DIM)));
        }
        spans.push(Span::styled(
            " \u{2190} share of row ",
            Style::default().fg(ROW_PCT),
        ));
    }
    spans
}

fn title_layout(view: TableView, area_width: u16, legend_width: usize) -> (String, bool) {
    let inner_width = area_width.saturating_sub(2) as usize;
    let full = format!(" Usage / API Cost ({}) ", view.description());
    let short = " Usage / API Cost ".to_string();
    let gap = usize::from(legend_width > 0);
    if full.chars().count() + legend_width + gap <= inner_width {
        (full, true)
    } else if short.chars().count() + legend_width + gap <= inner_width {
        (short, true)
    } else {
        (short, false)
    }
}

fn compact_cost_text(value: f64) -> String {
    let (mantissa, unit) = compact_cost_parts(value, 4);
    unit.map_or(mantissa.clone(), |unit| format!("{mantissa}{unit}"))
}

fn summary_title(summary: &CostSummary, width: usize) -> Option<String> {
    let daily = compact_cost_text(summary.daily);
    let weekly = compact_cost_text(summary.weekly);
    let monthly = compact_cost_text(summary.monthly);
    let savings = compact_cost_text(summary.savings);
    let rate = format_cost_per_mtok(summary.subscription_rate);
    let candidates = [
        format!(
            " Daily {} | Weekly {} | Monthly {} | Saving {} | {} / MTok ",
            format_cost_exact(summary.daily),
            format_cost_exact(summary.weekly),
            format_cost_exact(summary.monthly),
            format_cost_exact(summary.savings),
            rate,
        ),
        format!(
            " Daily {daily} | Weekly {weekly} | Monthly {monthly} | Saving {savings} | {rate} / MTok "
        ),
        format!(" Daily {daily} | Weekly {weekly} | Monthly {monthly} | Saving {savings} "),
        format!(" Daily {daily} | Monthly {monthly} | Saving {savings} "),
        format!(" Daily {daily} | Saving {savings} "),
    ];
    candidates
        .into_iter()
        .find(|candidate| candidate.chars().count() <= width)
}

pub(super) fn draw_table(frame: &mut Frame, area: Rect, dash: &Dashboard) {
    let show_harness = dash.tool.is_all();
    let cols = visible_cols(area.width, show_harness);
    let totals = &dash.totals;
    let column_widths = table_column_widths(area.width, &dash.rows, &cols);
    let metric_columns = metric_columns(&column_widths, &cols, dash);

    let mut header_cells: Vec<Cell> = Vec::new();
    if cols.vendor {
        header_cells.push(Cell::from("Vendor"));
    }
    header_cells.push(Cell::from("Model"));
    if cols.raw {
        header_cells.push(Cell::from("Model Id"));
    }
    if cols.harness {
        header_cells.push(Cell::from("Harness"));
    }
    for column in &metric_columns {
        header_cells.push(Cell::from(
            Line::from(column.kind.label()).alignment(Alignment::Center),
        ));
    }
    let widths = column_widths
        .iter()
        .copied()
        .map(Constraint::Length)
        .collect::<Vec<_>>();
    let n_cols = header_cells.len();
    let header = Row::new(header_cells).style(
        Style::default()
            .fg(Color::Indexed(252))
            .bg(TABLE_HEADER_BG)
            .add_modifier(Modifier::BOLD),
    );

    let mut rows: Vec<Row> = Vec::new();
    let mut data_row_index = 0usize;
    for display_row in &dash.rows {
        match display_row {
            DisplayRow::GroupHeader { vendor } => {
                let color = vendor_color(vendor_by_name(vendor));
                let mut cells = vec![Cell::from(Span::styled(
                    vendor.clone(),
                    Style::default().fg(color).bold(),
                ))];
                cells.resize_with(n_cols, || Cell::from(""));
                rows.push(Row::new(cells).style(Style::default().bg(GROUP_BG)));
            }
            DisplayRow::Data(d) => {
                let mut cells: Vec<Cell> = Vec::new();
                if cols.vendor {
                    cells.push(Cell::from(Span::styled(
                        d.vendor_label.clone(),
                        Style::default().fg(vendor_color(d.vendor)).bold(),
                    )));
                }
                cells.push(Cell::from(model_cell_text(
                    d,
                    &cols,
                    show_harness,
                    dash.view,
                )));
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
                metric_cells(&d.metrics, totals, &metric_columns, true, &mut cells);
                let style = if data_row_index % 2 == 1 {
                    Style::default().bg(ZEBRA_BG)
                } else {
                    Style::default()
                };
                rows.push(Row::new(cells).style(style));
                data_row_index += 1;
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
                metric_cells(metrics, totals, &metric_columns, true, &mut cells);
                rows.push(
                    Row::new(cells)
                        .style(Style::default().bg(SUBTOTAL_BG).add_modifier(Modifier::DIM)),
                );
            }
        }
    }

    let mut total_cells: Vec<Cell> =
        vec![Cell::from(Span::styled("TOTAL", Style::default().bold()))];
    if cols.vendor {
        total_cells.push(Cell::from(""));
    }
    if cols.raw {
        total_cells.push(Cell::from(""));
    }
    if cols.harness {
        total_cells.push(Cell::from(""));
    }
    metric_cells(totals, totals, &metric_columns, false, &mut total_cells);
    rows.push(
        Row::new(total_cells).style(Style::default().bg(TOTAL_BG).add_modifier(Modifier::BOLD)),
    );

    let legend_spans = percentage_legend(&metric_columns);
    let legend_width = legend_spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum::<usize>();
    let (title, show_legend) = title_layout(dash.view, area.width, legend_width);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(title, Style::default().fg(ACCENT).bold()));
    if show_legend && !legend_spans.is_empty() {
        block = block.title_top(Line::from(legend_spans).alignment(Alignment::Right));
    }
    let summary = &dash.summary;
    let summary_line = summary_title(summary, area.width.saturating_sub(2) as usize);
    let summary_len = summary_line.as_ref().map_or(0, |line| line.chars().count());
    if let Some(summary_line) = summary_line {
        block = block.title_bottom(Line::from(Span::styled(
            summary_line,
            Style::default().fg(DIM),
        )));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_values_promote_rounded_scale_boundaries() {
        assert_eq!(
            compact_quantity_parts(999_999_999, 3),
            ("1".to_string(), Some('B'))
        );
        assert_eq!(
            compact_cost_parts(999_999.99, 3),
            ("$1".to_string(), Some('M'))
        );
        assert_eq!(
            compact_cost_parts(999.999, 3),
            ("$1000.00".to_string(), None)
        );
    }

    #[test]
    fn constrained_rate_does_not_round_a_nonzero_value_to_zero() {
        let text = rate_text_for_width(0.000_001_2, 8);

        assert!(text.chars().count() <= 8);
        assert_ne!(text, "$0.00000");
        assert!(text.contains('1'), "{text}");
    }

    #[test]
    fn model_column_width_is_bounded_for_wide_table_layout() {
        assert_eq!(model_column_width(5), 14);
        assert_eq!(model_column_width(15), 17);
        assert_eq!(model_column_width(80), 26);
    }

    #[test]
    fn model_column_reserves_room_for_vendor_subtotals() {
        let rows = vec![
            DisplayRow::Data(Box::new(DataRow {
                vendor: Vendor::Anthropic,
                vendor_label: "Anthropic".to_string(),
                model_label: "Opus 5".to_string(),
                model_raw: "claude-opus-5".to_string(),
                harness_label: "Claude Code".to_string(),
                harness_short: "CC".to_string(),
                metrics: RowMetrics::default(),
            })),
            DisplayRow::Subtotal {
                vendor: "Anthropic".to_string(),
                metrics: RowMetrics::default(),
            },
        ];

        assert_eq!(dashboard_model_column_width(&rows), 17);
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
