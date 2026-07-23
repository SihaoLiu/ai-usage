use crate::constants::SubscriptionFees;
use crate::stats::ModelBreakdownRow;
use crate::table_view::{
    DataRow, DisplayRow, RowMetrics, TableView, build_table, table_totals,
};

fn fit_text_to_width(text: &str, width: usize) -> String {
    if text.len() <= width {
        text.to_string()
    } else if width <= 3 {
        text[..width].to_string()
    } else {
        format!("{}...", &text[..width - 3])
    }
}

/// Format number with thousand separators.
pub fn format_number(num: i64) -> String {
    let s = num.to_string();
    let bytes = s.as_bytes();
    let mut result = Vec::new();
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) && b != b'-' {
            result.push(b',');
        }
        result.push(b);
    }
    String::from_utf8(result).unwrap()
}

/// Format number compactly with K/M/B suffixes.
pub fn format_number_compact(value: i64) -> String {
    let v = value as f64;
    if v >= 1_000_000_000.0 {
        let val_b = v / 1_000_000_000.0;
        if val_b >= 100.0 {
            format!("{}B", val_b as i64)
        } else if val_b >= 10.0 {
            format!("{:.1}B", val_b)
        } else {
            format!("{:.2}B", val_b)
        }
    } else if v >= 1_000_000.0 {
        let val_m = v / 1_000_000.0;
        if val_m >= 100.0 {
            format!("{}M", val_m as i64)
        } else if val_m >= 10.0 {
            format!("{:.1}M", val_m)
        } else {
            format!("{:.2}M", val_m)
        }
    } else if v >= 1_000.0 {
        let val_k = v / 1_000.0;
        if val_k >= 100.0 {
            format!("{}K", val_k as i64)
        } else if val_k >= 10.0 {
            format!("{:.1}K", val_k)
        } else {
            format!("{:.2}K", val_k)
        }
    } else {
        format!("{}", value)
    }
}

/// Format Y-axis value to always be 5 characters with K/M units.
pub fn format_y_axis_value(value: f64) -> String {
    if value >= 1_000_000_000.0 {
        let val_b = value / 1_000_000_000.0;
        if val_b >= 100.0 {
            format!("{:3} B", val_b as i64)
        } else if val_b >= 10.0 {
            format!(" {:2} B", val_b as i64)
        } else {
            format!("{:3.1} B", val_b)
        }
    } else if value >= 1_000_000.0 {
        let val_m = value / 1_000_000.0;
        if val_m >= 100.0 {
            format!("{:3} M", val_m as i64)
        } else if val_m >= 10.0 {
            format!(" {:2} M", val_m as i64)
        } else {
            format!("{:3.1} M", val_m)
        }
    } else if value >= 1000.0 {
        let val_k = value / 1000.0;
        if val_k >= 100.0 {
            format!("{:3} K", val_k as i64)
        } else if val_k >= 10.0 {
            format!(" {:2} K", val_k as i64)
        } else {
            format!("{:3.1} K", val_k)
        }
    } else {
        format!("{:5}", value as i64)
    }
}

/// Format total value with B/M/K units.
pub fn format_total_value(value: f64) -> String {
    format_number_compact(value as i64)
}

/// Format cost per MTok with appropriate precision (at least 2 significant figures).
pub fn format_cost_per_mtok(value: f64) -> String {
    if value <= 0.0 {
        "$0.00".to_string()
    } else if value >= 0.1 {
        format!("${:.2}", value)
    } else {
        // For small values, ensure at least 2 significant digits
        let leading_zeros = (-value.log10()).ceil() as usize;
        let decimal_places = leading_zeros + 1;
        format!("${:.prec$}", value, prec = decimal_places)
    }
}

// ANSI color codes for percentage breakdown.
// COL_PCT_COLOR: model's share among models for one token type (column direction, marked with ↑).
// ROW_PCT_COLOR: token type's share within a single model/row (row direction, marked with ←).
const COL_PCT_COLOR: &str = "\x1b[36m"; // cyan
const ROW_PCT_COLOR: &str = "\x1b[33m"; // yellow
const COLOR_RESET: &str = "\x1b[0m";

/// Format a remaining duration as a zero-padded `HH:MM:SS` countdown string.
pub fn format_countdown(remaining: std::time::Duration) -> String {
    let total = remaining.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

fn pad_left(visible_len: usize, width: usize) -> String {
    if visible_len < width {
        " ".repeat(width - visible_len)
    } else {
        String::new()
    }
}

fn right_align_visible(text: String, visible_len: usize, width: usize) -> String {
    format!("{}{}", pad_left(visible_len, width), text)
}

fn format_cost_compact(value: f64) -> String {
    let sign = if value < 0.0 { "-" } else { "" };
    let abs = value.abs();
    if abs >= 1_000_000.0 {
        format!("{}${:.1}M", sign, abs / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{}${:.1}K", sign, abs / 1_000.0)
    } else {
        format!("{}${:.2}", sign, abs)
    }
}

fn format_cost_for_width(cost: f64, width: usize) -> String {
    let exact = format!("${:.2}", cost);
    let exact_len = exact.chars().count();
    if exact_len <= width {
        return right_align_visible(exact, exact_len, width);
    }

    let compact = format_cost_compact(cost);
    let compact_len = compact.chars().count();
    if compact_len <= width {
        return right_align_visible(compact, compact_len, width);
    }

    fit_text_to_width(&compact, width)
}

fn format_rate_for_width(rate: f64, width: usize) -> String {
    let exact = format_cost_per_mtok(rate);
    let exact_len = exact.chars().count();
    if exact_len <= width {
        return right_align_visible(exact, exact_len, width);
    }

    let compact = format_cost_compact(rate);
    let compact_len = compact.chars().count();
    if compact_len <= width {
        return right_align_visible(compact, compact_len, width);
    }

    fit_text_to_width(&compact, width)
}

/// Single column-percentage cell with cyan ↑ arrow (used for Messages column).
fn format_with_col_pct(value: i64, total: i64, width: usize) -> String {
    let pct = if total > 0 {
        value as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let pct_str = format!("{:.0}%", pct);
    for value_str in [format_number(value), format_number_compact(value)] {
        let visible = value_str.chars().count() + pct_str.chars().count() + 3;
        if visible <= width {
            return format!(
                "{pad}{val}(\u{2191}{cc}{pct}{rst})",
                pad = pad_left(visible, width),
                val = value_str,
                cc = COL_PCT_COLOR,
                pct = pct_str,
                rst = COLOR_RESET,
            );
        }
    }

    let value_str = format_number_compact(value);
    let visible = value_str.chars().count();
    if visible <= width {
        return right_align_visible(value_str, visible, width);
    }

    fit_text_to_width(&value_str, width)
}

/// Token cell with both column-% (↑ cyan, model share among models) and row-% (← yellow,
/// token type share within this row). When `compact` is true, the value is shown in
/// K/M/B form so dual-percent cells fit in tighter terminal widths without misaligning.
fn format_with_dual_pct(
    value: i64,
    col_total: i64,
    row_total: i64,
    width: usize,
    compact: bool,
) -> String {
    let col_pct = if col_total > 0 {
        value as f64 / col_total as f64 * 100.0
    } else {
        0.0
    };
    let row_pct = if row_total > 0 {
        value as f64 / row_total as f64 * 100.0
    } else {
        0.0
    };
    let col_str = format!("{:.0}%", col_pct);
    let row_str = format!("{:.0}%", row_pct);
    let mut value_options = if compact {
        vec![format_number_compact(value)]
    } else {
        vec![format_number(value), format_number_compact(value)]
    };
    value_options.dedup();

    for value_str in &value_options {
        // Visible: value + "(" + "↑" + col + "·" + "←" + row + ")" -> 5 fixed chars
        let visible =
            value_str.chars().count() + col_str.chars().count() + row_str.chars().count() + 5;
        if visible <= width {
            return format!(
                "{pad}{val}(\u{2191}{cc}{col}{rst}\u{00B7}\u{2190}{rc}{row}{rst})",
                pad = pad_left(visible, width),
                val = value_str,
                cc = COL_PCT_COLOR,
                col = col_str,
                rst = COLOR_RESET,
                rc = ROW_PCT_COLOR,
                row = row_str,
            );
        }
    }

    let value_str = format_number_compact(value);
    let visible = value_str.chars().count() + col_str.chars().count() + 3;
    if visible <= width {
        return format!(
            "{pad}{val}(\u{2191}{cc}{col}{rst})",
            pad = pad_left(visible, width),
            val = value_str,
            cc = COL_PCT_COLOR,
            col = col_str,
            rst = COLOR_RESET,
        );
    }

    let visible = value_str.chars().count();
    if visible <= width {
        return right_align_visible(value_str, visible, width);
    }

    fit_text_to_width(&value_str, width)
}

/// Cost cell with row-percentage (← yellow). Cost row has only one row, so column-%
/// would always be 100% and is omitted.
fn format_cost_with_row_pct(cost: f64, row_total: f64, width: usize) -> String {
    let row_pct = if row_total > 0.0 {
        cost / row_total * 100.0
    } else {
        0.0
    };
    let row_str = format!("{:.0}%", row_pct);
    for cost_str in [format!("${:.2}", cost), format_cost_compact(cost)] {
        let visible = cost_str.chars().count() + row_str.chars().count() + 3;
        if visible <= width {
            return format!(
                "{pad}{cost}(\u{2190}{rc}{row}{rst})",
                pad = pad_left(visible, width),
                cost = cost_str,
                rc = ROW_PCT_COLOR,
                row = row_str,
                rst = COLOR_RESET,
            );
        }
    }

    format_cost_for_width(cost, width)
}

fn format_model_cost_with_col_pct(cost: f64, total_cost: f64, width: usize) -> String {
    let col_pct = if total_cost > 0.0 {
        cost / total_cost * 100.0
    } else {
        0.0
    };
    let col_str = format!("{:.0}%", col_pct);
    for cost_str in [format!("${:.2}", cost), format_cost_compact(cost)] {
        let visible = cost_str.chars().count() + col_str.chars().count() + 3;
        if visible <= width {
            return format!(
                "{pad}{cost}(\u{2191}{cc}{col}{rst})",
                pad = pad_left(visible, width),
                cost = cost_str,
                cc = COL_PCT_COLOR,
                col = col_str,
                rst = COLOR_RESET,
            );
        }
    }

    format_cost_for_width(cost, width)
}

/// Centered legend line explaining the dual-percentage color coding.
fn dual_pct_legend(table_width: usize) -> String {
    let visible = "Legend:  \u{2191} % across models   \u{00B7}   \u{2190} % within model"
        .chars()
        .count();
    let lpad = if visible < table_width {
        " ".repeat((table_width - visible) / 2)
    } else {
        String::new()
    };
    format!(
        "{lpad}Legend:  {cc}\u{2191} % across models{rst}   \u{00B7}   {rc}\u{2190} % within model{rst}",
        lpad = lpad,
        cc = COL_PCT_COLOR,
        rst = COLOR_RESET,
        rc = ROW_PCT_COLOR,
    )
}

/// Determine table display mode based on terminal dimensions.
pub fn get_table_display_mode(
    terminal_width: u16,
    terminal_height: u16,
    num_models: usize,
) -> &'static str {
    let min_table_height = 10 + num_models;
    if (terminal_height as usize) < min_table_height + 20 {
        return "hidden";
    }
    if terminal_width >= 205 {
        "full"
    } else if terminal_width >= 137 {
        "medium"
    } else if terminal_width >= 84 {
        "compact"
    } else if terminal_width >= 70 {
        "minimal"
    } else {
        "hidden"
    }
}

/// Get the table width for a given display mode.
pub fn get_table_width(mode: &str) -> usize {
    match mode {
        "full" => 198,
        "medium" => 128,
        "compact" => 72,
        "minimal" => 54,
        _ => 0,
    }
}

/// Sub-column widths inside the leading name area of a table row. The name
/// area always renders to `total` visible characters; `vendor`/`harness` are
/// zero when that sub-column is hidden for the current mode.
struct NameLayout {
    vendor: usize,
    model: usize,
    harness: usize,
    total: usize,
}

fn name_layout(mode: &str, show_harness: bool) -> NameLayout {
    match (mode, show_harness) {
        ("full", true) => NameLayout {
            vendor: 9,
            model: 27,
            harness: 12,
            total: 50,
        },
        ("full", false) => NameLayout {
            vendor: 9,
            model: 40,
            harness: 0,
            total: 50,
        },
        ("medium", true) => NameLayout {
            vendor: 9,
            model: 14,
            harness: 6,
            total: 31,
        },
        ("medium", false) => NameLayout {
            vendor: 9,
            model: 21,
            harness: 0,
            total: 31,
        },
        _ => NameLayout {
            vendor: 0,
            model: 12,
            harness: 0,
            total: 12,
        },
    }
}

fn cell(text: &str, width: usize) -> String {
    format!("{:<w$}", fit_text_to_width(text, width), w = width)
}

/// Compose the vendor / model / harness sub-columns into one fixed-width
/// name-area string.
fn compose_name(l: &NameLayout, vendor: &str, model: &str, harness: &str) -> String {
    let mut s = String::with_capacity(l.total);
    if l.vendor > 0 {
        s.push_str(&cell(vendor, l.vendor));
        s.push(' ');
    }
    s.push_str(&cell(model, l.model));
    if l.harness > 0 {
        s.push(' ');
        s.push_str(&cell(harness, l.harness));
    }
    s
}

/// Single-column row label for the narrow (compact/minimal) layouts: the
/// harness tag is folded into the label because there is no room for columns.
fn narrow_name(row: &DataRow, show_harness: bool, view: TableView) -> String {
    if show_harness && view != TableView::Model {
        format!("{}:{}", row.harness_short, row.model_label)
    } else {
        row.model_label.clone()
    }
}

/// Print the usage breakdown table with responsive formatting.
/// Returns true if the table was printed, false if hidden.
pub fn print_model_breakdown(
    model_stats: &[ModelBreakdownRow],
    days_in_data: f64,
    terminal_width: Option<u16>,
    terminal_height: Option<u16>,
    tool: &str,
    subscription_fees: &SubscriptionFees,
    view: TableView,
) -> bool {
    let show_harness = tool == "all";
    let rows = build_table(model_stats, view);
    let totals = table_totals(model_stats);

    let mode = match (terminal_width, terminal_height) {
        (Some(w), Some(h)) => get_table_display_mode(w, h, rows.len()),
        _ => "full",
    };
    if mode == "hidden" {
        return false;
    }

    let subscription_price = subscription_fees.get(tool);
    let tw = terminal_width.unwrap_or(200) as usize;

    match mode {
        "full" | "medium" => print_table_rich(mode, &rows, &totals, show_harness, view, tw),
        "compact" => print_table_compact(&rows, &totals, show_harness, view, tw),
        "minimal" => print_table_minimal(&rows, &totals, show_harness, view, tw),
        _ => {}
    }

    // Print cost summary
    let summary = crate::table_view::cost_summary(&totals, days_in_data, subscription_price);
    let table_width = get_table_width(mode);
    let cost_pad = center_pad(tw, table_width);

    if mode == "full" || mode == "medium" {
        let line = format!(
            "Daily: ${:.2}, Weekly: ${:.2}, Monthly(30d): ${:.2}, Monthly Saving ${:.2}, {} / MTok",
            summary.daily,
            summary.weekly,
            summary.monthly,
            summary.savings,
            format_cost_per_mtok(summary.subscription_rate)
        );
        println!("{}{}", cost_pad, line);
        if let Some(line) = top_model_insight_line(&rows, summary.total_cost, show_harness) {
            println!("{}{}", cost_pad, fit_text_to_width(&line, table_width));
        }
    } else {
        let line = format!(
            "Daily: ${:.2}, Monthly: ${:.2}, Saving: ${:.2}",
            summary.daily, summary.monthly, summary.savings
        );
        println!("{}{}", cost_pad, line);
    }

    true
}

fn data_rows(rows: &[DisplayRow]) -> impl Iterator<Item = &DataRow> {
    rows.iter().filter_map(|row| match row {
        DisplayRow::Data(d) => Some(d.as_ref()),
        _ => None,
    })
}

pub(crate) fn top_model_insight_line(
    rows: &[DisplayRow],
    total_cost: f64,
    show_harness: bool,
) -> Option<String> {
    let top_spend = data_rows(rows).max_by(|a, b| a.metrics.cost().total_cmp(&b.metrics.cost()))?;
    let highest_rate = data_rows(rows)
        .filter(|d| d.metrics.tokens() > 0)
        .max_by(|a, b| {
            a.metrics
                .cost_per_mtok()
                .total_cmp(&b.metrics.cost_per_mtok())
        })?;

    let name = |d: &DataRow| {
        if show_harness {
            format!("{} ({})", d.model_label, d.harness_short)
        } else {
            d.model_label.clone()
        }
    };
    let spend = top_spend.metrics.cost();
    let spend_pct = if total_cost > 0.0 {
        spend / total_cost * 100.0
    } else {
        0.0
    };

    Some(format!(
        "Top spend: {} {} ({:.0}%) | Highest rate: {} {} / MTok",
        fit_text_to_width(&name(top_spend), 28),
        format_cost_compact(spend),
        spend_pct,
        fit_text_to_width(&name(highest_rate), 28),
        format_cost_per_mtok(highest_rate.metrics.cost_per_mtok())
    ))
}

/// Center a line of given content_width within terminal_width using left padding.
pub fn center_pad(terminal_width: usize, content_width: usize) -> String {
    if terminal_width > content_width {
        " ".repeat((terminal_width - content_width) / 2)
    } else {
        String::new()
    }
}

/// Column widths for the two wide (percent-annotated) table modes.
struct RichLayout {
    table_width: usize,
    name: NameLayout,
    w_msgs: usize,
    w_cache: usize,
    w_cost: usize,
    /// Zero hides the $/MTok column (medium mode).
    w_rate: usize,
    compact_cells: bool,
    raw_model: bool,
}

fn rich_layout(mode: &str, show_harness: bool) -> RichLayout {
    if mode == "full" {
        RichLayout {
            table_width: get_table_width("full"),
            name: name_layout("full", show_harness),
            w_msgs: 14,
            w_cache: 24,
            w_cost: 15,
            w_rate: 10,
            compact_cells: false,
            raw_model: true,
        }
    } else {
        RichLayout {
            table_width: get_table_width("medium"),
            name: name_layout("medium", show_harness),
            w_msgs: 13,
            w_cache: 15,
            w_cost: 12,
            w_rate: 0,
            compact_cells: true,
            raw_model: false,
        }
    }
}

fn print_table_rich(
    mode: &str,
    rows: &[DisplayRow],
    totals: &RowMetrics,
    show_harness: bool,
    view: TableView,
    terminal_width: usize,
) {
    let l = rich_layout(mode, show_harness);
    let p = center_pad(terminal_width, l.table_width);
    let total_cost = totals.cost();

    let metric_cells = |m: &RowMetrics| -> (String, String, String, String, String, String) {
        let row_total = m.tokens();
        (
            format_with_col_pct(m.count, totals.count, l.w_msgs),
            format_with_dual_pct(
                m.cache_hit,
                totals.cache_hit,
                row_total,
                l.w_cache,
                l.compact_cells,
            ),
            format_with_dual_pct(
                m.prefill,
                totals.prefill,
                row_total,
                l.w_cache,
                l.compact_cells,
            ),
            format_with_dual_pct(
                m.decoding,
                totals.decoding,
                row_total,
                l.w_cache,
                l.compact_cells,
            ),
            format_with_dual_pct(
                row_total,
                totals.tokens(),
                row_total,
                l.w_cache,
                l.compact_cells,
            ),
            format_model_cost_with_col_pct(m.cost(), total_cost, l.w_cost),
        )
    };
    let rate_cell = |m: &RowMetrics| {
        if l.w_rate > 0 {
            format!(" {}", format_rate_for_width(m.cost_per_mtok(), l.w_rate))
        } else {
            String::new()
        }
    };

    println!();
    println!(
        "{}{:^width$}",
        p,
        format!("Usage / API Cost ({})", view.description()),
        width = l.table_width
    );
    println!("{}{}", p, "=".repeat(l.table_width));
    println!("{}{}", p, dual_pct_legend(l.table_width));

    let (h_msgs, h_cache, h_prefill, h_decode) = if mode == "full" {
        ("Messages", "Cache Hit", "Prefill", "Decoding")
    } else {
        ("Msgs", "CacheHit", "Prefill", "Decode")
    };
    let rate_hdr = if l.w_rate > 0 {
        format!(" {:>w$}", "$/MTok", w = l.w_rate)
    } else {
        String::new()
    };
    let harness_hdr = if l.name.harness >= 7 { "Harness" } else { "Via" };
    println!(
        "{}| {} {:>wn$} | {:>wc$} {:>wc$} {:>wc$} {:>wc$} {:>wcost$}{} |",
        p,
        compose_name(&l.name, "Vendor", "Model", harness_hdr),
        h_msgs,
        h_cache,
        h_prefill,
        h_decode,
        "Total",
        "Cost",
        rate_hdr,
        wn = l.w_msgs,
        wc = l.w_cache,
        wcost = l.w_cost,
    );
    println!("{}|{}|", p, "-".repeat(l.table_width - 2));

    for row in rows {
        match row {
            DisplayRow::GroupHeader { vendor } => {
                println!("{}| {:<w$} |", p, vendor, w = l.table_width - 4);
            }
            DisplayRow::Data(d) => {
                let model = if l.raw_model {
                    d.model_raw.as_str()
                } else {
                    d.model_label.as_str()
                };
                let harness = if mode == "full" {
                    d.harness_label.as_str()
                } else {
                    d.harness_short.as_str()
                };
                let name = compose_name(&l.name, &d.vendor_label, model, harness);
                let (msgs, ch, pf, dc, tot, cost) = metric_cells(&d.metrics);
                println!(
                    "{}| {} {} | {} {} {} {} {}{} |",
                    p,
                    name,
                    msgs,
                    ch,
                    pf,
                    dc,
                    tot,
                    cost,
                    rate_cell(&d.metrics)
                );
            }
            DisplayRow::Subtotal { vendor, metrics } => {
                let name = compose_name(&l.name, vendor, "total", "");
                let (msgs, ch, pf, dc, tot, cost) = metric_cells(metrics);
                println!(
                    "{}| {} {} | {} {} {} {} {}{} |",
                    p,
                    name,
                    msgs,
                    ch,
                    pf,
                    dc,
                    tot,
                    cost,
                    rate_cell(metrics)
                );
            }
        }
    }

    println!("{}|{}|", p, "-".repeat(l.table_width - 2));
    let (msgs, ch, pf, dc, tot, cost) = metric_cells(totals);
    println!(
        "{}| {} {} | {} {} {} {} {}{} |",
        p,
        compose_name(&l.name, "TOTAL", "", ""),
        msgs,
        ch,
        pf,
        dc,
        tot,
        cost,
        rate_cell(totals)
    );

    println!(
        "{}| {} {:>wn$} | {} {} {} {} {}{} |",
        p,
        compose_name(&l.name, "Cost(API)", "", ""),
        "",
        format_cost_with_row_pct(totals.cache_hit_cost, total_cost, l.w_cache),
        format_cost_with_row_pct(totals.prefill_cost, total_cost, l.w_cache),
        format_cost_with_row_pct(totals.decoding_cost, total_cost, l.w_cache),
        format_cost_with_row_pct(total_cost, total_cost, l.w_cache),
        format_cost_with_row_pct(total_cost, total_cost, l.w_cost),
        rate_cell(totals),
        wn = l.w_msgs,
    );
    println!("{}{}", p, "=".repeat(l.table_width));
}

fn print_table_compact(
    rows: &[DisplayRow],
    totals: &RowMetrics,
    show_harness: bool,
    view: TableView,
    terminal_width: usize,
) {
    let w_name = 12;
    let w_msgs = 7;
    let w_val = 8;
    let w_cost = 9;
    let table_width = get_table_width("compact");
    let p = center_pad(terminal_width, table_width);

    println!();
    println!("{}{:^width$}", p, "Usage / API Cost", width = table_width);
    println!("{}{}", p, "=".repeat(table_width));

    println!(
        "{}| {:<w_name$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} {:>w_cost$} |",
        p,
        "Model",
        "Msgs",
        "CacheHit",
        "Prefill",
        "Decode",
        "Total",
        "Cost",
        w_name = w_name,
        w_msgs = w_msgs,
        w_val = w_val,
        w_cost = w_cost,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    let metric_row = |name: String, m: &RowMetrics| {
        println!(
            "{}| {:<w_name$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} {} |",
            p,
            name,
            format_number_compact(m.count),
            format_number_compact(m.cache_hit),
            format_number_compact(m.prefill),
            format_number_compact(m.decoding),
            format_number_compact(m.tokens()),
            format_cost_for_width(m.cost(), w_cost),
            w_name = w_name,
            w_msgs = w_msgs,
            w_val = w_val,
        );
    };

    for row in rows {
        match row {
            DisplayRow::GroupHeader { vendor } => {
                println!("{}| {:<w$} |", p, vendor, w = table_width - 4);
            }
            DisplayRow::Data(d) => {
                let name = fit_text_to_width(&narrow_name(d, show_harness, view), w_name);
                metric_row(name, &d.metrics);
            }
            DisplayRow::Subtotal { vendor, metrics } => {
                let name = fit_text_to_width(&format!("= {}", vendor), w_name);
                metric_row(name, metrics);
            }
        }
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    metric_row("TOTAL".to_string(), totals);

    println!(
        "{}| {:<w_name$} {:>w_msgs$} | {} {} {} {} {} |",
        p,
        "Cost",
        "",
        format_cost_for_width(totals.cache_hit_cost, w_val),
        format_cost_for_width(totals.prefill_cost, w_val),
        format_cost_for_width(totals.decoding_cost, w_val),
        format_cost_for_width(totals.cost(), w_val),
        format_cost_for_width(totals.cost(), w_cost),
        w_name = w_name,
        w_msgs = w_msgs,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

fn print_table_minimal(
    rows: &[DisplayRow],
    totals: &RowMetrics,
    show_harness: bool,
    view: TableView,
    terminal_width: usize,
) {
    let w_name = 12;
    let w_msgs = 6;
    let w_tokens = 9;
    let w_cost = 9;
    let w_rate = 8;
    let table_width = get_table_width("minimal");
    let p = center_pad(terminal_width, table_width);

    println!();
    println!("{}{:^width$}", p, "Usage / Cost", width = table_width);
    println!("{}{}", p, "=".repeat(table_width));

    println!(
        "{}| {:<w_name$} {:>w_msgs$} | {:>w_tokens$} {:>w_cost$} {:>w_rate$} |",
        p,
        "Model",
        "Msgs",
        "Tokens",
        "Cost",
        "$/MTok",
        w_name = w_name,
        w_msgs = w_msgs,
        w_tokens = w_tokens,
        w_cost = w_cost,
        w_rate = w_rate,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    let metric_row = |name: String, m: &RowMetrics| {
        println!(
            "{}| {:<w_name$} {:>w_msgs$} | {:>w_tokens$} {} {} |",
            p,
            name,
            format_number_compact(m.count),
            format_number_compact(m.tokens()),
            format_cost_for_width(m.cost(), w_cost),
            format_rate_for_width(m.cost_per_mtok(), w_rate),
            w_name = w_name,
            w_msgs = w_msgs,
            w_tokens = w_tokens,
        );
    };

    for row in rows {
        match row {
            DisplayRow::GroupHeader { vendor } => {
                println!("{}| {:<w$} |", p, vendor, w = table_width - 4);
            }
            DisplayRow::Data(d) => {
                let name = fit_text_to_width(&narrow_name(d, show_harness, view), w_name);
                metric_row(name, &d.metrics);
            }
            DisplayRow::Subtotal { vendor, metrics } => {
                let name = fit_text_to_width(&format!("= {}", vendor), w_name);
                metric_row(name, metrics);
            }
        }
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    metric_row("TOTAL".to_string(), totals);

    println!(
        "{}| {:<w_name$} {:>w_msgs$} | {:>w_tokens$} {} {} |",
        p,
        "Cost",
        "",
        "",
        format_cost_for_width(totals.cost(), w_cost),
        format_rate_for_width(totals.cost_per_mtok(), w_rate),
        w_name = w_name,
        w_msgs = w_msgs,
        w_tokens = w_tokens,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table_view::display_model_name;
    use std::time::Duration;

    fn visible_len(text: &str) -> usize {
        let mut len = 0;
        let mut chars = text.chars();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                for esc in chars.by_ref() {
                    if esc == 'm' {
                        break;
                    }
                }
            } else {
                len += 1;
            }
        }
        len
    }

    #[test]
    fn countdown_pads_to_hh_mm_ss() {
        assert_eq!(format_countdown(Duration::from_secs(0)), "00:00:00");
        assert_eq!(format_countdown(Duration::from_secs(5)), "00:00:05");
        assert_eq!(format_countdown(Duration::from_secs(65)), "00:01:05");
        assert_eq!(format_countdown(Duration::from_secs(3600)), "01:00:00");
        assert_eq!(format_countdown(Duration::from_secs(3661)), "01:01:01");
        assert_eq!(format_countdown(Duration::from_secs(360_000)), "100:00:00");
    }

    #[test]
    fn short_model_name_renders_new_models_without_a_table() {
        // The reported regression: an unmapped model truncated to "claude-opus-".
        assert_eq!(
            display_model_name("claude-opus-4-8"),
            "Opus 4.8"
        );
        assert_eq!(
            display_model_name("claude-opus-4-7"),
            "Opus 4.7"
        );
        assert_eq!(display_model_name("opus"), "Opus");
        assert_eq!(display_model_name("<synthetic>"), "synthetic");
        assert_eq!(display_model_name("gpt-5.5"), "GPT-5.5");
        assert_eq!(
            display_model_name("gpt-5.5 (high)"),
            "GPT-5.5(H)"
        );
        assert_eq!(
            display_model_name("gpt-5.5:xhigh"),
            "GPT-5.5(XH)"
        );
        assert_eq!(
            display_model_name("gemini-3.2-pro-preview"),
            "Gem 3.2 Pro"
        );
        // Meta-tool `omp`: family inferred from the id's real provider prefix.
        assert_eq!(
            display_model_name("anthropic/claude-opus-4-8"),
            "Opus 4.8"
        );
    }

    #[test]
    fn name_area_renders_to_fixed_width_in_every_layout() {
        for (mode, show_harness) in [
            ("full", true),
            ("full", false),
            ("medium", true),
            ("medium", false),
        ] {
            let l = name_layout(mode, show_harness);
            let composed = compose_name(&l, "Anthropic", "claude-opus-4-8", "Claude Code");
            assert_eq!(composed.chars().count(), l.total, "{mode}/{show_harness}");
            let header = compose_name(&l, "Vendor", "Model", "Harness");
            assert_eq!(header.chars().count(), l.total, "{mode}/{show_harness}");
        }
    }

    #[test]
    fn narrow_name_folds_harness_tag_only_outside_model_view() {
        let row = DataRow {
            vendor: crate::model_id::Vendor::OpenAI,
            vendor_label: "OpenAI".to_string(),
            model_label: "GPT-5.5".to_string(),
            model_raw: "gpt-5.5".to_string(),
            harness_label: "Codex".to_string(),
            harness_short: "Cdx".to_string(),
            metrics: RowMetrics::default(),
        };
        assert_eq!(narrow_name(&row, true, TableView::Flat), "Cdx:GPT-5.5");
        assert_eq!(narrow_name(&row, true, TableView::Model), "GPT-5.5");
        assert_eq!(narrow_name(&row, false, TableView::Flat), "GPT-5.5");
    }

    #[test]
    fn percent_cells_fit_their_requested_width() {
        let wide_total =
            format_with_dual_pct(10_569_375_339, 10_569_375_339, 10_569_375_339, 26, false);
        assert_eq!(visible_len(&wide_total), 26);

        let tight_total =
            format_with_dual_pct(10_569_375_339, 10_569_375_339, 10_569_375_339, 8, true);
        assert!(visible_len(&tight_total) <= 8);
    }

    #[test]
    fn metrics_drive_cost_cells_and_rates() {
        let row = ModelBreakdownRow {
            model: "gpt-5.5".to_string(),
            tool: "codex".to_string(),
            count: 3,
            input: 1_000_000,
            output: 400_000,
            cache_creation: 0,
            cache_read: 600_000,
            reasoning: 0,
            thinking: 0,
            total: 1_400_000,
            total_with_cache: 2_000_000,
            input_cost: 2.0,
            output_cost: 5.0,
            cache_read_cost: 1.0,
            cache_creation_cost: 2.0,
        };
        let metrics = RowMetrics::from_breakdown(&row);

        assert_eq!(metrics.cost(), 10.0);
        assert_eq!(metrics.cost_per_mtok(), 5.0);

        let cell = format_model_cost_with_col_pct(metrics.cost(), 20.0, 16);
        assert_eq!(visible_len(&cell), 16);
    }
}
