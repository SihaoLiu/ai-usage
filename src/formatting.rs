use crate::constants::SubscriptionFees;
use crate::model_id::{parse_model_identity, short_label};
use crate::model_overrides;
use crate::stats::ModelBreakdownRow;
use crate::tool::Tool;

/// Get short display name for a model.
///
/// Resolution order: user override file (`models.toml`) wins, otherwise the
/// label is derived algorithmically from the id. The `_tool` hint is unused
/// on purpose -- the parser infers the provider from the id itself, which is
/// this lets tool logs that carry a real `provider/` prefix and freshly
/// released models render correctly with no code change.
pub fn get_short_model_name(model: &str, _tool: &str) -> String {
    if let Some(label) = model_overrides::load().display.get(model) {
        return label.clone();
    }
    short_label(&parse_model_identity(model))
}

fn format_model_name_with_tool_prefix(
    model: &str,
    tool: &str,
    show_tool_prefix: bool,
    use_short_name: bool,
    prefix_width: usize,
) -> String {
    let name = if use_short_name {
        get_short_model_name(model, tool)
    } else {
        model.to_string()
    };
    if !show_tool_prefix {
        return name;
    }
    let display_tool = Tool::from_key(tool).map(Tool::display_name).unwrap_or(tool);
    format!("{:<width$}: {}", display_tool, name, width = prefix_width)
}

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
const WATERMARK_COLOR: &str = "\x1b[38;5;240m"; // muted gray for background watermark
const INTEGRITY_CHECKING_COLOR: &str = "\x1b[38;5;143m"; // muted yellow
const INTEGRITY_CHECKED_COLOR: &str = "\x1b[38;5;108m"; // muted green
const INTEGRITY_FAILED_COLOR: &str = "\x1b[38;5;203m"; // muted red
const COLOR_RESET: &str = "\x1b[0m";

/// Format a remaining duration as a zero-padded `HH:MM:SS` countdown string.
pub fn format_countdown(remaining: std::time::Duration) -> String {
    let total = remaining.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Watermark text shown as a dimmed placeholder in the monitor-mode prompt.
/// Returns `(colored_string, visible_width)`. The visible width is needed by
/// callers so they can move the cursor back over the watermark after rendering it.
/// When `refresh_in` is `Some`, a `refresh in HH:MM:SS` countdown is woven into
/// the text so the user can see how long until the next auto-refresh.
pub fn prompt_watermark(refresh_in: Option<std::time::Duration>) -> (String, usize) {
    let text = match refresh_in {
        Some(remaining) => format!(
            "ai-usage by SihaoLiu, v{}, refresh in {}, enter h or help for usage",
            env!("CARGO_PKG_VERSION"),
            format_countdown(remaining)
        ),
        None => format!(
            "ai-usage by SihaoLiu, v{}, enter h or help for usage",
            env!("CARGO_PKG_VERSION")
        ),
    };
    prompt_placeholder(&text)
}

pub fn prompt_placeholder(text: &str) -> (String, usize) {
    colored_prompt_placeholder(text, WATERMARK_COLOR)
}

pub fn integrity_checking_marker() -> (String, usize) {
    colored_prompt_placeholder("Integrity Checking", INTEGRITY_CHECKING_COLOR)
}

pub fn integrity_checked_marker(duration: &str) -> (String, usize) {
    colored_prompt_placeholder(
        &format!("Integrity Checked in {duration}"),
        INTEGRITY_CHECKED_COLOR,
    )
}

pub fn integrity_failed_marker() -> (String, usize) {
    colored_prompt_placeholder("Integrity Failed", INTEGRITY_FAILED_COLOR)
}

fn colored_prompt_placeholder(text: &str, color: &str) -> (String, usize) {
    let visible = text.chars().count();
    let colored = format!("{}{}{}", color, text, COLOR_RESET);
    (colored, visible)
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

fn model_api_cost(stats: &ModelBreakdownRow) -> f64 {
    stats.input_cost + stats.output_cost + stats.cache_read_cost + stats.cache_creation_cost
}

fn model_cost_per_mtok(stats: &ModelBreakdownRow) -> f64 {
    let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
    let total_tokens = cache_hit + prefill + decoding;
    if total_tokens > 0 {
        model_api_cost(stats) / (total_tokens as f64 / 1_000_000.0)
    } else {
        0.0
    }
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
        "medium" => 126,
        "compact" => 72,
        "minimal" => 54,
        _ => 0,
    }
}

/// Get strategy totals for a model breakdown row.
fn get_strategy_totals(stats: &ModelBreakdownRow) -> (i64, i64, i64) {
    let cache_hit = stats.cache_read;
    match stats.tool.as_str() {
        "claude" | "kimi" | "omp" => {
            let prefill = stats.input + stats.cache_creation;
            let decoding = stats.output;
            (cache_hit, prefill, decoding)
        }
        "codex" => {
            let prefill = stats.input;
            let decoding = stats.output + stats.reasoning;
            (cache_hit, prefill, decoding)
        }
        "gemini" => {
            let prefill = stats.input;
            let decoding = stats.output + stats.thinking;
            (cache_hit, prefill, decoding)
        }
        _ => (cache_hit, stats.input, stats.output),
    }
}

/// Get strategy costs grouped by buckets for display.
fn get_strategy_costs(
    input_cost: f64,
    output_cost: f64,
    cache_output_cost: f64,
    cache_input_cost: f64,
    tool: &str,
) -> (f64, f64, f64) {
    let cache_hit_cost = cache_input_cost;
    match tool {
        "claude" | "kimi" | "omp" => {
            let prefill_cost = input_cost + cache_output_cost;
            let decoding_cost = output_cost;
            (cache_hit_cost, prefill_cost, decoding_cost)
        }
        _ => {
            let prefill_cost = input_cost;
            let decoding_cost = output_cost + cache_output_cost;
            (cache_hit_cost, prefill_cost, decoding_cost)
        }
    }
}

/// Print model breakdown table with responsive formatting.
/// Returns true if table was printed, false if hidden.
pub fn print_model_breakdown(
    model_stats: &[ModelBreakdownRow],
    days_in_data: f64,
    terminal_width: Option<u16>,
    terminal_height: Option<u16>,
    tool: &str,
    subscription_fees: &SubscriptionFees,
) -> bool {
    // Calculate sums
    let mut sum_messages: i64 = 0;
    let mut sum_cache_hit: i64 = 0;
    let mut sum_prefill: i64 = 0;
    let mut sum_decoding: i64 = 0;
    let mut sum_total_with_cache: i64 = 0;

    for stats in model_stats {
        sum_messages += stats.count;
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        sum_cache_hit += cache_hit;
        sum_prefill += prefill;
        sum_decoding += decoding;
        sum_total_with_cache += cache_hit + prefill + decoding;
    }

    // Display every model row; the sums above already cover the full data set.
    let display_stats: Vec<&ModelBreakdownRow> = model_stats.iter().collect();

    // Determine display mode
    let mode = match (terminal_width, terminal_height) {
        (Some(w), Some(h)) => get_table_display_mode(w, h, display_stats.len()),
        _ => "full",
    };

    if mode == "hidden" {
        return false;
    }

    // Calculate costs from ALL rows (not just displayed ones)
    let mut cache_hit_cost: f64 = 0.0;
    let mut prefill_cost: f64 = 0.0;
    let mut decoding_cost: f64 = 0.0;

    let subscription_price = subscription_fees.get(tool);

    for stats in model_stats {
        // Costs are pre-computed per-entry during aggregation (so tiered
        // pricing for Claude 1M-context models is correct), so we just
        // re-bucket the four components into the display strategy.
        let (row_ch, row_pf, row_dc) = get_strategy_costs(
            stats.input_cost,
            stats.output_cost,
            stats.cache_creation_cost,
            stats.cache_read_cost,
            &stats.tool,
        );
        cache_hit_cost += row_ch;
        prefill_cost += row_pf;
        decoding_cost += row_dc;
    }

    let total_cost = cache_hit_cost + prefill_cost + decoding_cost;

    let show_tool_prefix = tool == "all";

    let tw = terminal_width.unwrap_or(200) as usize;

    // Print table based on mode (display rows only, sums include all data)
    match mode {
        "full" => print_table_full(
            &display_stats,
            sum_messages,
            sum_cache_hit,
            sum_prefill,
            sum_decoding,
            sum_total_with_cache,
            total_cost,
            cache_hit_cost,
            prefill_cost,
            decoding_cost,
            tool,
            show_tool_prefix,
            tw,
        ),
        "medium" => print_table_medium(
            &display_stats,
            sum_messages,
            sum_cache_hit,
            sum_prefill,
            sum_decoding,
            sum_total_with_cache,
            total_cost,
            cache_hit_cost,
            prefill_cost,
            decoding_cost,
            tool,
            show_tool_prefix,
            tw,
        ),
        "compact" => print_table_compact(
            &display_stats,
            sum_messages,
            sum_cache_hit,
            sum_prefill,
            sum_decoding,
            sum_total_with_cache,
            total_cost,
            cache_hit_cost,
            prefill_cost,
            decoding_cost,
            tool,
            show_tool_prefix,
            tw,
        ),
        "minimal" => print_table_minimal(
            &display_stats,
            sum_messages,
            sum_cache_hit,
            sum_prefill,
            sum_decoding,
            sum_total_with_cache,
            total_cost,
            cache_hit_cost,
            prefill_cost,
            decoding_cost,
            tool,
            show_tool_prefix,
            tw,
        ),
        _ => {}
    }

    // Print cost summary
    let daily_cost = if days_in_data > 0.0 {
        total_cost / days_in_data
    } else {
        0.0
    };
    let weekly_cost = daily_cost * 7.0;
    let monthly_cost = daily_cost * 30.0;
    let savings = monthly_cost - subscription_price;
    let monthly_tokens = if days_in_data > 0.0 {
        (sum_total_with_cache as f64 / days_in_data) * 30.0
    } else {
        0.0
    };
    let cost_per_mtok = if monthly_tokens > 0.0 {
        subscription_price / (monthly_tokens / 1_000_000.0)
    } else {
        0.0
    };

    let table_width = get_table_width(mode);
    let cost_pad = center_pad(tw, table_width);

    if mode == "full" || mode == "medium" {
        let line = format!(
            "Daily: ${:.2}, Weekly: ${:.2}, Monthly(30d): ${:.2}, Monthly Saving ${:.2}, {} / MTok",
            daily_cost,
            weekly_cost,
            monthly_cost,
            savings,
            format_cost_per_mtok(cost_per_mtok)
        );
        println!("{}{}", cost_pad, line);
        if let Some(line) = top_model_insight_line(&display_stats, total_cost, show_tool_prefix) {
            println!("{}{}", cost_pad, fit_text_to_width(&line, table_width));
        }
    } else {
        let line = format!(
            "Daily: ${:.2}, Monthly: ${:.2}, Saving: ${:.2}",
            daily_cost, monthly_cost, savings
        );
        println!("{}{}", cost_pad, line);
    }

    true
}

fn top_model_insight_line(
    model_stats: &[&ModelBreakdownRow],
    total_cost: f64,
    show_tool_prefix: bool,
) -> Option<String> {
    let top_spend = model_stats
        .iter()
        .copied()
        .max_by(|a, b| model_api_cost(a).total_cmp(&model_api_cost(b)))?;
    let highest_rate = model_stats
        .iter()
        .copied()
        .filter(|row| {
            let (cache_hit, prefill, decoding) = get_strategy_totals(row);
            cache_hit + prefill + decoding > 0
        })
        .max_by(|a, b| model_cost_per_mtok(a).total_cmp(&model_cost_per_mtok(b)))?;

    let spend_name = fit_text_to_width(
        &format_model_name_with_tool_prefix(
            &top_spend.model,
            &top_spend.tool,
            show_tool_prefix,
            true,
            0,
        ),
        28,
    );
    let rate_name = fit_text_to_width(
        &format_model_name_with_tool_prefix(
            &highest_rate.model,
            &highest_rate.tool,
            show_tool_prefix,
            true,
            0,
        ),
        28,
    );
    let spend = model_api_cost(top_spend);
    let spend_pct = if total_cost > 0.0 {
        spend / total_cost * 100.0
    } else {
        0.0
    };

    Some(format!(
        "Top spend: {} {} ({:.0}%) | Highest rate: {} {} / MTok",
        spend_name,
        format_cost_compact(spend),
        spend_pct,
        rate_name,
        format_cost_per_mtok(model_cost_per_mtok(highest_rate))
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

#[allow(clippy::too_many_arguments)]
fn print_table_full(
    model_stats: &[&ModelBreakdownRow],
    sum_messages: i64,
    sum_cache_hit: i64,
    sum_prefill: i64,
    sum_decoding: i64,
    sum_total_with_cache: i64,
    total_cost: f64,
    cache_hit_cost: f64,
    prefill_cost: f64,
    decoding_cost: f64,
    _tool: &str,
    show_tool_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 35;
    let w_msgs = 18;
    let w_cache = 26;
    let w_cost = 16;
    let w_rate = 12;
    let table_width = get_table_width("full");
    let p = center_pad(terminal_width, table_width);
    println!();
    println!(
        "{}{:^width$}",
        p,
        "Usage / API Cost by Model",
        width = table_width
    );
    println!("{}{}", p, "=".repeat(table_width));
    println!("{}{}", p, dual_pct_legend(table_width));

    println!(
        "{}| {:<wm$} {:>wn$} | {:>wc$} {:>wc$} {:>wc$} {:>wc$} {:>wcost$} {:>wrate$} |",
        p,
        "Model",
        "Messages",
        "Cache Hit",
        "Prefill",
        "Decoding",
        "Total",
        "Cost",
        "$/MTok",
        wm = w_model,
        wn = w_msgs,
        wc = w_cache,
        wcost = w_cost,
        wrate = w_rate,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_tool = &stats.tool;
        let model_name = fit_text_to_width(
            &format_model_name_with_tool_prefix(
                &stats.model,
                effective_tool,
                show_tool_prefix,
                false,
                0,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        let row_total = cache_hit + prefill + decoding;
        let row_cost = model_api_cost(stats);
        println!(
            "{}| {:<wm$} {} | {} {} {} {} {} {} |",
            p,
            model_name,
            format_with_col_pct(stats.count, sum_messages, w_msgs),
            format_with_dual_pct(cache_hit, sum_cache_hit, row_total, w_cache, false),
            format_with_dual_pct(prefill, sum_prefill, row_total, w_cache, false),
            format_with_dual_pct(decoding, sum_decoding, row_total, w_cache, false),
            format_with_dual_pct(row_total, sum_total_with_cache, row_total, w_cache, false),
            format_model_cost_with_col_pct(row_cost, total_cost, w_cost),
            format_rate_for_width(model_cost_per_mtok(stats), w_rate),
            wm = w_model,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<wm$} {} | {} {} {} {} {} {} |",
        p,
        "TOTAL",
        format_with_col_pct(sum_messages, sum_messages, w_msgs),
        format_with_dual_pct(
            sum_cache_hit,
            sum_cache_hit,
            sum_total_with_cache,
            w_cache,
            false
        ),
        format_with_dual_pct(
            sum_prefill,
            sum_prefill,
            sum_total_with_cache,
            w_cache,
            false
        ),
        format_with_dual_pct(
            sum_decoding,
            sum_decoding,
            sum_total_with_cache,
            w_cache,
            false
        ),
        format_with_dual_pct(
            sum_total_with_cache,
            sum_total_with_cache,
            sum_total_with_cache,
            w_cache,
            false
        ),
        format_model_cost_with_col_pct(total_cost, total_cost, w_cost),
        format_rate_for_width(
            if sum_total_with_cache > 0 {
                total_cost / (sum_total_with_cache as f64 / 1_000_000.0)
            } else {
                0.0
            },
            w_rate
        ),
        wm = w_model,
    );

    println!(
        "{}| {:<wm$} {:>wn$} | {} {} {} {} {} {} |",
        p,
        "Cost(API)",
        "",
        format_cost_with_row_pct(cache_hit_cost, total_cost, w_cache),
        format_cost_with_row_pct(prefill_cost, total_cost, w_cache),
        format_cost_with_row_pct(decoding_cost, total_cost, w_cache),
        format_cost_with_row_pct(total_cost, total_cost, w_cache),
        format_cost_with_row_pct(total_cost, total_cost, w_cost),
        format_rate_for_width(
            if sum_total_with_cache > 0 {
                total_cost / (sum_total_with_cache as f64 / 1_000_000.0)
            } else {
                0.0
            },
            w_rate
        ),
        wm = w_model,
        wn = w_msgs,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

#[allow(clippy::too_many_arguments)]
fn print_table_medium(
    model_stats: &[&ModelBreakdownRow],
    sum_messages: i64,
    sum_cache_hit: i64,
    sum_prefill: i64,
    sum_decoding: i64,
    sum_total_with_cache: i64,
    total_cost: f64,
    cache_hit_cost: f64,
    prefill_cost: f64,
    decoding_cost: f64,
    _tool: &str,
    show_tool_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 22;
    let w_msgs = 15;
    let w_cache = 16;
    let w_cost = 13;
    let table_width = get_table_width("medium");
    let p = center_pad(terminal_width, table_width);

    println!();
    println!(
        "{}{:^width$}",
        p,
        "Usage / API Cost by Model",
        width = table_width
    );
    println!("{}{}", p, "=".repeat(table_width));
    println!("{}{}", p, dual_pct_legend(table_width));

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_cache$} {:>w_cache$} {:>w_cache$} {:>w_cache$} {:>w_cost$} |",
        p,
        "Model",
        "Msgs",
        "CacheHit",
        "Prefill",
        "Decode",
        "Total",
        "Cost",
        w_model = w_model,
        w_msgs = w_msgs,
        w_cache = w_cache,
        w_cost = w_cost,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_tool = &stats.tool;
        let model_name = fit_text_to_width(
            &format_model_name_with_tool_prefix(
                &stats.model,
                effective_tool,
                show_tool_prefix,
                true,
                0,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        let row_total = cache_hit + prefill + decoding;
        let row_cost = model_api_cost(stats);
        println!(
            "{}| {:<w_model$} {} | {} {} {} {} {} |",
            p,
            model_name,
            format_with_col_pct(stats.count, sum_messages, w_msgs),
            format_with_dual_pct(cache_hit, sum_cache_hit, row_total, w_cache, true),
            format_with_dual_pct(prefill, sum_prefill, row_total, w_cache, true),
            format_with_dual_pct(decoding, sum_decoding, row_total, w_cache, true),
            format_with_dual_pct(row_total, sum_total_with_cache, row_total, w_cache, true),
            format_model_cost_with_col_pct(row_cost, total_cost, w_cost),
            w_model = w_model,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<w_model$} {} | {} {} {} {} {} |",
        p,
        "TOTAL",
        format_with_col_pct(sum_messages, sum_messages, w_msgs),
        format_with_dual_pct(
            sum_cache_hit,
            sum_cache_hit,
            sum_total_with_cache,
            w_cache,
            true
        ),
        format_with_dual_pct(
            sum_prefill,
            sum_prefill,
            sum_total_with_cache,
            w_cache,
            true
        ),
        format_with_dual_pct(
            sum_decoding,
            sum_decoding,
            sum_total_with_cache,
            w_cache,
            true
        ),
        format_with_dual_pct(
            sum_total_with_cache,
            sum_total_with_cache,
            sum_total_with_cache,
            w_cache,
            true
        ),
        format_model_cost_with_col_pct(total_cost, total_cost, w_cost),
        w_model = w_model,
    );

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {} {} {} {} {} |",
        p,
        "Cost(API)",
        "",
        format_cost_with_row_pct(cache_hit_cost, total_cost, w_cache),
        format_cost_with_row_pct(prefill_cost, total_cost, w_cache),
        format_cost_with_row_pct(decoding_cost, total_cost, w_cache),
        format_cost_with_row_pct(total_cost, total_cost, w_cache),
        format_cost_with_row_pct(total_cost, total_cost, w_cost),
        w_model = w_model,
        w_msgs = w_msgs,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

#[allow(clippy::too_many_arguments)]
fn print_table_compact(
    model_stats: &[&ModelBreakdownRow],
    sum_messages: i64,
    sum_cache_hit: i64,
    sum_prefill: i64,
    sum_decoding: i64,
    sum_total_with_cache: i64,
    total_cost: f64,
    cache_hit_cost: f64,
    prefill_cost: f64,
    decoding_cost: f64,
    _tool: &str,
    show_tool_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 12;
    let w_msgs = 7;
    let w_val = 8;
    let w_cost = 9;
    let table_width = get_table_width("compact");
    let p = center_pad(terminal_width, table_width);

    println!();
    println!("{}{:^width$}", p, "Usage / API Cost", width = table_width);
    println!("{}{}", p, "=".repeat(table_width));

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} {:>w_cost$} |",
        p,
        "Model",
        "Msgs",
        "CacheHit",
        "Prefill",
        "Decode",
        "Total",
        "Cost",
        w_model = w_model,
        w_msgs = w_msgs,
        w_val = w_val,
        w_cost = w_cost,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_tool = &stats.tool;
        let model_name = fit_text_to_width(
            &format_model_name_with_tool_prefix(
                &stats.model,
                effective_tool,
                show_tool_prefix,
                true,
                0,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        println!(
            "{}| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} {} |",
            p,
            model_name,
            format_number_compact(stats.count),
            format_number_compact(cache_hit),
            format_number_compact(prefill),
            format_number_compact(decoding),
            format_number_compact(cache_hit + prefill + decoding),
            format_cost_for_width(model_api_cost(stats), w_cost),
            w_model = w_model,
            w_msgs = w_msgs,
            w_val = w_val,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} {} |",
        p,
        "TOTAL",
        format_number_compact(sum_messages),
        format_number_compact(sum_cache_hit),
        format_number_compact(sum_prefill),
        format_number_compact(sum_decoding),
        format_number_compact(sum_total_with_cache),
        format_cost_for_width(total_cost, w_cost),
        w_model = w_model,
        w_msgs = w_msgs,
        w_val = w_val,
    );

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {} {} {} {} {} |",
        p,
        "Cost",
        "",
        format_cost_for_width(cache_hit_cost, w_val),
        format_cost_for_width(prefill_cost, w_val),
        format_cost_for_width(decoding_cost, w_val),
        format_cost_for_width(total_cost, w_val),
        format_cost_for_width(total_cost, w_cost),
        w_model = w_model,
        w_msgs = w_msgs,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

#[allow(clippy::too_many_arguments)]
fn print_table_minimal(
    model_stats: &[&ModelBreakdownRow],
    sum_messages: i64,
    _sum_cache_hit: i64,
    _sum_prefill: i64,
    _sum_decoding: i64,
    sum_total_with_cache: i64,
    total_cost: f64,
    _cache_hit_cost: f64,
    _prefill_cost: f64,
    _decoding_cost: f64,
    _tool: &str,
    show_tool_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 12;
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
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_tokens$} {:>w_cost$} {:>w_rate$} |",
        p,
        "Model",
        "Msgs",
        "Tokens",
        "Cost",
        "$/MTok",
        w_model = w_model,
        w_msgs = w_msgs,
        w_tokens = w_tokens,
        w_cost = w_cost,
        w_rate = w_rate,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_tool = &stats.tool;
        let model_name = fit_text_to_width(
            &format_model_name_with_tool_prefix(
                &stats.model,
                effective_tool,
                show_tool_prefix,
                true,
                0,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        let row_tokens = cache_hit + prefill + decoding;
        println!(
            "{}| {:<w_model$} {:>w_msgs$} | {:>w_tokens$} {} {} |",
            p,
            model_name,
            format_number_compact(stats.count),
            format_number_compact(row_tokens),
            format_cost_for_width(model_api_cost(stats), w_cost),
            format_rate_for_width(model_cost_per_mtok(stats), w_rate),
            w_model = w_model,
            w_msgs = w_msgs,
            w_tokens = w_tokens,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_tokens$} {} {} |",
        p,
        "TOTAL",
        format_number_compact(sum_messages),
        format_number_compact(sum_total_with_cache),
        format_cost_for_width(total_cost, w_cost),
        format_rate_for_width(
            if sum_total_with_cache > 0 {
                total_cost / (sum_total_with_cache as f64 / 1_000_000.0)
            } else {
                0.0
            },
            w_rate
        ),
        w_model = w_model,
        w_msgs = w_msgs,
        w_tokens = w_tokens,
    );

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_tokens$} {} {} |",
        p,
        "Cost",
        "",
        "",
        format_cost_for_width(total_cost, w_cost),
        format_rate_for_width(
            if sum_total_with_cache > 0 {
                total_cost / (sum_total_with_cache as f64 / 1_000_000.0)
            } else {
                0.0
            },
            w_rate
        ),
        w_model = w_model,
        w_msgs = w_msgs,
        w_tokens = w_tokens,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

#[cfg(test)]
mod tests {
    use super::*;
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
            get_short_model_name("claude-opus-4-8", "claude"),
            "Opus 4.8"
        );
        assert_eq!(
            get_short_model_name("claude-opus-4-7", "claude"),
            "Opus 4.7"
        );
        assert_eq!(get_short_model_name("opus", "claude"), "Opus");
        assert_eq!(get_short_model_name("<synthetic>", "claude"), "synthetic");
        assert_eq!(get_short_model_name("gpt-5.5", "codex"), "GPT-5.5");
        assert_eq!(
            get_short_model_name("gpt-5.5 (high)", "codex"),
            "GPT-5.5(H)"
        );
        assert_eq!(
            get_short_model_name("gpt-5.5:xhigh", "codex"),
            "GPT-5.5(XH)"
        );
        assert_eq!(
            get_short_model_name("gemini-3.2-pro-preview", "gemini"),
            "Gem 3.2 Pro"
        );
        // Meta-tool `omp`: family inferred from the id's real provider prefix.
        assert_eq!(
            get_short_model_name("anthropic/claude-opus-4-8", "omp"),
            "Opus 4.8"
        );
    }

    #[test]
    fn all_mode_prefixes_model_rows_with_tool_names() {
        assert_eq!(
            format_model_name_with_tool_prefix("gpt-5.5", "codex", true, true, 0),
            "Codex: GPT-5.5"
        );
        assert_eq!(
            format_model_name_with_tool_prefix("gemini-3.2-pro-preview", "gemini", true, true, 0),
            "Gemini CLI: Gem 3.2 Pro"
        );
        assert_eq!(
            format_model_name_with_tool_prefix("claude-opus-4-8", "claude", true, true, 0),
            "Claude Code: Opus 4.8"
        );
    }

    #[test]
    fn watermark_weaves_countdown_when_present() {
        let (text, _) = prompt_watermark(Some(Duration::from_secs(3661)));
        assert!(text.contains("refresh in 01:01:01"));
        assert!(text.contains("enter h or help for usage"));

        let (plain, _) = prompt_watermark(None);
        assert!(!plain.contains("refresh in"));
        assert!(plain.contains("enter h or help for usage"));
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
    fn model_cost_helpers_report_total_cost_and_rate() {
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

        assert_eq!(model_api_cost(&row), 10.0);
        assert_eq!(model_cost_per_mtok(&row), 5.0);

        let cell = format_model_cost_with_col_pct(model_api_cost(&row), 20.0, 16);
        assert_eq!(visible_len(&cell), 16);
    }
}
