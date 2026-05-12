use std::collections::HashMap;

use crate::constants::SubscriptionFees;
use crate::stats::ModelBreakdownRow;

// Short model name mappings for Claude
fn claude_short_model_names() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("claude-opus-4-7", "Opus 4.7"),
        ("claude-opus-4-6", "Opus 4.6"),
        ("claude-opus-4-5-20251101", "Opus 4.5"),
        ("claude-opus-4-1-20250805", "Opus 4.1"),
        ("claude-sonnet-4-6", "Sonnet 4.6"),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5"),
        ("claude-sonnet-4-20250514", "Sonnet 4"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5"),
        ("<synthetic>", "synthetic"),
    ])
}

// Short model name mappings for Codex
fn codex_short_model_names() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("gpt-5.4", "GPT-5.4"),
        ("gpt-5.4-codex", "GPT-5.4 Cdx"),
        ("gpt-5.4-mini", "GPT-5.4 Mini"),
        ("gpt-5.4-nano", "GPT-5.4 Nano"),
        ("gpt-5.3-codex", "GPT-5.3 Cdx"),
        ("gpt-5.3-codex-spark", "GPT-5.3 Sprk"),
        ("gpt-5.2-codex", "GPT-5.2 Cdx"),
        ("gpt-5.1-codex", "GPT-5.1 Cdx"),
        ("gpt-5.1-codex-max", "GPT-5.1 Max"),
        ("gpt-5.1-codex-mini", "GPT-5.1 Mini"),
        ("gpt-5-codex", "GPT-5 Codex"),
        ("codex-mini-latest", "Codex Mini"),
        ("gpt-5.2", "GPT-5.2"),
        ("gpt-5.1", "GPT-5.1"),
        ("gpt-5", "GPT-5"),
        ("gpt-5-mini", "GPT-5 Mini"),
        ("gpt-5-nano", "GPT-5 Nano"),
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-4.1-mini", "GPT-4.1 Mini"),
        ("gpt-4.1-nano", "GPT-4.1 Nano"),
        ("o1", "o1"),
        ("o3", "o3"),
        ("o3-mini", "o3-mini"),
        ("o4-mini", "o4-mini"),
    ])
}

// Short model name mappings for Gemini
fn gemini_short_model_names() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("gemini-3.1-pro-preview", "Gem 3.1 Pro"),
        ("gemini-3.1-flash-lite-preview", "Gem 3.1 Lt"),
        ("gemini-3-pro-preview", "Gem 3 Pro"),
        ("gemini-3-pro-image-preview", "Gem 3 Img"),
        ("gemini-3-flash-preview", "Gem 3 Fl"),
        ("gemini-2.5-pro", "Gem 2.5 Pro"),
        ("gemini-2.5-flash", "Gem 2.5 Fl"),
        ("gemini-2.5-flash-preview-09-2025", "Gem 2.5 Fl"),
        ("gemini-2.5-flash-lite", "Gem 2.5 Lt"),
        ("gemini-2.5-flash-lite-preview-09-2025", "Gem 2.5 Lt"),
        ("gemini-2.0-flash", "Gem 2.0 Fl"),
        ("gemini-2.0-flash-lite", "Gem 2.0 Lt"),
    ])
}

/// Get short display name for a model.
pub fn get_short_model_name(model: &str, vendor: &str) -> String {
    match vendor {
        "codex" => {
            // Some model names may include effort level in parentheses.
            if model.contains(" (")
                && model.ends_with(')')
                && let Some(idx) = model.rfind(" (")
            {
                let base_model = &model[..idx];
                let effort = &model[idx + 2..model.len() - 1];
                let names = codex_short_model_names();
                let short_base = names
                    .get(base_model)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| truncate(base_model, 10));
                let effort_short = match effort {
                    "low" => "L",
                    "medium" => "M",
                    "high" => "H",
                    "xhigh" => "XH",
                    _ => &effort[..1],
                };
                return format!("{}({})", short_base, effort_short);
            }
            let names = codex_short_model_names();
            names
                .get(model)
                .map(|s| s.to_string())
                .unwrap_or_else(|| truncate(model, 12))
        }
        "gemini" => {
            let names = gemini_short_model_names();
            names
                .get(model)
                .map(|s| s.to_string())
                .unwrap_or_else(|| truncate(model, 12))
        }
        _ => {
            let names = claude_short_model_names();
            names
                .get(model)
                .map(|s| s.to_string())
                .unwrap_or_else(|| truncate(model, 12))
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        s[..max_len].to_string()
    }
}

fn format_model_name_with_vendor_prefix(
    model: &str,
    vendor: &str,
    show_vendor_prefix: bool,
    use_short_name: bool,
    prefix_width: usize,
) -> String {
    let name = if use_short_name {
        get_short_model_name(model, vendor)
    } else {
        model.to_string()
    };
    if !show_vendor_prefix {
        return name;
    }
    let display_vendor = match vendor {
        "claude" => "Claude",
        "codex" => "OpenAI",
        "gemini" => "Google",
        _ => vendor,
    };
    format!("{:<width$}: {}", display_vendor, name, width = prefix_width)
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
const COLOR_RESET: &str = "\x1b[0m";

/// Watermark text shown as a dimmed placeholder in the monitor-mode prompt.
/// Returns `(colored_string, visible_width)`. The visible width is needed by
/// callers so they can move the cursor back over the watermark after rendering it.
pub fn prompt_watermark() -> (String, usize) {
    let text = format!(
        "ai-usage by SihaoLiu, v{}, enter h or help for usage",
        env!("CARGO_PKG_VERSION")
    );
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

/// Single column-percentage cell with cyan ↑ arrow (used for Messages column).
fn format_with_col_pct(value: i64, total: i64, width: usize) -> String {
    let pct = if total > 0 {
        value as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let value_str = format_number(value);
    let pct_str = format!("{:.0}%", pct);
    // Visible: value + "(" + "↑" + pct + ")"
    let visible = value_str.chars().count() + pct_str.chars().count() + 3;
    let pad = pad_left(visible, width);
    format!(
        "{pad}{val}(\u{2191}{cc}{pct}{rst})",
        pad = pad,
        val = value_str,
        cc = COL_PCT_COLOR,
        pct = pct_str,
        rst = COLOR_RESET,
    )
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
    let value_str = if compact {
        format_number_compact(value)
    } else {
        format_number(value)
    };
    let col_str = format!("{:.0}%", col_pct);
    let row_str = format!("{:.0}%", row_pct);
    // Visible: value + "(" + "↑" + col + "·" + "←" + row + ")" -> 5 fixed chars
    let visible = value_str.chars().count() + col_str.chars().count() + row_str.chars().count() + 5;
    let pad = pad_left(visible, width);
    format!(
        "{pad}{val}(\u{2191}{cc}{col}{rst}\u{00B7}\u{2190}{rc}{row}{rst})",
        pad = pad,
        val = value_str,
        cc = COL_PCT_COLOR,
        col = col_str,
        rst = COLOR_RESET,
        rc = ROW_PCT_COLOR,
        row = row_str,
    )
}

/// Cost cell with row-percentage (← yellow). Cost row has only one row, so column-%
/// would always be 100% and is omitted.
fn format_cost_with_row_pct(cost: f64, row_total: f64, width: usize) -> String {
    let row_pct = if row_total > 0.0 {
        cost / row_total * 100.0
    } else {
        0.0
    };
    let cost_str = format!("${:.2}", cost);
    let row_str = format!("{:.0}%", row_pct);
    // Visible: cost + "(" + "←" + row + ")" -> 3 fixed chars
    let visible = cost_str.chars().count() + row_str.chars().count() + 3;
    let pad = pad_left(visible, width);
    format!(
        "{pad}{cost}(\u{2190}{rc}{row}{rst})",
        pad = pad,
        cost = cost_str,
        rc = ROW_PCT_COLOR,
        row = row_str,
        rst = COLOR_RESET,
    )
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
        "full" => 168,
        "medium" => 128,
        "compact" => 62,
        "minimal" => 53,
        _ => 0,
    }
}

/// Get strategy totals for a model breakdown row.
fn get_strategy_totals(stats: &ModelBreakdownRow) -> (i64, i64, i64) {
    let cache_hit = stats.cache_read;
    match stats.vendor.as_str() {
        "claude" => {
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
    vendor: &str,
) -> (f64, f64, f64) {
    let cache_hit_cost = cache_input_cost;
    match vendor {
        "claude" => {
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
    vendor: &str,
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

    // Filter rows for display (1% message threshold), but keep all data for sums
    let threshold = (sum_messages as f64 * 0.01) as i64;
    let display_stats: Vec<&ModelBreakdownRow> = model_stats
        .iter()
        .filter(|r| r.count >= threshold)
        .collect();

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

    let subscription_price = subscription_fees.get(vendor);

    for stats in model_stats {
        // Costs are pre-computed per-entry during aggregation (so tiered
        // pricing for Claude 1M-context models is correct), so we just
        // re-bucket the four components into the display strategy.
        let (row_ch, row_pf, row_dc) = get_strategy_costs(
            stats.input_cost,
            stats.output_cost,
            stats.cache_creation_cost,
            stats.cache_read_cost,
            &stats.vendor,
        );
        cache_hit_cost += row_ch;
        prefill_cost += row_pf;
        decoding_cost += row_dc;
    }

    let total_cost = cache_hit_cost + prefill_cost + decoding_cost;

    let show_vendor_prefix = vendor == "all";

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
            vendor,
            show_vendor_prefix,
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
            vendor,
            show_vendor_prefix,
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
            vendor,
            show_vendor_prefix,
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
            vendor,
            show_vendor_prefix,
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

    let table_width = match mode {
        "full" => 168,
        "medium" => 128,
        "compact" => 62,
        _ => 53, // minimal
    };
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
    } else {
        let line = format!(
            "Daily: ${:.2}, Monthly: ${:.2}, Saving: ${:.2}",
            daily_cost, monthly_cost, savings
        );
        println!("{}{}", cost_pad, line);
    }

    true
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
    _vendor: &str,
    show_vendor_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 35;
    let w_msgs = 18;
    let w_cache = 26;
    // 2 ('| ') + w_model + 1 (' ') + w_msgs + 3 (' | ') + 4 * w_cache + 3 (3 spaces) + 2 (' |')
    let table_width = 2 + w_model + 1 + w_msgs + 3 + 4 * w_cache + 3 + 2;
    let p = center_pad(terminal_width, table_width);
    println!();
    println!(
        "{}{:^width$}",
        p,
        "Usage / Cost by Model",
        width = table_width
    );
    println!("{}{}", p, "=".repeat(table_width));
    println!("{}{}", p, dual_pct_legend(table_width));

    println!(
        "{}| {:<wm$} {:>wn$} | {:>wc$} {:>wc$} {:>wc$} {:>wc$} |",
        p,
        "Model",
        "Messages",
        "Cache Hit",
        "Prefill",
        "Decoding",
        "Total",
        wm = w_model,
        wn = w_msgs,
        wc = w_cache,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_vendor = &stats.vendor;
        let model_name = fit_text_to_width(
            &format_model_name_with_vendor_prefix(
                &stats.model,
                effective_vendor,
                show_vendor_prefix,
                false,
                6,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        let row_total = cache_hit + prefill + decoding;
        println!(
            "{}| {:<wm$} {} | {} {} {} {} |",
            p,
            model_name,
            format_with_col_pct(stats.count, sum_messages, w_msgs),
            format_with_dual_pct(cache_hit, sum_cache_hit, row_total, w_cache, false),
            format_with_dual_pct(prefill, sum_prefill, row_total, w_cache, false),
            format_with_dual_pct(decoding, sum_decoding, row_total, w_cache, false),
            format_with_dual_pct(row_total, sum_total_with_cache, row_total, w_cache, false),
            wm = w_model,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<wm$} {} | {} {} {} {} |",
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
        wm = w_model,
    );

    println!(
        "{}| {:<wm$} {:>wn$} | {} {} {} {} |",
        p,
        "Cost(API)",
        "",
        format_cost_with_row_pct(cache_hit_cost, total_cost, w_cache),
        format_cost_with_row_pct(prefill_cost, total_cost, w_cache),
        format_cost_with_row_pct(decoding_cost, total_cost, w_cache),
        format_cost_with_row_pct(total_cost, total_cost, w_cache),
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
    _vendor: &str,
    show_vendor_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 22;
    let w_msgs = 15;
    let w_cache = 20;
    let table_width = 128;
    let p = center_pad(terminal_width, table_width);

    println!();
    println!(
        "{}{:^width$}",
        p,
        "Usage / Cost by Model",
        width = table_width
    );
    println!("{}{}", p, "=".repeat(table_width));
    println!("{}{}", p, dual_pct_legend(table_width));

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_cache$} {:>w_cache$} {:>w_cache$} {:>w_cache$} |",
        p,
        "Model",
        "Msgs",
        "CacheHit",
        "Prefill",
        "Decode",
        "Total",
        w_model = w_model,
        w_msgs = w_msgs,
        w_cache = w_cache,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_vendor = &stats.vendor;
        let model_name = fit_text_to_width(
            &format_model_name_with_vendor_prefix(
                &stats.model,
                effective_vendor,
                show_vendor_prefix,
                true,
                6,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        let row_total = cache_hit + prefill + decoding;
        println!(
            "{}| {:<w_model$} {} | {} {} {} {} |",
            p,
            model_name,
            format_with_col_pct(stats.count, sum_messages, w_msgs),
            format_with_dual_pct(cache_hit, sum_cache_hit, row_total, w_cache, true),
            format_with_dual_pct(prefill, sum_prefill, row_total, w_cache, true),
            format_with_dual_pct(decoding, sum_decoding, row_total, w_cache, true),
            format_with_dual_pct(row_total, sum_total_with_cache, row_total, w_cache, true),
            w_model = w_model,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<w_model$} {} | {} {} {} {} |",
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
        w_model = w_model,
    );

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {} {} {} {} |",
        p,
        "Cost(API)",
        "",
        format_cost_with_row_pct(cache_hit_cost, total_cost, w_cache),
        format_cost_with_row_pct(prefill_cost, total_cost, w_cache),
        format_cost_with_row_pct(decoding_cost, total_cost, w_cache),
        format_cost_with_row_pct(total_cost, total_cost, w_cache),
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
    _vendor: &str,
    show_vendor_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 12;
    let w_msgs = 7;
    let w_val = 8;
    let table_width = 62;
    let p = center_pad(terminal_width, table_width);

    println!();
    println!(
        "{}{:^width$}",
        p,
        "Usage / Cost by Model",
        width = table_width
    );
    println!("{}{}", p, "=".repeat(table_width));

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} |",
        p,
        "Model",
        "Msgs",
        "CacheHit",
        "Prefill",
        "Decode",
        "Total",
        w_model = w_model,
        w_msgs = w_msgs,
        w_val = w_val,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_vendor = &stats.vendor;
        let model_name = fit_text_to_width(
            &format_model_name_with_vendor_prefix(
                &stats.model,
                effective_vendor,
                show_vendor_prefix,
                true,
                6,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        println!(
            "{}| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} |",
            p,
            model_name,
            format_number_compact(stats.count),
            format_number_compact(cache_hit),
            format_number_compact(prefill),
            format_number_compact(decoding),
            format_number_compact(cache_hit + prefill + decoding),
            w_model = w_model,
            w_msgs = w_msgs,
            w_val = w_val,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} |",
        p,
        "TOTAL",
        format_number_compact(sum_messages),
        format_number_compact(sum_cache_hit),
        format_number_compact(sum_prefill),
        format_number_compact(sum_decoding),
        format_number_compact(sum_total_with_cache),
        w_model = w_model,
        w_msgs = w_msgs,
        w_val = w_val,
    );

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | ${:>w_cost$.2} ${:>w_cost$.2} ${:>w_cost$.2} ${:>w_cost$.2} |",
        p,
        "Cost",
        "",
        cache_hit_cost,
        prefill_cost,
        decoding_cost,
        total_cost,
        w_model = w_model,
        w_msgs = w_msgs,
        w_cost = w_val - 1,
    );
    println!("{}{}", p, "=".repeat(table_width));
}

#[allow(clippy::too_many_arguments)]
fn print_table_minimal(
    model_stats: &[&ModelBreakdownRow],
    sum_messages: i64,
    sum_cache_hit: i64,
    sum_prefill: i64,
    sum_decoding: i64,
    sum_total_with_cache: i64,
    total_cost: f64,
    _cache_hit_cost: f64,
    _prefill_cost: f64,
    _decoding_cost: f64,
    _vendor: &str,
    show_vendor_prefix: bool,
    terminal_width: usize,
) {
    let w_model = 12;
    let w_msgs = 7;
    let w_strategy = 10;
    let table_width = 2 + w_model + 1 + w_msgs + 1 + 2 + w_strategy * 4 + 3 + 2;
    let p = center_pad(terminal_width, table_width);

    println!();
    println!("{}{:^width$}", p, "Usage Summary", width = table_width);
    println!("{}{}", p, "=".repeat(table_width));

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} |",
        p,
        "Model",
        "Msgs",
        "Cache Hit",
        "Prefill",
        "Decode",
        "All",
        w_model = w_model,
        w_msgs = w_msgs,
        w_strategy = w_strategy,
    );
    println!("{}|{}|", p, "-".repeat(table_width - 2));

    for stats in model_stats {
        let effective_vendor = &stats.vendor;
        let model_name = fit_text_to_width(
            &format_model_name_with_vendor_prefix(
                &stats.model,
                effective_vendor,
                show_vendor_prefix,
                true,
                6,
            ),
            w_model,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        println!(
            "{}| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} |",
            p,
            model_name,
            format_number_compact(stats.count),
            format_number_compact(cache_hit),
            format_number_compact(prefill),
            format_number_compact(decoding),
            format_number_compact(cache_hit + prefill + decoding),
            w_model = w_model,
            w_msgs = w_msgs,
            w_strategy = w_strategy,
        );
    }

    println!("{}|{}|", p, "-".repeat(table_width - 2));
    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} |",
        p,
        "TOTAL",
        format_number_compact(sum_messages),
        format_number_compact(sum_cache_hit),
        format_number_compact(sum_prefill),
        format_number_compact(sum_decoding),
        format_number_compact(sum_total_with_cache),
        w_model = w_model,
        w_msgs = w_msgs,
        w_strategy = w_strategy,
    );

    println!(
        "{}| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} ${:>w_cost$.2} |",
        p,
        "Cost",
        "",
        "",
        "",
        "",
        total_cost,
        w_model = w_model,
        w_msgs = w_msgs,
        w_strategy = w_strategy,
        w_cost = w_strategy - 1,
    );
    println!("{}{}", p, "=".repeat(table_width));
}
