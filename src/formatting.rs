use std::collections::HashMap;

use crate::constants::{AllPricing, SubscriptionFees};
use crate::stats::ModelBreakdownRow;

// Short model name mappings for Claude
fn claude_short_model_names() -> HashMap<&'static str, &'static str> {
    HashMap::from([
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
            // Codex model names may include effort level like "gpt-5-codex (high)"
            if model.contains(" (") && model.ends_with(')') {
                if let Some(idx) = model.rfind(" (") {
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
    format!("{:<width$}: {}", vendor, name, width = prefix_width)
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
        if i > 0 && (len - i) % 3 == 0 && b != b'-' {
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
    if value >= 1_000_000.0 {
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

/// Format cost per MTok with appropriate precision.
pub fn format_cost_per_mtok(value: f64) -> String {
    if value >= 0.01 {
        format!("${:.2}", value)
    } else if value <= 0.0 {
        "$0.00".to_string()
    } else {
        let decimal_places = (-value.log10()).ceil() as usize;
        format!("${:.prec$}", value, prec = decimal_places)
    }
}

fn format_with_pct(value: i64, total: i64, width: usize) -> String {
    let pct = if total > 0 {
        value as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let s = format!("{}({:4.1}%)", format_number(value), pct);
    format!("{:>width$}", s, width = width)
}

fn format_with_100pct_up(value: i64, width: usize) -> String {
    let s = format!("{}(\u{2191}100%)", format_number(value));
    format!("{:>width$}", s, width = width)
}

fn format_with_100pct_left(value: f64, width: usize) -> String {
    let s = format!("${:.2}(\u{2190}100%)", value);
    format!("{:>width$}", s, width = width)
}

fn format_cost_with_pct(cost: f64, total: f64, width: usize) -> String {
    let pct = if total > 0.0 {
        cost / total * 100.0
    } else {
        0.0
    };
    let s = format!("${:.2}({:4.1}%)", cost, pct);
    format!("{:>width$}", s, width = width)
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
    days_in_data: i64,
    terminal_width: Option<u16>,
    terminal_height: Option<u16>,
    vendor: &str,
    subscription_fees: &SubscriptionFees,
    pricing: &AllPricing,
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

    // Determine display mode
    let mode = match (terminal_width, terminal_height) {
        (Some(w), Some(h)) => get_table_display_mode(w, h, model_stats.len()),
        _ => "full",
    };

    if mode == "hidden" {
        return false;
    }

    // Calculate costs per-row, applying vendor-specific cost strategy per model
    let mut cache_hit_cost: f64 = 0.0;
    let mut prefill_cost: f64 = 0.0;
    let mut decoding_cost: f64 = 0.0;

    let subscription_price = subscription_fees.get(vendor);

    for stats in model_stats {
        let resolved_vendor = &stats.vendor;
        let model = &stats.model;

        // For Codex, extract base model name
        let base_model = if resolved_vendor == "codex" && model.contains(" (") && model.ends_with(')') {
            model.rsplit_once(" (").map(|(base, _)| base).unwrap_or(model)
        } else {
            model.as_str()
        };

        let p = pricing.get_pricing(resolved_vendor, base_model);
        let row_input_cost = stats.input as f64 * p.input / 1_000_000.0;
        let row_output_cost = stats.output as f64 * p.output / 1_000_000.0;
        let row_cache_input_cost = stats.cache_read as f64 * p.cache_input / 1_000_000.0;
        let row_cache_output_cost = match resolved_vendor.as_str() {
            "codex" => stats.reasoning as f64 * p.output / 1_000_000.0,
            "gemini" => stats.thinking as f64 * p.output / 1_000_000.0,
            _ => stats.cache_creation as f64 * p.cache_output / 1_000_000.0,
        };

        let (row_ch, row_pf, row_dc) = get_strategy_costs(
            row_input_cost, row_output_cost, row_cache_output_cost, row_cache_input_cost, resolved_vendor,
        );
        cache_hit_cost += row_ch;
        prefill_cost += row_pf;
        decoding_cost += row_dc;
    }

    let total_cost = cache_hit_cost + prefill_cost + decoding_cost;

    let show_vendor_prefix = vendor == "all";

    // Print table based on mode
    match mode {
        "full" => print_table_full(
            model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding,
            sum_total_with_cache, total_cost, cache_hit_cost, prefill_cost,
            decoding_cost, vendor, show_vendor_prefix,
        ),
        "medium" => print_table_medium(
            model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding,
            sum_total_with_cache, total_cost, cache_hit_cost, prefill_cost,
            decoding_cost, vendor, show_vendor_prefix,
        ),
        "compact" => print_table_compact(
            model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding,
            sum_total_with_cache, total_cost, cache_hit_cost, prefill_cost,
            decoding_cost, vendor, show_vendor_prefix,
        ),
        "minimal" => print_table_minimal(
            model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding,
            sum_total_with_cache, total_cost, cache_hit_cost, prefill_cost,
            decoding_cost, vendor, show_vendor_prefix,
        ),
        _ => {}
    }

    // Print cost summary
    let daily_cost = if days_in_data > 0 {
        total_cost / days_in_data as f64
    } else {
        0.0
    };
    let weekly_cost = daily_cost * 7.0;
    let monthly_cost = daily_cost * 30.0;
    let savings = monthly_cost - subscription_price;
    let monthly_tokens = if days_in_data > 0 {
        (sum_total_with_cache as f64 / days_in_data as f64) * 30.0
    } else {
        0.0
    };
    let cost_per_mtok = if monthly_tokens > 0.0 {
        subscription_price / (monthly_tokens / 1_000_000.0)
    } else {
        0.0
    };

    if mode == "full" || mode == "medium" {
        println!(
            "Daily: ${:.2}, Weekly: ${:.2}, Monthly(30d): ${:.2}, Monthly Saving ${:.2}, {} / MTok",
            daily_cost, weekly_cost, monthly_cost, savings,
            format_cost_per_mtok(cost_per_mtok)
        );
    } else {
        println!(
            "Daily: ${:.2}, Monthly: ${:.2}, Saving: ${:.2}",
            daily_cost, monthly_cost, savings
        );
    }

    true
}

fn print_table_full(
    model_stats: &[ModelBreakdownRow],
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
) {
    let table_width = 152;
    println!("Usage / Cost by Model");
    println!("{}", "=".repeat(table_width));

    println!(
        "| {:<35} {:>18} | {:>22} {:>22} {:>22} {:>22} |",
        "Model", "Messages", "Cache Hit", "Prefill", "Decoding", "Total"
    );
    println!("|{}|", "-".repeat(table_width - 2));

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
            35,
        );
        let (cache_hit, prefill, decoding) = get_strategy_totals(stats);
        println!(
            "| {:<35} {} | {} {} {} {} |",
            model_name,
            format_with_pct(stats.count, sum_messages, 18),
            format_with_pct(cache_hit, sum_cache_hit, 22),
            format_with_pct(prefill, sum_prefill, 22),
            format_with_pct(decoding, sum_decoding, 22),
            format_with_pct(cache_hit + prefill + decoding, sum_total_with_cache, 22),
        );
    }

    println!("|{}|", "-".repeat(table_width - 2));
    println!(
        "| {:<35} {} | {} {} {} {} |",
        "TOTAL",
        format_with_100pct_up(sum_messages, 18),
        format_with_100pct_up(sum_cache_hit, 22),
        format_with_100pct_up(sum_prefill, 22),
        format_with_100pct_up(sum_decoding, 22),
        format_with_100pct_up(sum_total_with_cache, 22),
    );

    println!(
        "| {:<35} {:>18} | {} {} {} {} |",
        "Cost(API)",
        "",
        format_cost_with_pct(cache_hit_cost, total_cost, 22),
        format_cost_with_pct(prefill_cost, total_cost, 22),
        format_cost_with_pct(decoding_cost, total_cost, 22),
        format_with_100pct_left(total_cost, 22),
    );
    println!("{}", "=".repeat(table_width));
}

fn print_table_medium(
    model_stats: &[ModelBreakdownRow],
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
) {
    let w_model = 22;
    let w_msgs = 15;
    let w_cache = 20;
    let table_width = 128;

    println!("Usage / Cost by Model");
    println!("{}", "=".repeat(table_width));

    println!(
        "| {:<w_model$} {:>w_msgs$} | {:>w_cache$} {:>w_cache$} {:>w_cache$} {:>w_cache$} |",
        "Model", "Msgs", "CacheHit", "Prefill", "Decode", "Total",
        w_model = w_model, w_msgs = w_msgs, w_cache = w_cache,
    );
    println!("|{}|", "-".repeat(table_width - 2));

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
            "| {:<w_model$} {} | {} {} {} {} |",
            model_name,
            format_with_pct(stats.count, sum_messages, w_msgs),
            format_with_pct(cache_hit, sum_cache_hit, w_cache),
            format_with_pct(prefill, sum_prefill, w_cache),
            format_with_pct(decoding, sum_decoding, w_cache),
            format_with_pct(cache_hit + prefill + decoding, sum_total_with_cache, w_cache),
            w_model = w_model,
        );
    }

    println!("|{}|", "-".repeat(table_width - 2));
    println!(
        "| {:<w_model$} {} | {} {} {} {} |",
        "TOTAL",
        format_with_100pct_up(sum_messages, w_msgs),
        format_with_100pct_up(sum_cache_hit, w_cache),
        format_with_100pct_up(sum_prefill, w_cache),
        format_with_100pct_up(sum_decoding, w_cache),
        format_with_100pct_up(sum_total_with_cache, w_cache),
        w_model = w_model,
    );

    println!(
        "| {:<w_model$} {:>w_msgs$} {} {} {} {} |",
        "Cost(API)",
        "",
        format_cost_with_pct(cache_hit_cost, total_cost, w_cache),
        format_cost_with_pct(prefill_cost, total_cost, w_cache),
        format_cost_with_pct(decoding_cost, total_cost, w_cache),
        format_with_100pct_left(total_cost, w_cache),
        w_model = w_model, w_msgs = w_msgs,
    );
    println!("{}", "=".repeat(table_width));
}

fn print_table_compact(
    model_stats: &[ModelBreakdownRow],
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
) {
    let w_model = 12;
    let w_msgs = 7;
    let w_val = 8;
    let table_width = 62;

    println!("Usage / Cost by Model");
    println!("{}", "=".repeat(table_width));

    println!(
        "| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} |",
        "Model", "Msgs", "CacheHit", "Prefill", "Decode", "Total",
        w_model = w_model, w_msgs = w_msgs, w_val = w_val,
    );
    println!("|{}|", "-".repeat(table_width - 2));

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
            "| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} |",
            model_name,
            format_number_compact(stats.count),
            format_number_compact(cache_hit),
            format_number_compact(prefill),
            format_number_compact(decoding),
            format_number_compact(cache_hit + prefill + decoding),
            w_model = w_model, w_msgs = w_msgs, w_val = w_val,
        );
    }

    println!("|{}|", "-".repeat(table_width - 2));
    println!(
        "| {:<w_model$} {:>w_msgs$} | {:>w_val$} {:>w_val$} {:>w_val$} {:>w_val$} |",
        "TOTAL",
        format_number_compact(sum_messages),
        format_number_compact(sum_cache_hit),
        format_number_compact(sum_prefill),
        format_number_compact(sum_decoding),
        format_number_compact(sum_total_with_cache),
        w_model = w_model, w_msgs = w_msgs, w_val = w_val,
    );

    println!(
        "| {:<w_model$} {:>w_msgs$} | ${:>w_cost$.2} ${:>w_cost$.2} ${:>w_cost$.2} ${:>w_cost$.2} |",
        "Cost",
        "",
        cache_hit_cost,
        prefill_cost,
        decoding_cost,
        total_cost,
        w_model = w_model, w_msgs = w_msgs, w_cost = w_val - 1,
    );
    println!("{}", "=".repeat(table_width));
}

fn print_table_minimal(
    model_stats: &[ModelBreakdownRow],
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
) {
    let w_model = 12;
    let w_msgs = 7;
    let w_strategy = 10;
    let table_width = 2 + w_model + 1 + w_msgs + 1 + 2 + w_strategy * 4 + 3 + 2;

    println!("Usage Summary");
    println!("{}", "=".repeat(table_width));

    println!(
        "| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} |",
        "Model", "Msgs", "Cache Hit", "Prefill", "Decode", "All",
        w_model = w_model, w_msgs = w_msgs, w_strategy = w_strategy,
    );
    println!("|{}|", "-".repeat(table_width - 2));

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
            "| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} |",
            model_name,
            format_number_compact(stats.count),
            format_number_compact(cache_hit),
            format_number_compact(prefill),
            format_number_compact(decoding),
            format_number_compact(cache_hit + prefill + decoding),
            w_model = w_model, w_msgs = w_msgs, w_strategy = w_strategy,
        );
    }

    println!("|{}|", "-".repeat(table_width - 2));
    println!(
        "| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} |",
        "TOTAL",
        format_number_compact(sum_messages),
        format_number_compact(sum_cache_hit),
        format_number_compact(sum_prefill),
        format_number_compact(sum_decoding),
        format_number_compact(sum_total_with_cache),
        w_model = w_model, w_msgs = w_msgs, w_strategy = w_strategy,
    );

    println!(
        "| {:<w_model$} {:>w_msgs$} | {:>w_strategy$} {:>w_strategy$} {:>w_strategy$} ${:>w_cost$.2} |",
        "Cost", "", "", "", "",
        total_cost,
        w_model = w_model, w_msgs = w_msgs, w_strategy = w_strategy, w_cost = w_strategy - 1,
    );
    println!("{}", "=".repeat(table_width));
}
