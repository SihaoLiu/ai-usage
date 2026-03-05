mod charts;
mod constants;
mod data;
mod formatting;
mod stats;
mod time_utils;

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use chrono::{Duration, Local, Timelike};
use clap::Parser;
use crossterm::terminal;

use constants::{AllPricing, SubscriptionFees, load_subscription_fees, prompt_subscription_fees};
use data::UsageEntry;
use data::claude::read_all_jsonl_files_dedup;
use data::codex::get_codex_dir;
use data::gemini::get_gemini_dir;
use formatting::print_model_breakdown;
use stats::{ModelBreakdownRow, VendorTimeSeries};

const MIN_TERMINAL_WIDTH: u16 = 60;
const MIN_TERMINAL_HEIGHT: u16 = 35;

struct VersionCacheEntry {
    version_str: String,
    timestamp: Instant,
}

struct AppState {
    vendor: String,
    days: i64,
    monitor_interval: u64,
    pricing: AllPricing,
    subscription_fees: SubscriptionFees,
    version_cache: HashMap<String, VersionCacheEntry>,
    all_vendor_prompt: Option<String>,
}

#[derive(Parser, Debug)]
#[command(about = "Analyze AI coding assistant usage statistics")]
struct Args {
    /// Number of days to look back
    #[arg(long, default_value = "3")]
    days: i64,

    /// Run once and exit (default: monitor mode with 1 hour refresh)
    #[arg(long)]
    once: bool,

    /// Vendor to collect statistics from
    #[arg(long, default_value = "all", value_parser = ["claude", "codex", "gemini", "all"])]
    vendor: String,
}

fn get_terminal_size() -> (u16, u16) {
    match terminal::size() {
        Ok(size) => size,
        Err(_) => {
            let cols = std::env::var("COLUMNS").ok()
                .and_then(|v| v.parse::<u16>().ok()).unwrap_or(80);
            let rows = std::env::var("LINES").ok()
                .and_then(|v| v.parse::<u16>().ok()).unwrap_or(24);
            (cols, rows)
        }
    }
}

fn check_terminal_size() -> (bool, u16, u16) {
    let (width, height) = get_terminal_size();
    (width >= MIN_TERMINAL_WIDTH && height >= MIN_TERMINAL_HEIGHT, width, height)
}

fn print_terminal_too_small(width: u16, height: u16) {
    print!("\x1b[2J\x1b[H");

    let lines = [
        "Terminal size too small:".to_string(),
        format!("  Width = {}  Height = {}", width, height),
        String::new(),
        "Needed for current config:".to_string(),
        format!("  Width = {}  Height = {}", MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT),
    ];

    let max_line_len = lines.iter().map(|l| l.len()).max().unwrap_or(0);
    let total_lines = lines.len();
    let top_padding = ((height as usize).saturating_sub(total_lines)) / 2;

    for _ in 0..top_padding {
        println!();
    }
    for line in &lines {
        let left_padding = ((width as usize).saturating_sub(max_line_len)) / 2;
        println!("{}{}", " ".repeat(left_padding), line);
    }
}

fn calculate_chart_height(is_monitor_mode: bool, table_printed: bool) -> usize {
    let (_, height) = get_terminal_size();
    let header_lines = 5usize;
    let breakdown_table_lines = if table_printed { 15 } else { 0 };
    let chart1_overhead = 6;
    let chart2_overhead = 13;
    let final_lines = 1;
    let monitor_prompt_lines = if is_monitor_mode { 4 } else { 0 };

    let fixed_overhead = header_lines + breakdown_table_lines + chart1_overhead + chart2_overhead
        + final_lines + monitor_prompt_lines;

    let available_height = (height as usize).saturating_sub(fixed_overhead);
    let chart_height = available_height / 2;

    chart_height.max(10).min(60)
}

fn calculate_optimal_interval_minutes(days_back: i64, target_width: usize) -> f64 {
    let total_minutes = days_back as f64 * 24.0 * 60.0;
    let min_interval = total_minutes / 100.0;
    let y_axis_width = 7.0;
    let chart_width = (target_width as f64 - y_axis_width - days_back as f64).max(50.0);
    let terminal_interval = total_minutes / chart_width;
    min_interval.max(terminal_interval)
}

fn round_to_nice_interval(optimal: f64) -> i64 {
    let nice = [1i64, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440];
    for &n in &nice {
        if n as f64 >= optimal {
            return n;
        }
    }
    *nice.last().unwrap()
}

/// Get the data directory for a vendor, or None for "all".
fn get_vendor_data_dir(vendor: &str) -> Option<PathBuf> {
    match vendor {
        "codex" => Some(get_codex_dir().join("sessions")),
        "gemini" => Some(get_gemini_dir().join("tmp")),
        "claude" => {
            let dirs = data::claude::get_claude_dirs();
            Some(dirs.into_iter()
                .map(|d| d.join("projects"))
                .find(|p| p.exists())
                .unwrap_or_else(|| PathBuf::from("~/.claude/projects")))
        }
        _ => None, // "all" has no single directory
    }
}

fn read_vendor_data(vendor: &str, days: i64) -> Vec<UsageEntry> {
    let max_age = Some(days);
    match vendor {
        "claude" => read_all_jsonl_files_dedup(max_age),
        "codex" => {
            let dir = get_codex_dir().join("sessions");
            data::codex::read_codex_jsonl_files(&dir, max_age)
        }
        "gemini" => {
            let dir = get_gemini_dir().join("tmp");
            data::gemini::read_gemini_json_files(&dir, max_age)
        }
        _ => Vec::new(),
    }
}

/// Loaded and filtered data for all vendors.
struct AllVendorData {
    claude: Vec<UsageEntry>,
    codex: Vec<UsageEntry>,
    gemini: Vec<UsageEntry>,
}

fn load_all_vendor_data(state: &AppState) -> AllVendorData {
    let claude_raw = read_all_jsonl_files_dedup(Some(state.days));
    let claude = data::filter_usage_data_by_days(&claude_raw, state.days);

    let codex_dir = get_codex_dir().join("sessions");
    let codex = if codex_dir.exists() {
        let raw = data::codex::read_codex_jsonl_files(&codex_dir, Some(state.days));
        data::filter_usage_data_by_days(&raw, state.days)
    } else {
        Vec::new()
    };

    let gemini_dir = get_gemini_dir().join("tmp");
    let gemini = if gemini_dir.exists() {
        let raw = data::gemini::read_gemini_json_files(&gemini_dir, Some(state.days));
        data::filter_usage_data_by_days(&raw, state.days)
    } else {
        Vec::new()
    };

    AllVendorData { claude, codex, gemini }
}

fn calculate_vendor_aggregate_time_series(
    all_data: &AllVendorData,
    interval_minutes: i64,
) -> VendorTimeSeries {
    let mut time_series: VendorTimeSeries = HashMap::new();

    let process_data = |entries: &[UsageEntry], vendor_label: &str, ts: &mut VendorTimeSeries| {
        for entry in entries {
            if entry.timestamp.is_empty() {
                continue;
            }

            let total = match vendor_label {
                "Codex" => {
                    entry.usage.input_tokens + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens
                        + entry.usage.reasoning_output_tokens
                }
                _ => {
                    entry.usage.input_tokens + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens
                        + entry.usage.cache_creation_input_tokens
                }
            } as f64;

            let parsed = entry.parsed_timestamp
                .or_else(|| time_utils::parse_timestamp(&entry.timestamp));

            if let Some(dt) = parsed {
                let interval_time = time_utils::to_interval(&dt, interval_minutes);
                *ts.entry(interval_time)
                    .or_default()
                    .entry(vendor_label.to_string())
                    .or_insert(0.0) += total;
            }
        }
    };

    if !all_data.claude.is_empty() {
        process_data(&all_data.claude, "Claude", &mut time_series);
    }
    if !all_data.codex.is_empty() {
        process_data(&all_data.codex, "Codex", &mut time_series);
    }
    if !all_data.gemini.is_empty() {
        process_data(&all_data.gemini, "Gemini", &mut time_series);
    }

    time_series
}

fn calculate_all_model_breakdown(all_data: &AllVendorData) -> Vec<ModelBreakdownRow> {
    let mut all_stats: Vec<ModelBreakdownRow> = Vec::new();

    if !all_data.claude.is_empty() {
        all_stats.extend(stats::calculate_claude_model_breakdown(&all_data.claude));
    }
    if !all_data.codex.is_empty() {
        all_stats.extend(stats::calculate_codex_model_breakdown(&all_data.codex));
    }
    if !all_data.gemini.is_empty() {
        all_stats.extend(stats::calculate_gemini_model_breakdown(&all_data.gemini));
    }

    all_stats.sort_by(|a, b| b.count.cmp(&a.count));
    all_stats
}

/// Calculate weighted average cost per MTok and total monthly savings across all vendors.
/// Returns (weighted_cost_per_mtok, total_monthly_savings).
fn calculate_weighted_cost_per_mtok(
    all_data: &AllVendorData,
    days: i64,
    pricing: &AllPricing,
    subscription_fees: &SubscriptionFees,
) -> (f64, f64) {
    // Per-vendor: (total_tokens, api_cost)
    // Match Python: Gemini weighted cost uses sessions dir (not tmp)
    let gemini_sessions_exist = get_gemini_dir().join("sessions").exists();
    let gemini_entries: &[UsageEntry] = if gemini_sessions_exist {
        &all_data.gemini
    } else {
        &[]
    };
    let vendor_configs: &[(&str, &[UsageEntry], f64)] = &[
        ("claude", &all_data.claude, subscription_fees.claude),
        ("codex", &all_data.codex, subscription_fees.codex),
        ("gemini", gemini_entries, subscription_fees.gemini),
    ];

    let mut vendor_data: Vec<(i64, f64, f64)> = Vec::new(); // (tokens, api_cost, sub_price)

    for &(vendor, entries, sub_price) in vendor_configs {
        if entries.is_empty() {
            continue;
        }

        // Accumulate per-model stats
        let mut model_stats: HashMap<String, (i64, i64, i64, i64)> = HashMap::new();
        let mut total_tokens: i64 = 0;

        for entry in entries {
            let model = &entry.model;
            let e = model_stats.entry(model.clone()).or_insert((0, 0, 0, 0));
            e.0 += entry.usage.input_tokens;
            e.1 += entry.usage.output_tokens;
            e.2 += entry.usage.cache_read_input_tokens;
            match vendor {
                "codex" => {
                    e.3 += entry.usage.reasoning_output_tokens;
                    total_tokens += entry.usage.input_tokens + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens + entry.usage.reasoning_output_tokens;
                }
                "gemini" => {
                    e.3 += entry.usage.cache_creation_input_tokens; // thinking stored here
                    total_tokens += entry.usage.input_tokens + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens + entry.usage.cache_creation_input_tokens;
                }
                _ => {
                    e.3 += entry.usage.cache_creation_input_tokens;
                    total_tokens += entry.usage.input_tokens + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens + entry.usage.cache_creation_input_tokens;
                }
            }
        }

        if total_tokens == 0 {
            continue;
        }

        // Calculate API cost
        let mut api_cost = 0.0;
        for (model, (input, output, cache_read, extra)) in &model_stats {
            let base_model = if vendor == "codex" && model.contains(" (") && model.ends_with(')') {
                model.rsplit_once(" (").map(|(base, _)| base).unwrap_or(model)
            } else {
                model.as_str()
            };
            let p = pricing.get_pricing(vendor, base_model);
            api_cost += *input as f64 * p.input / 1_000_000.0;
            api_cost += *output as f64 * p.output / 1_000_000.0;
            api_cost += *cache_read as f64 * p.cache_input / 1_000_000.0;
            api_cost += match vendor {
                "codex" | "gemini" => *extra as f64 * p.output / 1_000_000.0,
                _ => *extra as f64 * p.cache_output / 1_000_000.0,
            };
        }

        vendor_data.push((total_tokens, api_cost, sub_price));
    }

    let grand_total: i64 = vendor_data.iter().map(|(t, _, _)| t).sum();
    if grand_total == 0 || days <= 0 {
        return (0.0, 0.0);
    }

    let mut weighted_cost = 0.0;
    let mut total_savings = 0.0;

    for (tokens, api_cost, sub_price) in &vendor_data {
        let percentage = *tokens as f64 / grand_total as f64;
        let monthly_tokens = (*tokens as f64 / days as f64) * 30.0;
        let cost_per_mtok = if monthly_tokens > 0.0 {
            sub_price / (monthly_tokens / 1_000_000.0)
        } else {
            0.0
        };
        let daily_api_cost = api_cost / days as f64;
        let monthly_api_cost = daily_api_cost * 30.0;
        let savings = monthly_api_cost - sub_price;

        weighted_cost += percentage * cost_per_mtok;
        total_savings += savings;
    }

    (weighted_cost, total_savings)
}

fn get_version(state: &mut AppState, vendor: &str) -> String {
    let cache_ttl = std::time::Duration::from_secs(300);

    if let Some(cached) = state.version_cache.get(vendor) {
        if cached.timestamp.elapsed() < cache_ttl {
            return cached.version_str.clone();
        }
    }

    let (cmd, npm_pkg, display_name) = match vendor {
        "claude" => ("claude", "@anthropic-ai/claude-code", "Claude Code"),
        "codex" => ("codex", "@openai/codex", "Codex"),
        "gemini" => ("gemini", "@google/gemini-cli", "Gemini CLI"),
        _ => return String::new(),
    };

    let current_version = Command::new(cmd)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
                s.split_whitespace().next()
                    .and_then(|v| {
                        if v.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                            Some(v.to_string())
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        let re_match: String = s.chars()
                            .skip_while(|c| !c.is_ascii_digit())
                            .take_while(|c| c.is_ascii_digit() || *c == '.')
                            .collect();
                        if re_match.is_empty() { None } else { Some(re_match) }
                    })
            } else {
                None
            }
        });

    let version_str = match current_version {
        Some(ref ver) => {
            let latest = fetch_npm_version(npm_pkg);
            match latest {
                Some(ref latest_ver) if latest_ver == ver => {
                    format!("{} ({}, up-to-date)", display_name, ver)
                }
                Some(ref latest_ver) => {
                    format!("{} ({}, a newer version {} available)", display_name, ver, latest_ver)
                }
                None => format!("{} ({})", display_name, ver),
            }
        }
        None => display_name.to_string(),
    };

    state.version_cache.insert(vendor.to_string(), VersionCacheEntry {
        version_str: version_str.clone(),
        timestamp: Instant::now(),
    });

    version_str
}

fn fetch_npm_version(package: &str) -> Option<String> {
    let url = format!("https://registry.npmjs.org/{}/latest", package);
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .build()
        .new_agent();
    let body: String = agent.get(&url)
        .call().ok()?
        .body_mut().read_to_string().ok()?;
    serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("version").and_then(|v| v.as_str()).map(String::from))
}

fn get_chart_target_width() -> usize {
    let (width, _) = get_terminal_size();
    (width as f64 * 0.99) as usize
}

fn print_time_span_info(days: i64, interval_minutes: i64, terminal_width: u16) {
    let now = Local::now();
    let start_time = now - Duration::days(days);
    let total_minutes = start_time.hour() as i64 * 60 + start_time.minute() as i64;
    let interval_start = (total_minutes / interval_minutes) * interval_minutes;
    let start_rounded = start_time
        .with_hour((interval_start / 60) as u32).unwrap()
        .with_minute((interval_start % 60) as u32).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap();

    let mut data_points = 0;
    let mut current = start_rounded;
    while current <= now {
        data_points += 1;
        current = current + Duration::minutes(interval_minutes);
    }

    let interval_str = if interval_minutes >= 60 {
        if interval_minutes % 60 == 0 {
            format!("{}h", interval_minutes / 60)
        } else {
            format!("{}h{}m", interval_minutes / 60, interval_minutes % 60)
        }
    } else {
        format!("{}m", interval_minutes)
    };

    let now_str = now.format("%Y-%m-%d %H:%M:%S").to_string();
    let start_full = now.format("%Y-%m-%d %H:%M").to_string();
    let end_full = start_rounded.format("%Y-%m-%d %H:%M").to_string();
    let start_short = now.format("%m/%d %H:%M").to_string();
    let end_short = start_rounded.format("%m/%d %H:%M").to_string();
    let now_short = now.format("%m/%d %H:%M:%S").to_string();

    let full_line = format!(
        "Last updated: {} | Time span: {} to {} | Interval: {} | Data points: {}",
        now_str, start_full, end_full, interval_str, data_points
    );
    let short_line = format!(
        "Updated: {} | Span: {} - {} | {} | {} dp",
        now_short, start_short, end_short, interval_str, data_points
    );

    if terminal_width as usize >= full_line.len() {
        println!("{}", full_line);
    } else {
        println!("{}", short_line);
    }
    println!();
}

fn print_stats_single(state: &mut AppState, once: bool) -> Option<bool> {
    let (size_ok, width, height) = check_terminal_size();
    if !size_ok {
        print_terminal_too_small(width, height);
        return None;
    }

    if !once {
        print!("\x1b[2J\x1b[H");
    }

    let vendor = &state.vendor;
    let vendor_name = match vendor.as_str() {
        "codex" => "Codex",
        "gemini" => "Gemini CLI",
        _ => "Claude Code",
    };

    println!("Calculating {} usage...", vendor_name);
    println!("Showing data from last {} days", state.days);
    if !once {
        println!("Monitor mode: Refreshing every {} seconds (Press Ctrl+C to exit)", state.monitor_interval);
    }

    let usage_data = read_vendor_data(vendor, state.days);
    if usage_data.is_empty() {
        println!("No usage data found.");
        return Some(false);
    }

    let filtered = data::filter_usage_data_by_days(&usage_data, state.days);
    if filtered.is_empty() {
        println!("No usage data found in the last {} days.", state.days);
        return Some(false);
    }

    let model_stats = match vendor.as_str() {
        "codex" => stats::calculate_codex_model_breakdown(&filtered),
        "gemini" => stats::calculate_gemini_model_breakdown(&filtered),
        _ => stats::calculate_claude_model_breakdown(&filtered),
    };

    let table_printed = print_model_breakdown(
        &model_stats,
        state.days,
        Some(width),
        Some(height),
        vendor,
        &state.subscription_fees,
        &state.pricing,
    );

    let target_width = get_chart_target_width();
    let chart_height = calculate_chart_height(!once, table_printed);

    let optimal = calculate_optimal_interval_minutes(state.days, target_width);
    let interval_minutes = round_to_nice_interval(optimal);

    let model_ts = match vendor.as_str() {
        "codex" => stats::calculate_codex_model_token_breakdown_time_series(&filtered, interval_minutes),
        "gemini" => stats::calculate_gemini_model_token_breakdown_time_series(&filtered, interval_minutes),
        _ => stats::calculate_claude_model_token_breakdown_time_series(&filtered, interval_minutes),
    };

    let included_models: HashSet<String> = model_stats.iter().map(|s| s.model.clone()).collect();

    if !model_ts.is_empty() {
        print_time_span_info(state.days, interval_minutes, width);
    }

    charts::print_multi_line_chart(
        &model_ts, chart_height, state.days, "io", false,
        Some(target_width), interval_minutes, vendor, Some(&included_models), true,
    );
    charts::print_multi_line_chart(
        &model_ts, chart_height, state.days, "cache", true,
        Some(target_width), interval_minutes, vendor, Some(&included_models), true,
    );

    Some(true)
}

fn print_stats_all(state: &mut AppState, once: bool) -> Option<bool> {
    let (size_ok, width, height) = check_terminal_size();
    if !size_ok {
        print_terminal_too_small(width, height);
        return None;
    }

    if !once {
        print!("\x1b[2J\x1b[H");
    }

    println!("Calculating usage across all vendors...");
    println!("Showing data from last {} days", state.days);
    if !once {
        println!("Monitor mode: Refreshing every {} seconds (Press Ctrl+C to exit)", state.monitor_interval);
    }

    let target_width = get_chart_target_width();
    let optimal = calculate_optimal_interval_minutes(state.days, target_width);
    let interval_minutes = round_to_nice_interval(optimal);

    let all_data = load_all_vendor_data(state);

    // Compute and cache the weighted cost prompt for show_prompt reuse
    let (weighted_cost, total_savings) = calculate_weighted_cost_per_mtok(
        &all_data, state.days, &state.pricing, &state.subscription_fees,
    );
    state.all_vendor_prompt = if weighted_cost > 0.0 {
        Some(format!(
            "All Vendors Comparison, {} / MTok, Monthly Saving ${:.2}",
            formatting::format_cost_per_mtok(weighted_cost),
            total_savings,
        ))
    } else {
        None
    };

    let vendor_time_series = calculate_vendor_aggregate_time_series(&all_data, interval_minutes);
    let all_model_stats = calculate_all_model_breakdown(&all_data);

    let mut table_printed = false;
    if !all_model_stats.is_empty() {
        table_printed = print_model_breakdown(
            &all_model_stats,
            state.days,
            Some(width),
            Some(height),
            "all",
            &state.subscription_fees,
            &state.pricing,
        );
    }

    if vendor_time_series.is_empty() {
        println!("No usage data found from any vendor.");
        return Some(false);
    }

    let chart_height = calculate_chart_height(!once, table_printed) * 2;

    print_time_span_info(state.days, interval_minutes, width);

    charts::print_vendor_comparison_chart(
        &vendor_time_series,
        chart_height,
        state.days,
        Some(target_width),
        interval_minutes,
        true,
    );

    Some(true)
}

fn print_stats(state: &mut AppState, once: bool) -> Option<bool> {
    if state.vendor == "all" {
        print_stats_all(state, once)
    } else {
        print_stats_single(state, once)
    }
}

fn main() {
    let args = Args::parse();

    let pricing = AllPricing::load();
    let subscription_fees = load_subscription_fees()
        .unwrap_or_else(|| prompt_subscription_fees());

    // Validate vendor data directory on startup (matches Python behavior)
    if let Some(data_dir) = get_vendor_data_dir(&args.vendor) {
        if !data_dir.exists() {
            eprintln!("Error: Data directory not found at {}", data_dir.display());
            std::process::exit(1);
        }
    }

    let mut state = AppState {
        vendor: args.vendor.clone(),
        days: args.days,
        monitor_interval: 3600,
        pricing,
        subscription_fees,
        version_cache: HashMap::new(),
        all_vendor_prompt: None,
    };

    if args.once {
        let result = print_stats(&mut state, true);
        match result {
            None => std::process::exit(1),
            Some(false) => std::process::exit(0),
            Some(true) => {}
        }
    } else {
        // Monitor mode
        let (width, _) = get_terminal_size();
        println!("\n{}", "=".repeat(width as usize));
        println!("Interactive Monitor Mode (type h for help)");
        println!("{}", "=".repeat(width as usize));
        println!("Auto-refresh: {}s | Vendor: {} | Days: {}",
                 state.monitor_interval, state.vendor, state.days);
        println!("{}\n", "=".repeat(width as usize));

        // Helper: disable raw mode, run print_stats, re-enable raw mode.
        // This ensures println! in formatting/charts code outputs \r\n properly.
        let refresh_display = |state: &mut AppState| -> Option<bool> {
            crossterm::terminal::disable_raw_mode().ok();
            let result = print_stats(state, false);
            crossterm::terminal::enable_raw_mode().ok();
            result
        };

        // Enable raw mode for non-blocking input
        crossterm::terminal::enable_raw_mode().ok();

        let result = refresh_display(&mut state);
        let mut terminal_too_small = result.is_none();
        let mut last_size = get_terminal_size();

        let show_prompt = |state: &mut AppState, too_small: bool| {
            if too_small {
                return;
            }
            let (width, _) = get_terminal_size();
            let version = if state.vendor == "all" {
                state.all_vendor_prompt.clone()
                    .unwrap_or_else(|| "All Vendors Comparison".to_string())
            } else {
                get_version(state, &state.vendor.clone())
            };
            println!("\n\r{}\r", version);
            println!("\r{}\r", "-".repeat(width as usize));
            print!("\r> ");
            io::stdout().flush().unwrap();
        };

        show_prompt(&mut state, terminal_too_small);

        let mut next_refresh = std::time::Instant::now()
            + std::time::Duration::from_secs(state.monitor_interval);
        let mut input_buf = String::new();

        let cleanup_and_break = |msg: &str| {
            crossterm::terminal::disable_raw_mode().ok();
            let (width, _) = get_terminal_size();
            println!("\n\r{}\r", "-".repeat(width as usize));
            println!("\r{}", msg);
        };

        'monitor: loop {
            // Check terminal resize
            let current_size = get_terminal_size();
            if current_size != last_size {
                last_size = current_size;
                if !terminal_too_small {
                    let (width, _) = current_size;
                    println!("\r{}\r", " ".repeat(width as usize + 2));
                    println!("{}\r", "-".repeat(width as usize));
                    println!("\n\r{}\r", "=".repeat(width as usize));
                    println!("TERMINAL RESIZED (width: {}, height: {})\r", current_size.0, current_size.1);
                    println!("{}\n\r", "=".repeat(width as usize));
                }
                let result = refresh_display(&mut state);
                terminal_too_small = result.is_none();
                next_refresh = std::time::Instant::now()
                    + std::time::Duration::from_secs(state.monitor_interval);
                show_prompt(&mut state, terminal_too_small);
            }

            // Check auto-refresh
            if std::time::Instant::now() >= next_refresh {
                if !terminal_too_small {
                    let (width, _) = get_terminal_size();
                    println!("\r{}\r", " ".repeat(width as usize + 2));
                    println!("{}\r", "-".repeat(width as usize));
                    println!("\n\r{}\r", "=".repeat(width as usize));
                    println!("AUTO-REFRESH\r");
                    println!("{}\n\r", "=".repeat(width as usize));
                }
                let result = refresh_display(&mut state);
                terminal_too_small = result.is_none();
                next_refresh = std::time::Instant::now()
                    + std::time::Duration::from_secs(state.monitor_interval);
                show_prompt(&mut state, terminal_too_small);
            }

            // Poll for input with 1s timeout
            let timeout = std::time::Duration::from_secs(1)
                .min(next_refresh.saturating_duration_since(std::time::Instant::now()));
            if crossterm::event::poll(timeout).unwrap_or(false) {
                use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
                if let Ok(event) = crossterm::event::read() {
                    match event {
                        Event::Key(KeyEvent { code: KeyCode::Char('c'), modifiers, .. })
                            if modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            cleanup_and_break("Monitoring stopped.");
                            break 'monitor;
                        }
                        Event::Key(KeyEvent { code: KeyCode::Char('d'), modifiers, .. })
                            if modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            cleanup_and_break("Exiting monitor mode...");
                            break 'monitor;
                        }
                        Event::Key(KeyEvent { code: KeyCode::Enter, .. }) => {
                            println!("\r");
                            let command = input_buf.trim().to_string();
                            input_buf.clear();
                            let (width, _) = get_terminal_size();

                            let mut did_refresh = false;
                            match command.as_str() {
                                "r" | "refresh" => {
                                    println!("{}\r", "-".repeat(width as usize));
                                    println!("\n\r{}\r", "=".repeat(width as usize));
                                    println!("MANUAL REFRESH\r");
                                    println!("{}\n\r", "=".repeat(width as usize));
                                    let result = refresh_display(&mut state);
                                    terminal_too_small = result.is_none();
                                    did_refresh = true;
                                }
                                "n" => {
                                    let rotation = ["all", "claude", "codex", "gemini"];
                                    let idx = rotation.iter().position(|&v| v == state.vendor).unwrap_or(0);
                                    let mut new_vendor = rotation[(idx + 1) % rotation.len()];
                                    // Validate directory; skip missing vendors
                                    for _ in 0..rotation.len() {
                                        if let Some(dir) = get_vendor_data_dir(new_vendor) {
                                            if !dir.exists() {
                                                println!("Skipping {} (no data dir)...\r", new_vendor);
                                                let skip_idx = rotation.iter().position(|&v| v == new_vendor).unwrap_or(0);
                                                new_vendor = rotation[(skip_idx + 1) % rotation.len()];
                                                continue;
                                            }
                                        }
                                        break;
                                    }
                                    state.vendor = new_vendor.to_string();
                                    println!("{}\r", "-".repeat(width as usize));
                                    println!("\n\r{}\r", "=".repeat(width as usize));
                                    println!("SWITCHED TO {}\r", state.vendor.to_uppercase());
                                    println!("{}\n\r", "=".repeat(width as usize));
                                    let result = refresh_display(&mut state);
                                    terminal_too_small = result.is_none();
                                    did_refresh = true;
                                }
                                "a" => {
                                    if state.vendor != "all" {
                                        state.vendor = "all".to_string();
                                        println!("{}\r", "-".repeat(width as usize));
                                        println!("\n\r{}\r", "=".repeat(width as usize));
                                        println!("SWITCHED TO ALL VENDORS\r");
                                        println!("{}\n\r", "=".repeat(width as usize));
                                        let result = refresh_display(&mut state);
                                        terminal_too_small = result.is_none();
                                        did_refresh = true;
                                    } else {
                                        println!("Already monitoring all vendors.\r");
                                    }
                                }
                                "d" | "day" | "days" => {
                                    if state.days != 1 {
                                        state.days = 1;
                                        println!("{}\r", "-".repeat(width as usize));
                                        println!("\n\r{}\r", "=".repeat(width as usize));
                                        println!("CHANGED TO 1 DAY\r");
                                        println!("{}\n\r", "=".repeat(width as usize));
                                        let result = refresh_display(&mut state);
                                        terminal_too_small = result.is_none();
                                        did_refresh = true;
                                    } else {
                                        println!("Already showing 1 day.\r");
                                    }
                                }
                                "w" | "week" => {
                                    if state.days != 7 {
                                        state.days = 7;
                                        println!("{}\r", "-".repeat(width as usize));
                                        println!("\n\r{}\r", "=".repeat(width as usize));
                                        println!("CHANGED TO 7 DAYS (WEEK MODE)\r");
                                        println!("{}\n\r", "=".repeat(width as usize));
                                        let result = refresh_display(&mut state);
                                        terminal_too_small = result.is_none();
                                        did_refresh = true;
                                    } else {
                                        println!("Already showing 7 days (week mode).\r");
                                    }
                                }
                                "m" | "month" => {
                                    if state.days != 30 {
                                        state.days = 30;
                                        println!("{}\r", "-".repeat(width as usize));
                                        println!("\n\r{}\r", "=".repeat(width as usize));
                                        println!("CHANGED TO 30 DAYS (MONTH MODE)\r");
                                        println!("{}\n\r", "=".repeat(width as usize));
                                        let result = refresh_display(&mut state);
                                        terminal_too_small = result.is_none();
                                        did_refresh = true;
                                    } else {
                                        println!("Already showing 30 days (month mode).\r");
                                    }
                                }
                                "h" | "help" => {
                                    println!("{}\r", "-".repeat(width as usize));
                                    println!("Available Commands:\r");
                                    println!("  r, refresh       - Refresh statistics immediately\r");
                                    println!("  v, vendor [X]    - Switch vendor (claude|codex|gemini|all)\r");
                                    println!("  n                - Rotate to next vendor\r");
                                    println!("  a                - Jump to vendor=all\r");
                                    println!("  d, day, days [N] - Change days (default: 1 if no N)\r");
                                    println!("  w, week          - Week mode (7 days)\r");
                                    println!("  m, month         - Month mode (30 days)\r");
                                    println!("  i, interval <N>  - Change refresh interval (seconds)\r");
                                    println!("  h, help          - Show this help\r");
                                    println!("  e, exit          - Exit monitor mode\r");
                                    println!("  Ctrl+C, Ctrl+D   - Exit monitor mode\r");
                                    println!("{}\r", "-".repeat(width as usize));
                                    println!("Current: vendor={}, days={}, interval={}s\r",
                                             state.vendor, state.days, state.monitor_interval);
                                }
                                "e" | "exit" => {
                                    cleanup_and_break("Exiting monitor mode...");
                                    break 'monitor;
                                }
                                "" => {}
                                _ => {
                                    let parts: Vec<&str> = command.splitn(2, ' ').collect();
                                    match parts[0] {
                                        "v" | "vendor" if parts.len() == 2 => {
                                            let nv = parts[1];
                                            if ["claude", "codex", "gemini", "all"].contains(&nv) {
                                                // Validate directory before switching
                                                if let Some(dir) = get_vendor_data_dir(nv) {
                                                    if !dir.exists() {
                                                        println!("Error: Data directory not found at {}\r", dir.display());
                                                        show_prompt(&mut state, terminal_too_small);
                                                        continue 'monitor;
                                                    }
                                                }
                                                state.vendor = nv.to_string();
                                                println!("{}\r", "-".repeat(width as usize));
                                                println!("\n\r{}\r", "=".repeat(width as usize));
                                                println!("SWITCHED TO {}\r", nv.to_uppercase());
                                                println!("{}\n\r", "=".repeat(width as usize));
                                                let result = refresh_display(&mut state);
                                                terminal_too_small = result.is_none();
                                                did_refresh = true;
                                            } else {
                                                println!("Usage: v, vendor [claude|codex|gemini|all]\r");
                                            }
                                        }
                                        "d" | "day" | "days" if parts.len() == 2 => {
                                            if let Ok(n) = parts[1].parse::<i64>() {
                                                if n >= 1 {
                                                    state.days = n;
                                                    println!("{}\r", "-".repeat(width as usize));
                                                    println!("\n\r{}\r", "=".repeat(width as usize));
                                                    println!("CHANGED TO {} DAYS\r", n);
                                                    println!("{}\n\r", "=".repeat(width as usize));
                                                    let result = refresh_display(&mut state);
                                                    terminal_too_small = result.is_none();
                                                    did_refresh = true;
                                                } else {
                                                    println!("Days must be at least 1.\r");
                                                }
                                            } else {
                                                println!("Invalid days value.\r");
                                            }
                                        }
                                        "i" | "interval" if parts.len() == 2 => {
                                            if let Ok(n) = parts[1].parse::<u64>() {
                                                if n >= 1 {
                                                    state.monitor_interval = n;
                                                    next_refresh = std::time::Instant::now()
                                                        + std::time::Duration::from_secs(n);
                                                    println!("Refresh interval changed to {} seconds.\r", n);
                                                } else {
                                                    println!("Interval must be at least 1 second.\r");
                                                }
                                            } else {
                                                println!("Invalid interval value.\r");
                                            }
                                        }
                                        "v" | "vendor" => {
                                            println!("Current vendor: {}\r", state.vendor);
                                            println!("Usage: v, vendor [claude|codex|gemini|all]\r");
                                        }
                                        "i" | "interval" => {
                                            println!("Current interval: {} seconds\r", state.monitor_interval);
                                            println!("Usage: i <N> or interval <N>\r");
                                        }
                                        _ => {
                                            println!("Unknown command: '{}'. Type h for help.\r", command);
                                        }
                                    }
                                }
                            }
                            if did_refresh {
                                next_refresh = std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval);
                            }
                            show_prompt(&mut state, terminal_too_small);
                        }
                        Event::Key(KeyEvent { code: KeyCode::Backspace, .. }) => {
                            if !input_buf.is_empty() {
                                input_buf.pop();
                                print!("\x08 \x08");
                                io::stdout().flush().unwrap();
                            }
                        }
                        Event::Key(KeyEvent { code: KeyCode::Char(c), modifiers, .. })
                            if !modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            input_buf.push(c);
                            print!("{}", c);
                            io::stdout().flush().unwrap();
                        }
                        Event::Resize(w, h) => {
                            last_size = (w, h);
                            if !terminal_too_small {
                                println!("\r{}\r", " ".repeat(w as usize + 2));
                                println!("{}\r", "-".repeat(w as usize));
                                println!("\n\r{}\r", "=".repeat(w as usize));
                                println!("TERMINAL RESIZED (width: {}, height: {})\r", w, h);
                                println!("{}\n\r", "=".repeat(w as usize));
                            }
                            let result = refresh_display(&mut state);
                            terminal_too_small = result.is_none();
                            next_refresh = std::time::Instant::now()
                                + std::time::Duration::from_secs(state.monitor_interval);
                            show_prompt(&mut state, terminal_too_small);
                        }
                        _ => {}
                    }
                }
            }
        }
        crossterm::terminal::disable_raw_mode().ok();
    }
}
