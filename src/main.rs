mod charts;
mod constants;
mod data;
mod formatting;
mod stats;
mod time_utils;

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
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

fn calculate_vendor_aggregate_time_series(
    state: &AppState,
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

    // Claude
    let claude_data = read_all_jsonl_files_dedup(Some(state.days));
    if !claude_data.is_empty() {
        let filtered = data::filter_usage_data_by_days(&claude_data, state.days);
        process_data(&filtered, "Claude", &mut time_series);
    }

    // Codex
    let codex_dir = get_codex_dir().join("sessions");
    if codex_dir.exists() {
        let codex_data = data::codex::read_codex_jsonl_files(&codex_dir, Some(state.days));
        let filtered = data::filter_usage_data_by_days(&codex_data, state.days);
        process_data(&filtered, "Codex", &mut time_series);
    }

    // Gemini
    let gemini_dir = get_gemini_dir().join("tmp");
    if gemini_dir.exists() {
        let gemini_data = data::gemini::read_gemini_json_files(&gemini_dir, Some(state.days));
        let filtered = data::filter_usage_data_by_days(&gemini_data, state.days);
        process_data(&filtered, "Gemini", &mut time_series);
    }

    time_series
}

fn calculate_all_model_breakdown(state: &AppState) -> Vec<ModelBreakdownRow> {
    let mut all_stats: Vec<ModelBreakdownRow> = Vec::new();

    // Claude
    let claude_data = read_all_jsonl_files_dedup(Some(state.days));
    if !claude_data.is_empty() {
        let filtered = data::filter_usage_data_by_days(&claude_data, state.days);
        all_stats.extend(stats::calculate_claude_model_breakdown(&filtered));
    }

    // Codex
    let codex_dir = get_codex_dir().join("sessions");
    if codex_dir.exists() {
        let codex_data = data::codex::read_codex_jsonl_files(&codex_dir, Some(state.days));
        if !codex_data.is_empty() {
            let filtered = data::filter_usage_data_by_days(&codex_data, state.days);
            all_stats.extend(stats::calculate_codex_model_breakdown(&filtered));
        }
    }

    // Gemini
    let gemini_dir = get_gemini_dir().join("tmp");
    if gemini_dir.exists() {
        let gemini_data = data::gemini::read_gemini_json_files(&gemini_dir, Some(state.days));
        if !gemini_data.is_empty() {
            let filtered = data::filter_usage_data_by_days(&gemini_data, state.days);
            all_stats.extend(stats::calculate_gemini_model_breakdown(&filtered));
        }
    }

    all_stats.sort_by(|a, b| b.total_with_cache.cmp(&a.total_with_cache));
    all_stats
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
    Command::new("curl")
        .args(["-sf", "--max-time", "10", &url])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let body = String::from_utf8_lossy(&output.stdout);
                serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|v| v.get("version").and_then(|v| v.as_str()).map(String::from))
            } else {
                None
            }
        })
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

fn print_stats_single(state: &AppState, once: bool) -> Option<bool> {
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

fn print_stats_all(state: &AppState, once: bool) -> Option<bool> {
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

    let vendor_time_series = calculate_vendor_aggregate_time_series(state, interval_minutes);
    let all_model_stats = calculate_all_model_breakdown(state);

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
        println!();
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

fn print_stats(state: &AppState, once: bool) -> Option<bool> {
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

    let mut state = AppState {
        vendor: args.vendor.clone(),
        days: args.days,
        monitor_interval: 3600,
        pricing,
        subscription_fees,
        version_cache: HashMap::new(),
    };

    if args.once {
        let result = print_stats(&state, true);
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

        let result = print_stats(&state, false);
        let mut terminal_too_small = result.is_none();

        let show_prompt = |state: &mut AppState, too_small: bool| {
            if too_small {
                return;
            }
            let (width, _) = get_terminal_size();
            let version = if state.vendor == "all" {
                "All Vendors Comparison".to_string()
            } else {
                get_version(state, &state.vendor.clone())
            };
            println!("\n{}", version);
            println!("{}", "-".repeat(width as usize));
            print!("> ");
            io::stdout().flush().unwrap();
        };

        show_prompt(&mut state, terminal_too_small);

        let stdin = io::stdin();
        loop {
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => {
                    let (width, _) = get_terminal_size();
                    println!("\n{}", "-".repeat(width as usize));
                    println!("\nExiting monitor mode...");
                    break;
                }
                Ok(_) => {
                    let command = line.trim();
                    let (width, _) = get_terminal_size();

                    match command {
                        "r" | "refresh" => {
                            println!("{}", "-".repeat(width as usize));
                            println!("\n{}", "=".repeat(width as usize));
                            println!("MANUAL REFRESH");
                            println!("{}\n", "=".repeat(width as usize));
                            let result = print_stats(&state, false);
                            terminal_too_small = result.is_none();
                        }
                        "n" => {
                            let rotation = ["all", "claude", "codex", "gemini"];
                            let idx = rotation.iter().position(|&v| v == state.vendor).unwrap_or(0);
                            state.vendor = rotation[(idx + 1) % rotation.len()].to_string();
                            println!("{}", "-".repeat(width as usize));
                            println!("\n{}", "=".repeat(width as usize));
                            println!("SWITCHED TO {}", state.vendor.to_uppercase());
                            println!("{}\n", "=".repeat(width as usize));
                            let result = print_stats(&state, false);
                            terminal_too_small = result.is_none();
                        }
                        "a" => {
                            if state.vendor != "all" {
                                state.vendor = "all".to_string();
                                println!("{}", "-".repeat(width as usize));
                                println!("\n{}", "=".repeat(width as usize));
                                println!("SWITCHED TO ALL VENDORS");
                                println!("{}\n", "=".repeat(width as usize));
                                let result = print_stats(&state, false);
                                terminal_too_small = result.is_none();
                            } else {
                                println!("Already monitoring all vendors.");
                            }
                        }
                        "d" | "day" | "days" => {
                            if state.days != 1 {
                                state.days = 1;
                                println!("{}", "-".repeat(width as usize));
                                println!("\n{}", "=".repeat(width as usize));
                                println!("CHANGED TO 1 DAY");
                                println!("{}\n", "=".repeat(width as usize));
                                let result = print_stats(&state, false);
                                terminal_too_small = result.is_none();
                            } else {
                                println!("Already showing 1 day.");
                            }
                        }
                        "w" | "week" => {
                            if state.days != 7 {
                                state.days = 7;
                                println!("{}", "-".repeat(width as usize));
                                println!("\n{}", "=".repeat(width as usize));
                                println!("CHANGED TO 7 DAYS (WEEK MODE)");
                                println!("{}\n", "=".repeat(width as usize));
                                let result = print_stats(&state, false);
                                terminal_too_small = result.is_none();
                            } else {
                                println!("Already showing 7 days (week mode).");
                            }
                        }
                        "m" | "month" => {
                            if state.days != 30 {
                                state.days = 30;
                                println!("{}", "-".repeat(width as usize));
                                println!("\n{}", "=".repeat(width as usize));
                                println!("CHANGED TO 30 DAYS (MONTH MODE)");
                                println!("{}\n", "=".repeat(width as usize));
                                let result = print_stats(&state, false);
                                terminal_too_small = result.is_none();
                            } else {
                                println!("Already showing 30 days (month mode).");
                            }
                        }
                        "h" | "help" => {
                            println!("{}", "-".repeat(width as usize));
                            println!("Available Commands:");
                            println!("  r, refresh       - Refresh statistics immediately");
                            println!("  v, vendor [X]    - Switch vendor (claude|codex|gemini|all)");
                            println!("  n                - Rotate to next vendor");
                            println!("  a                - Jump to vendor=all");
                            println!("  d, day, days [N] - Change days (default: 1 if no N)");
                            println!("  w, week          - Week mode (7 days)");
                            println!("  m, month         - Month mode (30 days)");
                            println!("  i, interval <N>  - Change refresh interval (seconds)");
                            println!("  h, help          - Show this help");
                            println!("  e, exit          - Exit monitor mode");
                            println!("  Ctrl+C, Ctrl+D   - Exit monitor mode");
                            println!("{}", "-".repeat(width as usize));
                            println!("Current: vendor={}, days={}, interval={}s",
                                     state.vendor, state.days, state.monitor_interval);
                        }
                        "e" | "exit" => {
                            println!("{}", "-".repeat(width as usize));
                            println!("\nExiting monitor mode...");
                            break;
                        }
                        "" => {}
                        _ => {
                            let parts: Vec<&str> = command.splitn(2, ' ').collect();
                            match parts[0] {
                                "v" | "vendor" if parts.len() == 2 => {
                                    let nv = parts[1];
                                    if ["claude", "codex", "gemini", "all"].contains(&nv) {
                                        state.vendor = nv.to_string();
                                        println!("{}", "-".repeat(width as usize));
                                        println!("\n{}", "=".repeat(width as usize));
                                        println!("SWITCHED TO {}", nv.to_uppercase());
                                        println!("{}\n", "=".repeat(width as usize));
                                        let result = print_stats(&state, false);
                                        terminal_too_small = result.is_none();
                                    } else {
                                        println!("Usage: v, vendor [claude|codex|gemini|all]");
                                    }
                                }
                                "d" | "day" | "days" if parts.len() == 2 => {
                                    if let Ok(n) = parts[1].parse::<i64>() {
                                        if n >= 1 {
                                            state.days = n;
                                            println!("{}", "-".repeat(width as usize));
                                            println!("\n{}", "=".repeat(width as usize));
                                            println!("CHANGED TO {} DAYS", n);
                                            println!("{}\n", "=".repeat(width as usize));
                                            let result = print_stats(&state, false);
                                            terminal_too_small = result.is_none();
                                        } else {
                                            println!("Days must be at least 1.");
                                        }
                                    } else {
                                        println!("Invalid days value.");
                                    }
                                }
                                "i" | "interval" if parts.len() == 2 => {
                                    if let Ok(n) = parts[1].parse::<u64>() {
                                        if n >= 1 {
                                            state.monitor_interval = n;
                                            println!("Refresh interval changed to {} seconds.", n);
                                        } else {
                                            println!("Interval must be at least 1 second.");
                                        }
                                    } else {
                                        println!("Invalid interval value.");
                                    }
                                }
                                "v" | "vendor" => {
                                    println!("Current vendor: {}", state.vendor);
                                    println!("Usage: v, vendor [claude|codex|gemini|all]");
                                }
                                "i" | "interval" => {
                                    println!("Current interval: {} seconds", state.monitor_interval);
                                    println!("Usage: i <N> or interval <N>");
                                }
                                _ => {
                                    println!("Unknown command: '{}'. Type h for help.", command);
                                }
                            }
                        }
                    }
                    show_prompt(&mut state, terminal_too_small);
                }
                Err(_) => break,
            }
        }
    }
}
