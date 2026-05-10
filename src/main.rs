mod charts;
mod constants;
mod data;
mod formatting;
mod pricing;
mod stats;
mod time_utils;

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use chrono::{DateTime, Local};
use clap::Parser;
use crossterm::terminal;

use constants::{AllPricing, ModelPricing, SubscriptionFees, load_subscription_fees, prompt_subscription_fees};
use data::UsageEntry;
use data::claude::read_all_jsonl_files_dedup;
use data::codex::get_codex_dir;
use data::gemini::get_gemini_dir;
use formatting::print_model_breakdown;
use stats::{ModelBreakdownRow, VendorTimeSeries};
use time_utils::TimeWindow;

const MIN_TERMINAL_WIDTH: u16 = 60;
const MIN_TERMINAL_HEIGHT: u16 = 35;

struct VersionCacheEntry {
    version_str: String,
    timestamp: Instant,
}

struct AppState {
    vendor: String,
    days: i64,
    time_window: TimeWindow,
    monitor_interval: u64,
    pricing: AllPricing,
    subscription_fees: SubscriptionFees,
    version_cache: HashMap<String, VersionCacheEntry>,
    all_vendor_prompt: Option<String>,
}

/// Shell-like in-memory command history for the monitor prompt.
///
/// `cursor == None` means the user is editing a fresh line. `Up` saves that
/// line into `draft` and moves to the most recent entry. `Down` walks back
/// toward the draft, ending with `cursor = None` and `input_buf = draft`.
struct CommandHistory {
    entries: Vec<String>,
    cursor: Option<usize>,
    draft: String,
}

impl CommandHistory {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            cursor: None,
            draft: String::new(),
        }
    }

    /// Append `command` to history (unless empty or identical to the last
    /// entry) and always reset navigation state to the fresh-line position.
    fn record(&mut self, command: &str) {
        self.cursor = None;
        self.draft.clear();
        if command.is_empty() {
            return;
        }
        if self.entries.last().map(|s| s.as_str()) == Some(command) {
            return;
        }
        self.entries.push(command.to_string());
    }

    /// Return the previous entry to display, or `None` if there is nothing
    /// older to walk to. When stepping off the fresh line, `current_buf` is
    /// saved as the draft so `navigate_down` can restore it later.
    fn navigate_up(&mut self, current_buf: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let new_cursor = match self.cursor {
            None => {
                self.draft = current_buf.to_string();
                self.entries.len() - 1
            }
            Some(0) => return None,
            Some(n) => n - 1,
        };
        self.cursor = Some(new_cursor);
        Some(self.entries[new_cursor].clone())
    }

    /// Return the next entry to display, or the saved draft when stepping
    /// back to the fresh-line position. `None` means the cursor is already
    /// at the fresh line and nothing changes.
    fn navigate_down(&mut self) -> Option<String> {
        match self.cursor {
            None => None,
            Some(n) if n + 1 < self.entries.len() => {
                self.cursor = Some(n + 1);
                Some(self.entries[n + 1].clone())
            }
            Some(_) => {
                self.cursor = None;
                Some(std::mem::take(&mut self.draft))
            }
        }
    }
}

/// Editable prompt buffer with a char-aligned cursor. Drives the shell-style
/// editing behavior (insert at cursor, backspace before cursor, left/right
/// arrows moving the cursor without changing the text).
struct InputLine {
    buf: String,
    cursor_chars: usize,
}

impl InputLine {
    fn new() -> Self {
        Self {
            buf: String::new(),
            cursor_chars: 0,
        }
    }

    fn snapshot(&self) -> &str {
        &self.buf
    }

    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn char_count(&self) -> usize {
        self.buf.chars().count()
    }

    fn cursor_chars(&self) -> usize {
        self.cursor_chars
    }

    fn insert_char(&mut self, c: char) {
        let byte_pos = byte_index_for_char(&self.buf, self.cursor_chars);
        self.buf.insert(byte_pos, c);
        self.cursor_chars += 1;
    }

    /// Delete the char immediately before the cursor. Returns whether the
    /// buffer changed.
    fn backspace(&mut self) -> bool {
        if self.cursor_chars == 0 {
            return false;
        }
        let prev = self.cursor_chars - 1;
        let byte_pos = byte_index_for_char(&self.buf, prev);
        self.buf.remove(byte_pos);
        self.cursor_chars = prev;
        true
    }

    fn move_left(&mut self) -> bool {
        if self.cursor_chars == 0 {
            return false;
        }
        self.cursor_chars -= 1;
        true
    }

    fn move_right(&mut self) -> bool {
        if self.cursor_chars >= self.char_count() {
            return false;
        }
        self.cursor_chars += 1;
        true
    }

    /// Replace the buffer (used by history recall) and park the cursor at
    /// the end so the user can keep typing immediately.
    fn replace(&mut self, s: String) {
        self.cursor_chars = s.chars().count();
        self.buf = s;
    }

    fn clear(&mut self) {
        self.buf.clear();
        self.cursor_chars = 0;
    }
}

fn byte_index_for_char(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
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

/// Calculate chart height(s) that fit within the terminal.
/// For single vendor: returns per-chart height (2 charts displayed).
/// For all vendor: returns the single chart height.
/// Also returns whether the layout fits (true) or overflows (false).
fn calculate_chart_height(
    is_monitor_mode: bool,
    table_printed: bool,
    num_models: usize,
    is_all_vendor: bool,
) -> (usize, bool) {
    let (_, height) = get_terminal_size();
    let th = height as usize;

    // Header: "Calculating...", "Showing data...", "Monitor mode..." (or 2 if --once)
    let header_lines = if is_monitor_mode { 3 } else { 2 };

    // Table: 1 blank + 1 title + 1 =border + 1 header + 1 -border
    //        + num_models rows + 1 -border + 1 TOTAL + 1 Cost + 1 =border
    //        + 1 cost summary = 10 + num_models
    let table_lines = if table_printed { 10 + num_models } else { 0 };

    // Time span info: 1 line + 1 blank
    let time_span_lines = 2;

    // Per-chart overhead: 1 blank + 1 title + 1 =border + 2 daily_header
    //                     + 1 x-axis bottom line + 1 legend = 7
    // X-axis labels: typically 2 lines (blank + label chars), up to 5
    let chart_overhead = 7;
    let x_axis_label_lines = 3; // blank + typically 2 label rows

    // Monitor prompt: 1 blank + 1 version + 1 separator + 1 "> " = 4
    let prompt_lines = if is_monitor_mode { 4 } else { 0 };

    let min_chart = 5usize;

    if is_all_vendor {
        // Single chart
        let fixed = header_lines + table_lines + time_span_lines
            + chart_overhead + x_axis_label_lines + prompt_lines;
        let available = th.saturating_sub(fixed);
        let chart_height = available.max(min_chart).min(60);
        let fits = th >= fixed + min_chart;
        (chart_height, fits)
    } else {
        // Two charts: chart1 (io, no x-axis labels) + chart2 (cache, with x-axis labels)
        let chart1_fixed = chart_overhead;
        let chart2_fixed = chart_overhead + x_axis_label_lines;
        let fixed = header_lines + table_lines + time_span_lines
            + chart1_fixed + chart2_fixed + prompt_lines;
        let available = th.saturating_sub(fixed);
        let per_chart = (available / 2).max(min_chart).min(60);
        let fits = th >= fixed + min_chart * 2;
        (per_chart, fits)
    }
}

fn calculate_optimal_interval_minutes(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    target_width: usize,
) -> f64 {
    let total_minutes = ((*range_end - *range_start).num_seconds() as f64 / 60.0).max(1.0);
    let min_interval = total_minutes / 100.0;
    let y_axis_width = 7.0;
    let span_days = (total_minutes / (24.0 * 60.0)).ceil().max(1.0);
    let chart_width = (target_width as f64 - y_axis_width - span_days).max(50.0);
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

fn read_vendor_data(vendor: &str, max_age: Option<i64>) -> Vec<UsageEntry> {
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

fn load_all_vendor_data(state: &AppState, now: DateTime<Local>) -> AllVendorData {
    let max_age = state.time_window.file_scan_days(now);

    let claude_raw = read_all_jsonl_files_dedup(max_age);
    let claude = data::filter_usage_data_by_window(&claude_raw, &state.time_window, now);

    let codex_dir = get_codex_dir().join("sessions");
    let codex = if codex_dir.exists() {
        let raw = data::codex::read_codex_jsonl_files(&codex_dir, max_age);
        data::filter_usage_data_by_window(&raw, &state.time_window, now)
    } else {
        Vec::new()
    };

    let gemini_dir = get_gemini_dir().join("tmp");
    let gemini = if gemini_dir.exists() {
        let raw = data::gemini::read_gemini_json_files(&gemini_dir, max_age);
        data::filter_usage_data_by_window(&raw, &state.time_window, now)
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

fn calculate_all_model_breakdown(
    all_data: &AllVendorData,
    pricing: &AllPricing,
) -> Vec<ModelBreakdownRow> {
    let mut all_stats: Vec<ModelBreakdownRow> = Vec::new();

    if !all_data.claude.is_empty() {
        all_stats.extend(stats::calculate_claude_model_breakdown(
            &all_data.claude,
            pricing,
        ));
    }
    if !all_data.codex.is_empty() {
        all_stats.extend(stats::calculate_codex_model_breakdown(
            &all_data.codex,
            pricing,
        ));
    }
    if !all_data.gemini.is_empty() {
        all_stats.extend(stats::calculate_gemini_model_breakdown(
            &all_data.gemini,
            pricing,
        ));
    }

    all_stats.sort_by(|a, b| b.count.cmp(&a.count));
    all_stats
}

/// Calculate weighted average cost per MTok and total monthly savings across all vendors.
/// Returns (weighted_cost_per_mtok, total_monthly_savings).
fn calculate_weighted_cost_per_mtok(
    all_data: &AllVendorData,
    days: f64,
    pricing: &AllPricing,
    subscription_fees: &SubscriptionFees,
) -> (f64, f64) {
    let vendor_configs: &[(&str, &[UsageEntry], f64)] = &[
        ("claude", &all_data.claude, subscription_fees.claude),
        ("codex", &all_data.codex, subscription_fees.codex),
        ("gemini", &all_data.gemini, subscription_fees.gemini),
    ];

    let mut vendor_data: Vec<(i64, f64, f64)> = Vec::new(); // (tokens, api_cost, sub_price)

    for &(vendor, entries, sub_price) in vendor_configs {
        if entries.is_empty() {
            continue;
        }

        let mut total_tokens: i64 = 0;
        let mut api_cost: f64 = 0.0;

        // Compute cost per-entry so tiered pricing (Claude 1M-context >200k
        // premium) is applied correctly. Aggregating tokens first and then
        // multiplying by the tier rate would overstate cost when many entries
        // are individually below 200k.
        for entry in entries {
            if entry.model.contains("<synthetic>") {
                continue;
            }

            let extra = match vendor {
                "codex" => entry.usage.reasoning_output_tokens,
                _ => entry.usage.cache_creation_input_tokens,
            };
            total_tokens += entry.usage.input_tokens
                + entry.usage.output_tokens
                + entry.usage.cache_read_input_tokens
                + extra;

            let p = pricing.get_pricing(vendor, &entry.model);
            api_cost +=
                ModelPricing::tier_cost(entry.usage.input_tokens, p.input, p.input_above_200k);
            api_cost +=
                ModelPricing::tier_cost(entry.usage.output_tokens, p.output, p.output_above_200k);
            api_cost += ModelPricing::tier_cost(
                entry.usage.cache_read_input_tokens,
                p.cache_input,
                p.cache_input_above_200k,
            );
            api_cost += match vendor {
                "codex" => ModelPricing::tier_cost(
                    entry.usage.reasoning_output_tokens,
                    p.output,
                    p.output_above_200k,
                ),
                "gemini" => ModelPricing::tier_cost(
                    entry.usage.cache_creation_input_tokens,
                    p.output,
                    p.output_above_200k,
                ),
                _ => ModelPricing::tier_cost(
                    entry.usage.cache_creation_input_tokens,
                    p.cache_output,
                    p.cache_output_above_200k,
                ),
            };
        }

        if total_tokens == 0 {
            continue;
        }

        vendor_data.push((total_tokens, api_cost, sub_price));
    }

    let grand_total: i64 = vendor_data.iter().map(|(t, _, _)| t).sum();
    if grand_total == 0 || days <= 0.0 {
        return (0.0, 0.0);
    }

    let mut weighted_cost = 0.0;
    let mut total_savings = 0.0;

    for (tokens, api_cost, sub_price) in &vendor_data {
        let percentage = *tokens as f64 / grand_total as f64;
        let monthly_tokens = (*tokens as f64 / days) * 30.0;
        let cost_per_mtok = if monthly_tokens > 0.0 {
            sub_price / (monthly_tokens / 1_000_000.0)
        } else {
            0.0
        };
        let daily_api_cost = api_cost / days;
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

fn print_time_span_info(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    interval_minutes: i64,
    terminal_width: u16,
    left_pad: &str,
) {
    let now = Local::now();
    let start_rounded = time_utils::round_to_interval_start(range_start, interval_minutes);

    let data_points = time_utils::generate_interval_times(
        &start_rounded, range_end, interval_minutes,
    ).len();

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
    let start_full = start_rounded.format("%Y-%m-%d %H:%M").to_string();
    let end_full = range_end.format("%Y-%m-%d %H:%M").to_string();
    let start_short = start_rounded.format("%m/%d %H:%M").to_string();
    let end_short = range_end.format("%m/%d %H:%M").to_string();
    let now_short = now.format("%m/%d %H:%M:%S").to_string();

    let full_line = format!(
        "Last updated: {} | Time span: {} to {} | Interval: {} | Data points: {}",
        now_str, start_full, end_full, interval_str, data_points
    );
    let short_line = format!(
        "Updated: {} | Span: {} - {} | {} | {} dp",
        now_short, start_short, end_short, interval_str, data_points
    );

    if terminal_width as usize >= full_line.len() + left_pad.len() {
        println!("{}{}", left_pad, full_line);
    } else {
        println!("{}{}", left_pad, short_line);
    }
    println!();
}

fn showing_data_line(window: &TimeWindow, now: DateTime<Local>) -> String {
    format!("Showing data from {}", window.display_label(now))
}

fn parse_time_window_command(command: &str, current_days: i64) -> Option<Result<TimeWindow, String>> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    match parts.as_slice() {
        ["date", date] => Some(TimeWindow::from_date(date)),
        ["date"] => Some(Err("Usage: date YYYY-MM-DD".to_string())),
        ["range", start, end] => Some(TimeWindow::from_range(start, end)),
        ["range"] | ["range", _] => Some(Err("Usage: range YYYY-MM-DD YYYY-MM-DD".to_string())),
        ["latest"] | ["last"] => Some(Ok(TimeWindow::rolling_days(current_days))),
        _ => None,
    }
}

fn print_stats_single(state: &mut AppState, once: bool) -> Option<bool> {
    let (size_ok, width, height) = check_terminal_size();
    if !size_ok {
        print_terminal_too_small(width, height);
        return None;
    }

    let now = Local::now();
    let (range_start, range_end) = state.time_window.bounds(now);
    let projection_days = state.time_window.projection_days(now);
    let max_age = state.time_window.file_scan_days(now);
    let vendor = &state.vendor;

    // Pre-load data to determine model count for height check
    let usage_data = read_vendor_data(vendor, max_age);
    if usage_data.is_empty() {
        if !once { print!("\x1b[2J\x1b[H"); }
        println!("No usage data found.");
        return Some(false);
    }
    let filtered = data::filter_usage_data_by_window(&usage_data, &state.time_window, now);
    if filtered.is_empty() {
        if !once { print!("\x1b[2J\x1b[H"); }
        println!("No usage data found in {}.", state.time_window.display_label(now));
        return Some(false);
    }
    let model_stats = match vendor.as_str() {
        "codex" => stats::calculate_codex_model_breakdown(&filtered, &state.pricing),
        "gemini" => stats::calculate_gemini_model_breakdown(&filtered, &state.pricing),
        _ => stats::calculate_claude_model_breakdown(&filtered, &state.pricing),
    };

    // Pre-check whether table will be displayed and total height fits
    let table_mode = formatting::get_table_display_mode(width, height, model_stats.len());
    let mut will_print_table = table_mode != "hidden";

    let target_width = get_chart_target_width();
    let (mut chart_height, mut fits) = calculate_chart_height(
        !once, will_print_table, model_stats.len(), false,
    );
    // If it doesn't fit with table, try without
    if !fits && will_print_table {
        will_print_table = false;
        let result = calculate_chart_height(!once, false, model_stats.len(), false);
        chart_height = result.0;
        fits = result.1;
    }
    if !fits {
        print_terminal_too_small(width, height);
        return None;
    }

    if !once {
        print!("\x1b[2J\x1b[H");
    }

    let vendor_name = match vendor.as_str() {
        "codex" => "Codex",
        "gemini" => "Gemini CLI",
        _ => "Claude Code",
    };

    println!("Calculating {} usage...", vendor_name);
    println!("{}", showing_data_line(&state.time_window, now));
    if !once {
        println!("Monitor mode: Refreshing every {} seconds (Press Ctrl+C to exit)", state.monitor_interval);
    }

    // Only print table if height allows it
    let effective_height = if will_print_table { height } else { 0 };
    print_model_breakdown(
        &model_stats,
        projection_days,
        Some(width),
        Some(effective_height),
        vendor,
        &state.subscription_fees,
    );

    let optimal = calculate_optimal_interval_minutes(&range_start, &range_end, target_width);
    let interval_minutes = round_to_nice_interval(optimal);

    let model_ts = match vendor.as_str() {
        "codex" => stats::calculate_codex_model_token_breakdown_time_series(&filtered, interval_minutes),
        "gemini" => stats::calculate_gemini_model_token_breakdown_time_series(&filtered, interval_minutes),
        _ => stats::calculate_claude_model_token_breakdown_time_series(&filtered, interval_minutes),
    };

    let included_models: HashSet<String> = model_stats.iter().map(|s| s.model.clone()).collect();

    let table_w = if will_print_table {
        formatting::get_table_width(&formatting::get_table_display_mode(width, height, model_stats.len()))
    } else { 0 };
    let table_pad = formatting::center_pad(width as usize, table_w);

    if !model_ts.is_empty() {
        print_time_span_info(&range_start, &range_end, interval_minutes, width, &table_pad);
    }

    charts::print_multi_line_chart(
        &model_ts, chart_height, &range_start, &range_end, "io", false,
        Some(target_width), interval_minutes, vendor, Some(&included_models), true,
        Some(width as usize),
    );
    charts::print_multi_line_chart(
        &model_ts, chart_height, &range_start, &range_end, "cache", true,
        Some(target_width), interval_minutes, vendor, Some(&included_models), true,
        Some(width as usize),
    );

    Some(true)
}

fn print_stats_all(state: &mut AppState, once: bool) -> Option<bool> {
    let (size_ok, width, height) = check_terminal_size();
    if !size_ok {
        print_terminal_too_small(width, height);
        return None;
    }

    let now = Local::now();
    let (range_start, range_end) = state.time_window.bounds(now);
    let projection_days = state.time_window.projection_days(now);
    let target_width = get_chart_target_width();
    let optimal = calculate_optimal_interval_minutes(&range_start, &range_end, target_width);
    let interval_minutes = round_to_nice_interval(optimal);

    let all_data = load_all_vendor_data(state, now);

    // Compute and cache the weighted cost prompt for show_prompt reuse
    let (weighted_cost, total_savings) = calculate_weighted_cost_per_mtok(
        &all_data, projection_days, &state.pricing, &state.subscription_fees,
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
    let all_model_stats = calculate_all_model_breakdown(&all_data, &state.pricing);

    // Pre-check whether table will be displayed
    let table_mode = formatting::get_table_display_mode(width, height, all_model_stats.len());
    let mut will_print_table = table_mode != "hidden" && !all_model_stats.is_empty();

    // Check total height fits before printing anything
    let (mut chart_height, mut fits) = calculate_chart_height(
        !once, will_print_table, all_model_stats.len(), true,
    );
    // If it doesn't fit with table, try without
    if !fits && will_print_table {
        will_print_table = false;
        let result = calculate_chart_height(!once, false, all_model_stats.len(), true);
        chart_height = result.0;
        fits = result.1;
    }
    if !fits {
        print_terminal_too_small(width, height);
        return None;
    }

    if !once {
        print!("\x1b[2J\x1b[H");
    }

    println!("Calculating usage across all vendors...");
    println!("{}", showing_data_line(&state.time_window, now));
    if !once {
        println!("Monitor mode: Refreshing every {} seconds (Press Ctrl+C to exit)", state.monitor_interval);
    }

    if !all_model_stats.is_empty() {
        let effective_height = if will_print_table { height } else { 0 };
        print_model_breakdown(
            &all_model_stats,
            projection_days,
            Some(width),
            Some(effective_height),
            "all",
            &state.subscription_fees,
        );
    }

    if vendor_time_series.is_empty() {
        println!("No usage data found from any vendor.");
        return Some(false);
    }

    let table_w = if will_print_table {
        formatting::get_table_width(&formatting::get_table_display_mode(width, height, all_model_stats.len()))
    } else { 0 };
    let table_pad = formatting::center_pad(width as usize, table_w);

    print_time_span_info(&range_start, &range_end, interval_minutes, width, &table_pad);

    charts::print_vendor_comparison_chart(
        &vendor_time_series,
        chart_height,
        &range_start,
        &range_end,
        Some(target_width),
        interval_minutes,
        true,
        Some(width as usize),
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

    let pricing = pricing::load_layered();
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
        time_window: TimeWindow::rolling_days(args.days),
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
        println!("Auto-refresh: {}s | Vendor: {} | Window: {}",
                 state.monitor_interval, state.vendor,
                 state.time_window.display_label(Local::now()));
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
            // Render dimmed watermark as placeholder, then move cursor back so the
            // first keystroke lands right after "> ". The watermark is wiped via
            // \x1b[K when the user starts typing (see Char handler below). Skip the
            // watermark if it would not fit on the prompt line and wrap awkwardly.
            let (mark, mark_visible) = formatting::prompt_watermark();
            if (width as usize) >= 2 + mark_visible {
                print!("{}\x1b[{}D", mark, mark_visible);
            }
            io::stdout().flush().unwrap();
        };

        show_prompt(&mut state, terminal_too_small);

        let mut next_refresh = std::time::Instant::now()
            + std::time::Duration::from_secs(state.monitor_interval);
        let mut input = InputLine::new();
        let mut history = CommandHistory::new();

        // Redraw the prompt line in place: clears it, reprints "> {buf}",
        // restores the dimmed watermark when empty, and finally moves the
        // terminal cursor to match `input`'s logical position so left/right
        // arrows feel like a real shell.
        let render_input = |input: &InputLine, too_small: bool| {
            if too_small {
                return;
            }
            let (width, _) = get_terminal_size();
            print!("\r\x1b[K> {}", input.snapshot());
            if input.is_empty() {
                let (mark, mark_visible) = formatting::prompt_watermark();
                if (width as usize) >= 2 + mark_visible {
                    print!("{}\x1b[{}D", mark, mark_visible);
                }
            } else {
                let trailing = input.char_count().saturating_sub(input.cursor_chars());
                if trailing > 0 {
                    print!("\x1b[{}D", trailing);
                }
            }
            io::stdout().flush().unwrap();
        };

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
                            let command = input.snapshot().trim().to_string();
                            history.record(&command);
                            input.clear();
                            let (width, _) = get_terminal_size();

                            let mut did_refresh = false;
                            match command.as_str() {
                                "" | "r" | "refresh" => {
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
                                        state.time_window = TimeWindow::rolling_days(1);
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
                                        state.time_window = TimeWindow::rolling_days(7);
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
                                        state.time_window = TimeWindow::rolling_days(30);
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
                                    println!("  date YYYY-MM-DD  - Show one complete local day\r");
                                    println!("  range A B        - Show inclusive local date span (any order)\r");
                                    println!("  latest           - Return to rolling days window\r");
                                    println!("  i, interval <N>  - Change refresh interval (seconds)\r");
                                    println!("  h, help          - Show this help\r");
                                    println!("{}\r", "-".repeat(width as usize));
                                    println!("Current: vendor={}, window={}, interval={}s\r",
                                             state.vendor,
                                             state.time_window.display_label(Local::now()),
                                             state.monitor_interval);
                                }
                                "e" | "exit" => {
                                    cleanup_and_break("Exiting monitor mode...");
                                    break 'monitor;
                                }
                                _ => {
                                    if let Some(parsed) = parse_time_window_command(&command, state.days) {
                                        match parsed {
                                            Ok(window) => {
                                                state.time_window = window;
                                                println!("{}\r", "-".repeat(width as usize));
                                                println!("\n\r{}\r", "=".repeat(width as usize));
                                                println!(
                                                    "SET TIME WINDOW: {}\r",
                                                    state.time_window.display_label(Local::now())
                                                );
                                                println!("{}\n\r", "=".repeat(width as usize));
                                                let result = refresh_display(&mut state);
                                                terminal_too_small = result.is_none();
                                                did_refresh = true;
                                            }
                                            Err(err) => {
                                                println!("{}\r", err);
                                            }
                                        }
                                    } else {
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
                                                    state.time_window = TimeWindow::rolling_days(n);
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
                            }
                            if did_refresh {
                                next_refresh = std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval);
                            }
                            show_prompt(&mut state, terminal_too_small);
                        }
                        Event::Key(KeyEvent { code: KeyCode::Up, .. }) => {
                            if let Some(recalled) = history.navigate_up(input.snapshot()) {
                                input.replace(recalled);
                                render_input(&input, terminal_too_small);
                            }
                        }
                        Event::Key(KeyEvent { code: KeyCode::Down, .. }) => {
                            if let Some(recalled) = history.navigate_down() {
                                input.replace(recalled);
                                render_input(&input, terminal_too_small);
                            }
                        }
                        Event::Key(KeyEvent { code: KeyCode::Left, .. }) => {
                            if input.move_left() {
                                print!("\x1b[D");
                                io::stdout().flush().unwrap();
                            }
                        }
                        Event::Key(KeyEvent { code: KeyCode::Right, .. }) => {
                            if input.move_right() {
                                print!("\x1b[C");
                                io::stdout().flush().unwrap();
                            }
                        }
                        Event::Key(KeyEvent { code: KeyCode::Backspace, .. }) => {
                            if input.backspace() {
                                render_input(&input, terminal_too_small);
                            }
                        }
                        Event::Key(KeyEvent { code: KeyCode::Char(c), modifiers, .. })
                            if !modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            input.insert_char(c);
                            render_input(&input, terminal_too_small);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_command_selects_single_inclusive_day() {
        let command = parse_time_window_command("date 2026-05-07", 3)
            .expect("recognized command")
            .expect("valid date");
        let TimeWindow::ExplicitRange { start, end, projection_days } = command else {
            panic!("date command should create an explicit window");
        };

        assert_eq!(start.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-05-07 00:00:00");
        assert_eq!(end.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-05-07 23:59:59");
        assert_eq!(projection_days, 1.0);
    }

    #[test]
    fn range_command_selects_inclusive_date_span() {
        let command = parse_time_window_command("range 2026-05-01 2026-05-07", 3)
            .expect("recognized command")
            .expect("valid range");

        assert_eq!(command.projection_days(Local::now()), 7.0);
    }

    #[test]
    fn history_navigate_up_walks_back_then_stops_at_oldest() {
        let mut history = CommandHistory::new();
        history.record("v claude");
        history.record("range 2026-05-01 2026-05-07");
        history.record("w");

        assert_eq!(history.navigate_up("draft"), Some("w".to_string()));
        assert_eq!(
            history.navigate_up("ignored"),
            Some("range 2026-05-01 2026-05-07".to_string())
        );
        assert_eq!(history.navigate_up("ignored"), Some("v claude".to_string()));
        assert_eq!(history.navigate_up("ignored"), None);
    }

    #[test]
    fn history_navigate_down_walks_forward_and_restores_draft() {
        let mut history = CommandHistory::new();
        history.record("w");
        history.record("m");

        assert_eq!(history.navigate_up("typed"), Some("m".to_string()));
        assert_eq!(history.navigate_up("ignored"), Some("w".to_string()));
        assert_eq!(history.navigate_down(), Some("m".to_string()));
        assert_eq!(history.navigate_down(), Some("typed".to_string()));
        assert_eq!(history.navigate_down(), None);
    }

    #[test]
    fn history_record_dedupes_consecutive_repeats_and_skips_empty() {
        let mut history = CommandHistory::new();
        history.record("w");
        history.record("w");
        history.record("");
        history.record("m");
        history.record("w");

        assert_eq!(history.navigate_up(""), Some("w".to_string()));
        assert_eq!(history.navigate_up(""), Some("m".to_string()));
        assert_eq!(history.navigate_up(""), Some("w".to_string()));
        assert_eq!(history.navigate_up(""), None);
    }

    #[test]
    fn history_navigate_down_on_fresh_line_is_noop() {
        let mut history = CommandHistory::new();
        history.record("w");

        assert_eq!(history.navigate_down(), None);
    }

    #[test]
    fn history_record_after_navigation_resets_cursor() {
        let mut history = CommandHistory::new();
        history.record("a");
        history.record("b");

        assert_eq!(history.navigate_up("draft"), Some("b".to_string()));
        assert_eq!(history.navigate_up("ignored"), Some("a".to_string()));

        history.record("c");

        // After recording, Up should start at the newest entry "c" again.
        assert_eq!(history.navigate_up("fresh"), Some("c".to_string()));
    }

    #[test]
    fn input_line_insert_advances_cursor_and_appends_to_buffer() {
        let mut input = InputLine::new();
        input.insert_char('a');
        input.insert_char('b');
        input.insert_char('c');

        assert_eq!(input.snapshot(), "abc");
        assert_eq!(input.cursor_chars(), 3);
    }

    #[test]
    fn input_line_left_right_arrows_move_cursor_without_changing_text() {
        let mut input = InputLine::new();
        for c in "hello".chars() {
            input.insert_char(c);
        }

        assert!(input.move_left());
        assert!(input.move_left());
        assert_eq!(input.cursor_chars(), 3);
        assert_eq!(input.snapshot(), "hello");

        assert!(input.move_right());
        assert_eq!(input.cursor_chars(), 4);
        assert_eq!(input.snapshot(), "hello");
    }

    #[test]
    fn input_line_left_at_start_and_right_at_end_are_noops() {
        let mut input = InputLine::new();
        assert!(!input.move_left());
        assert!(!input.move_right());

        input.insert_char('x');
        assert!(!input.move_right());
        assert!(input.move_left());
        assert!(!input.move_left());
    }

    #[test]
    fn input_line_insert_in_middle_splits_and_keeps_tail() {
        let mut input = InputLine::new();
        for c in "ac".chars() {
            input.insert_char(c);
        }
        input.move_left();
        input.insert_char('b');

        assert_eq!(input.snapshot(), "abc");
        assert_eq!(input.cursor_chars(), 2);
    }

    #[test]
    fn input_line_backspace_deletes_char_before_cursor() {
        let mut input = InputLine::new();
        for c in "abc".chars() {
            input.insert_char(c);
        }
        input.move_left();
        assert!(input.backspace());

        assert_eq!(input.snapshot(), "ac");
        assert_eq!(input.cursor_chars(), 1);

        assert!(input.backspace());
        assert_eq!(input.snapshot(), "c");
        assert_eq!(input.cursor_chars(), 0);
        assert!(!input.backspace());
    }

    #[test]
    fn input_line_replace_parks_cursor_at_end() {
        let mut input = InputLine::new();
        input.insert_char('x');
        input.replace("recalled command".to_string());

        assert_eq!(input.snapshot(), "recalled command");
        assert_eq!(input.cursor_chars(), "recalled command".chars().count());
    }

    #[test]
    fn latest_command_returns_to_rolling_days() {
        let command = parse_time_window_command("latest", 5)
            .expect("recognized command")
            .expect("valid latest command");
        let TimeWindow::RollingDays { days } = command else {
            panic!("latest should create a rolling window");
        };

        assert_eq!(days, 5);
    }
}
