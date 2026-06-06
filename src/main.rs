mod charts;
mod constants;
mod data;
mod formatting;
mod model_id;
mod model_overrides;
mod pricing;
mod stats;
mod sync;
mod time_utils;
mod updater;

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Read, Write};
use std::os::unix::io::FromRawFd;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::sync::mpsc;
use std::thread;

use chrono::{DateTime, Duration, Local};
use clap::{Parser, Subcommand};
use crossterm::{
    event::{Event, KeyCode, KeyEvent, KeyModifiers},
    terminal,
};

use constants::{
    AllPricing, ModelPricing, SubscriptionFees, load_subscription_fees, prompt_subscription_fees,
};
use data::UsageEntry;
use data::claude::detect_fast_tier_snapshot as detect_claude_fast_tier_snapshot;
use data::codex::{detect_fast_tier_snapshot as detect_codex_fast_tier_snapshot, get_codex_dir};
use data::gemini::get_gemini_dir;
use data::omp::get_omp_dir;
use formatting::print_model_breakdown;
use stats::{ModelBreakdownRow, VendorTimeSeries};
use time_utils::TimeWindow;

const MIN_TERMINAL_WIDTH: u16 = 60;
const MIN_TERMINAL_HEIGHT: u16 = 35;
const PREFETCH_PAGE_WINDOWS: i32 = 8;
const FULL_CACHE_HORIZON: i64 = i64::MAX / 4;
const DATA_LOADED_NOTICE_MS: u64 = 3_000;

/// Render `body` into an off-screen buffer (by redirecting stdout into a
/// pipe), then write the whole frame in a single `write_all` so the terminal
/// only redraws once. Every captured newline gets a `\x1b[K` injected before
/// it so each line wipes the previous frame's leftover characters, and the
/// frame is bracketed with cursor-hide/home + clear-to-end/cursor-show.
fn render_frame<F: FnOnce()>(body: F) -> io::Result<()> {
    let _ = io::stdout().flush();
    let stdout_fd: libc::c_int = 1;

    let saved = unsafe { libc::dup(stdout_fd) };
    if saved < 0 {
        return Err(io::Error::last_os_error());
    }

    let mut fds: [libc::c_int; 2] = [0; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        let err = io::Error::last_os_error();
        unsafe {
            libc::close(saved);
        }
        return Err(err);
    }
    let read_fd = fds[0];
    let write_fd = fds[1];

    if unsafe { libc::dup2(write_fd, stdout_fd) } < 0 {
        let err = io::Error::last_os_error();
        unsafe {
            libc::close(read_fd);
            libc::close(write_fd);
            libc::close(saved);
        }
        return Err(err);
    }
    unsafe {
        libc::close(write_fd);
    }

    // Drain the pipe on a background thread so writes never block on a
    // full kernel buffer (default ~64 KiB on Linux).
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let drain = thread::spawn(move || {
        let mut buf = Vec::new();
        let mut reader = unsafe { std::fs::File::from_raw_fd(read_fd) };
        let _ = reader.read_to_end(&mut buf);
        let _ = tx.send(buf);
    });

    body();
    let _ = io::stdout().flush();

    // Restoring stdout drops the dup'd write end so the reader hits EOF.
    if unsafe { libc::dup2(saved, stdout_fd) } < 0 {
        let err = io::Error::last_os_error();
        unsafe {
            libc::close(saved);
        }
        let _ = drain.join();
        return Err(err);
    }
    unsafe {
        libc::close(saved);
    }
    let _ = drain.join();
    let captured = rx.recv().unwrap_or_default();

    let mut frame = Vec::with_capacity(captured.len() + 32);
    frame.extend_from_slice(b"\x1b[?25l\x1b[H");
    for &byte in &captured {
        if byte == b'\n' {
            frame.extend_from_slice(b"\x1b[K");
        }
        frame.push(byte);
    }
    frame.extend_from_slice(b"\x1b[J\x1b[?25h");

    let mut stdout = io::stdout().lock();
    stdout.write_all(&frame)?;
    stdout.flush()
}

struct VersionCacheEntry {
    version_str: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntegrityStatus {
    Checking,
    Checked { duration: std::time::Duration },
    Failed,
}

struct AppState {
    vendor: String,
    host: Option<String>,
    local_host_id: Option<String>,
    days: i64,
    time_window: TimeWindow,
    monitor_interval: u64,
    pricing: AllPricing,
    subscription_fees: SubscriptionFees,
    version_cache: HashMap<String, VersionCacheEntry>,
    all_vendor_prompt: Option<String>,
    raw_cache: Option<RawDataCache>,
    raw_refresh: Option<mpsc::Receiver<RawDataCache>>,
    integrity_status: IntegrityStatus,
    integrity_started_at: Option<std::time::Instant>,
}

/// In-memory snapshot of raw vendor entries, scoped to a known scan
/// horizon (in days back from `now`). PageUp/PageDown reuse this cache so
/// they feel instant; only manual `r` and auto-refresh invalidate it.
struct RawDataCache {
    claude: Vec<UsageEntry>,
    codex: Vec<UsageEntry>,
    gemini: Vec<UsageEntry>,
    omp: Vec<UsageEntry>,
    horizon_days: i64,
    local_host_id: Option<String>,
    local_record_keys: HashSet<(String, String)>,
}

struct PromptNotice {
    message: String,
    expires_at: std::time::Instant,
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
#[command(
    about = "Analyze AI coding assistant usage statistics",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Args {
    /// Number of days to look back
    #[arg(long, default_value = "3")]
    days: i64,

    /// Run once and exit (default: monitor mode with 1 hour refresh)
    #[arg(long)]
    once: bool,

    /// Vendor to collect statistics from
    #[arg(long, default_value = "all", value_parser = ["claude", "codex", "gemini", "omp", "all"])]
    vendor: String,

    /// Filter usage to a single machine id
    #[arg(long)]
    host: Option<String>,

    /// Check GitHub releases periodically in monitor mode and restart into a newer binary
    #[arg(long)]
    auto_update: bool,

    /// Seconds between automatic release checks when --auto-update is enabled
    #[arg(
        long,
        default_value_t = vibe_usage_updater::DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS
    )]
    auto_update_interval_seconds: u64,

    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Subcommand, Debug, Clone, Copy, PartialEq, Eq)]
enum CliCommand {
    Sync {
        #[command(subcommand)]
        command: SyncCommand,
    },
}

#[derive(Subcommand, Debug, Clone, Copy, PartialEq, Eq)]
enum SyncCommand {
    Push,
    Pull,
    Status,
    Init {
        /// Replace an existing sync config template
        #[arg(long)]
        force: bool,
    },
    /// Drop the local pulled-records cache and the pull cursor, then refetch
    /// every record this host can see from the server.
    Clean,
}

fn get_terminal_size() -> (u16, u16) {
    match terminal::size() {
        Ok(size) => size,
        Err(_) => {
            let cols = std::env::var("COLUMNS")
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(80);
            let rows = std::env::var("LINES")
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(24);
            (cols, rows)
        }
    }
}

fn check_terminal_size() -> (bool, u16, u16) {
    let (width, height) = get_terminal_size();
    (
        width >= MIN_TERMINAL_WIDTH && height >= MIN_TERMINAL_HEIGHT,
        width,
        height,
    )
}

fn print_terminal_too_small(width: u16, height: u16) {
    print!("\x1b[2J\x1b[H");

    let lines = [
        "Terminal size too small:".to_string(),
        format!("  Width = {}  Height = {}", width, height),
        String::new(),
        "Needed for current config:".to_string(),
        format!(
            "  Width = {}  Height = {}",
            MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT
        ),
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

fn print_loading_data_screen(width: u16, height: u16) {
    print!("\x1b[2J\x1b[H\x1b[?25l");

    let lines = [
        "Loading data...".to_string(),
        "Reading cache and preparing the first view.".to_string(),
    ];
    let max_line_len = lines.iter().map(|line| line.len()).max().unwrap_or(0);
    let top_padding = ((height as usize).saturating_sub(lines.len())) / 2;

    for _ in 0..top_padding {
        println!();
    }
    for line in &lines {
        let left_padding = ((width as usize).saturating_sub(max_line_len)) / 2;
        println!("{}{}", " ".repeat(left_padding), line);
    }
    let _ = io::stdout().flush();
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
    let x_axis_label_lines = 4; // blank + typically 2 label rows + 1 pager hint

    // Monitor prompt: 1 blank + 1 version + 1 separator + 1 "> " = 4
    let prompt_lines = if is_monitor_mode { 4 } else { 0 };

    let min_chart = 5usize;

    if is_all_vendor {
        // Single chart
        let fixed = header_lines
            + table_lines
            + time_span_lines
            + chart_overhead
            + x_axis_label_lines
            + prompt_lines;
        let available = th.saturating_sub(fixed);
        let chart_height = available.max(min_chart).min(60);
        let fits = th >= fixed + min_chart;
        (chart_height, fits)
    } else {
        // Two charts: chart1 (io, no x-axis labels) + chart2 (cache, with x-axis labels)
        let chart1_fixed = chart_overhead;
        let chart2_fixed = chart_overhead + x_axis_label_lines;
        let fixed = header_lines
            + table_lines
            + time_span_lines
            + chart1_fixed
            + chart2_fixed
            + prompt_lines;
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
    granularity: charts::ChartGranularity,
) -> f64 {
    let total_minutes = ((*range_end - *range_start).num_seconds() as f64 / 60.0).max(1.0);
    let min_interval = total_minutes / 100.0;
    let y_axis_width = 7.0;
    let separator_estimate =
        estimate_chart_separator_count(range_start, range_end, granularity) as f64;
    let chart_width = (target_width as f64 - y_axis_width - separator_estimate).max(50.0);
    let terminal_interval = total_minutes / chart_width;
    min_interval.max(terminal_interval)
}

fn estimate_chart_separator_count(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    granularity: charts::ChartGranularity,
) -> usize {
    let span_minutes = ((*range_end - *range_start).num_seconds() as f64 / 60.0).max(1.0);
    let segment_minutes = match granularity {
        charts::ChartGranularity::Hour => 60.0,
        charts::ChartGranularity::Day => 24.0 * 60.0,
        charts::ChartGranularity::Week => 7.0 * 24.0 * 60.0,
        charts::ChartGranularity::Month => 30.0 * 24.0 * 60.0,
        charts::ChartGranularity::Year => 365.0 * 24.0 * 60.0,
    };
    (span_minutes / segment_minutes).ceil().max(1.0) as usize
}

fn display_chart_granularity(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
) -> charts::ChartGranularity {
    let span_minutes = ((*range_end - *range_start).num_seconds() / 60).max(1);
    charts::ChartGranularity::from_span_minutes(span_minutes)
}

fn round_to_nice_interval(optimal: f64) -> i64 {
    let nice = [
        1i64, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320, 5760, 10080, 20160, 40320,
        80640,
    ];
    for &n in &nice {
        if n as f64 >= optimal {
            return n;
        }
    }
    *nice.last().unwrap()
}

#[derive(Clone, Copy)]
enum IntervalSlideDirection {
    Older,
    Newer,
}

fn display_interval_minutes_for_window(
    window: &TimeWindow,
    now: DateTime<Local>,
    target_width: usize,
) -> i64 {
    let (range_start, range_end) = window.bounds(now);
    let granularity = display_chart_granularity(&range_start, &range_end);
    let optimal =
        calculate_optimal_interval_minutes(&range_start, &range_end, target_width, granularity);
    round_to_nice_interval(optimal)
}

fn slide_window_by_display_interval(
    window: &TimeWindow,
    now: DateTime<Local>,
    target_width: usize,
    direction: IntervalSlideDirection,
) -> Option<TimeWindow> {
    let interval_minutes = display_interval_minutes_for_window(window, now, target_width);
    let step = Duration::minutes(interval_minutes.max(1));
    match direction {
        IntervalSlideDirection::Older => window.slide_back_by(now, step),
        IntervalSlideDirection::Newer => window.slide_forward_by(now, step),
    }
}

fn apply_interval_slide_directions<I>(
    window: &TimeWindow,
    now: DateTime<Local>,
    target_width: usize,
    directions: I,
) -> Option<TimeWindow>
where
    I: IntoIterator<Item = IntervalSlideDirection>,
{
    let mut current = window.clone();
    let mut changed = false;
    for direction in directions {
        if let Some(next) = slide_window_by_display_interval(&current, now, target_width, direction)
        {
            current = next;
            changed = true;
        }
    }
    changed.then_some(current)
}

fn interval_slide_direction_for_event(event: &Event) -> Option<IntervalSlideDirection> {
    match event {
        Event::Key(KeyEvent {
            code: KeyCode::Left,
            ..
        }) => Some(IntervalSlideDirection::Newer),
        Event::Key(KeyEvent {
            code: KeyCode::Right,
            ..
        }) => Some(IntervalSlideDirection::Older),
        _ => None,
    }
}

fn collect_interval_slide_directions(
    pending_events: &mut VecDeque<Event>,
    first_direction: IntervalSlideDirection,
) -> Vec<IntervalSlideDirection> {
    let mut directions = vec![first_direction];
    while crossterm::event::poll(std::time::Duration::ZERO).unwrap_or(false) {
        match crossterm::event::read() {
            Ok(event) => {
                if let Some(direction) = interval_slide_direction_for_event(&event) {
                    directions.push(direction);
                } else {
                    pending_events.push_back(event);
                    break;
                }
            }
            Err(_) => break,
        }
    }
    directions
}

/// Get the data directory for a vendor, or None for "all".
fn get_vendor_data_dir(vendor: &str) -> Option<PathBuf> {
    match vendor {
        "codex" => Some(get_codex_dir().join("sessions")),
        "gemini" => Some(get_gemini_dir().join("tmp")),
        "omp" => Some(get_omp_dir().join("agent").join("sessions")),
        "claude" => {
            let dirs = data::claude::get_claude_dirs();
            Some(
                dirs.into_iter()
                    .map(|d| d.join("projects"))
                    .find(|p| p.exists())
                    .unwrap_or_else(|| PathBuf::from("~/.claude/projects")),
            )
        }
        _ => None, // "all" has no single directory
    }
}

/// Days of history that must be on disk to render the current window and
/// several prefetched PageUp windows without going back to the filesystem.
fn compute_required_horizon(window: &TimeWindow, now: DateTime<Local>) -> i64 {
    let (start, _) = window.bounds(now);
    let step = window.page_step();
    let prefetch_start = start - step * PREFETCH_PAGE_WINDOWS;
    let days = now.signed_duration_since(prefetch_start).num_days() + 2;
    days.max(1)
}

fn include_local_for_host_filter(host_filter: Option<&str>, local_host_id: Option<&str>) -> bool {
    match host_filter {
        None => true,
        Some(host) => local_host_id == Some(host),
    }
}

fn merge_remote_records_into_raw_cache(
    cache: &mut RawDataCache,
    records: Vec<data::cache::RemoteUsageRecord>,
) {
    for record in records {
        let Some(host_id) = record.entry.host_id.as_deref() else {
            continue;
        };
        if cache.local_host_id.as_deref() == Some(host_id)
            && cache
                .local_record_keys
                .contains(&(record.vendor.clone(), record.dedup_key.clone()))
        {
            continue;
        }
        match record.vendor.as_str() {
            "claude" => cache.claude.push(record.entry),
            "codex" => cache.codex.push(record.entry),
            "gemini" => cache.gemini.push(record.entry),
            "omp" => cache.omp.push(record.entry),
            _ => {}
        }
    }
}

fn load_local_vendor_cached_records(
    cache_root: &Path,
    vendor: &str,
) -> (Vec<UsageEntry>, HashSet<(String, String)>) {
    let records = data::cache::load_vendor_cached_records(cache_root, vendor);
    let entries = records
        .iter()
        .map(|record| record.entry.clone())
        .collect::<Vec<_>>();
    let keys = records
        .into_iter()
        .filter(|record| !record.dedup_key.is_empty())
        .map(|record| (vendor.to_string(), record.dedup_key))
        .collect::<HashSet<_>>();
    (entries, keys)
}

fn local_cached_raw_cache(
    cache_root: &Path,
    include_local: bool,
    local_host_id: Option<&str>,
) -> RawDataCache {
    let mut local_record_keys = HashSet::new();
    let (claude, claude_keys) = if include_local {
        load_local_vendor_cached_records(cache_root, "claude")
    } else {
        (Vec::new(), HashSet::new())
    };
    local_record_keys.extend(claude_keys);
    let (codex, codex_keys) = if include_local {
        load_local_vendor_cached_records(cache_root, "codex")
    } else {
        (Vec::new(), HashSet::new())
    };
    local_record_keys.extend(codex_keys);
    let (gemini, gemini_keys) = if include_local {
        load_local_vendor_cached_records(cache_root, "gemini")
    } else {
        (Vec::new(), HashSet::new())
    };
    local_record_keys.extend(gemini_keys);
    let (omp, omp_keys) = if include_local {
        load_local_vendor_cached_records(cache_root, "omp")
    } else {
        (Vec::new(), HashSet::new())
    };
    local_record_keys.extend(omp_keys);

    RawDataCache {
        claude,
        codex,
        gemini,
        omp,
        horizon_days: FULL_CACHE_HORIZON,
        local_host_id: local_host_id.map(str::to_string),
        local_record_keys,
    }
}

fn clear_local_raw_cache(cache: &mut RawDataCache) {
    cache.claude.clear();
    cache.codex.clear();
    cache.gemini.clear();
    cache.omp.clear();
    cache.local_record_keys.clear();
}

fn read_all_vendor_cached_snapshot_for_hosts(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
) -> RawDataCache {
    let cache_root = data::cache::default_cache_dir();
    let include_local = include_local_for_host_filter(host_filter, local_host_id);
    let mut cache = local_cached_raw_cache(&cache_root, include_local, local_host_id);
    let host_set = host_filter.map(|host| HashSet::from([host.to_string()]));
    let remote_records = data::cache::load_remote_entries(&cache_root, host_set.as_ref());
    merge_remote_records_into_raw_cache(&mut cache, remote_records);
    cache
}

fn refresh_all_vendor_raw_full(local_host_id: Option<&str>) -> RawDataCache {
    let cache_root = data::cache::default_cache_dir();
    let claude_fast_tier = detect_claude_fast_tier_snapshot();
    let _ = data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "claude",
        data::claude::collect_usage_files(None),
        claude_fast_tier,
        data::claude::read_jsonl_file_records,
    );

    let codex_dir = get_codex_dir().join("sessions");
    let codex_fast_tier = detect_codex_fast_tier_snapshot();
    let _ = data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "codex",
        data::codex::collect_usage_files(&codex_dir, None),
        codex_fast_tier,
        data::codex::read_codex_file_records,
    );

    let gemini_dir = get_gemini_dir().join("tmp");
    let _ = data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "gemini",
        data::gemini::collect_usage_files(&gemini_dir, None),
        0,
        data::gemini::read_gemini_file_records,
    );

    let omp_dir = get_omp_dir().join("agent").join("sessions");
    let _ = data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "omp",
        data::omp::collect_usage_files(&omp_dir, None),
        0,
        data::omp::read_omp_file_records,
    );

    local_cached_raw_cache(&cache_root, true, local_host_id)
}

/// Ensure `state.raw_cache` covers at least `required_horizon` days back.
/// Returns a reference to the populated cache so callers can filter without
/// touching the filesystem again.
fn ensure_raw_cache(state: &mut AppState, required_horizon: i64) -> &RawDataCache {
    let needs_load = match &state.raw_cache {
        None => true,
        Some(cache) => cache.horizon_days < required_horizon,
    };
    if needs_load {
        state.raw_cache = Some(read_all_vendor_cached_snapshot_for_hosts(
            state.host.as_deref(),
            state.local_host_id.as_deref(),
        ));
    }
    state.raw_cache.as_ref().unwrap()
}

fn start_background_raw_refresh(state: &mut AppState) {
    if state.raw_refresh.is_some() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    let host_filter = state.host.clone();
    let local_host_id = state.local_host_id.clone();
    state.raw_refresh = Some(rx);
    state.integrity_status = IntegrityStatus::Checking;
    state.integrity_started_at = Some(std::time::Instant::now());
    thread::spawn(move || {
        let mut refreshed = refresh_all_vendor_raw_full(local_host_id.as_deref());
        if !include_local_for_host_filter(host_filter.as_deref(), local_host_id.as_deref()) {
            clear_local_raw_cache(&mut refreshed);
        }
        let cache_root = data::cache::default_cache_dir();
        let host_set = host_filter
            .as_ref()
            .map(|host| HashSet::from([host.clone()]));
        let remote_records = data::cache::load_remote_entries(&cache_root, host_set.as_ref());
        merge_remote_records_into_raw_cache(&mut refreshed, remote_records);
        let _ = tx.send(refreshed);
    });
}

fn poll_background_raw_refresh(state: &mut AppState) -> bool {
    let Some(rx) = state.raw_refresh.take() else {
        return false;
    };
    match rx.try_recv() {
        Ok(cache) => {
            state.raw_cache = Some(cache);
            let duration = state
                .integrity_started_at
                .take()
                .map(|started_at| started_at.elapsed())
                .unwrap_or_default();
            state.integrity_status = IntegrityStatus::Checked { duration };
            true
        }
        Err(mpsc::TryRecvError::Empty) => {
            state.raw_refresh = Some(rx);
            false
        }
        Err(mpsc::TryRecvError::Disconnected) => false,
    }
}

fn poll_sync_worker_status(
    worker: Option<&sync::worker::SyncWorker>,
    observed_revision: &mut u64,
) -> Option<sync::worker::SyncStats> {
    let worker = worker?;
    let stats = worker.stats();
    if stats.revision == *observed_revision {
        return None;
    }
    *observed_revision = stats.revision;
    Some(stats)
}

fn current_sync_status(worker: Option<&sync::worker::SyncWorker>) -> Option<String> {
    worker.and_then(|worker| format_monitor_sync_status(&worker.stats()))
}

fn data_loaded_notice(load_duration: std::time::Duration) -> PromptNotice {
    PromptNotice {
        message: format!("Data loaded in {} ms", load_duration.as_millis()),
        expires_at: std::time::Instant::now()
            + std::time::Duration::from_millis(DATA_LOADED_NOTICE_MS),
    }
}

fn active_prompt_placeholder(
    notice: Option<&PromptNotice>,
    refresh_in: Option<std::time::Duration>,
) -> (String, usize) {
    if let Some(notice) = notice {
        formatting::prompt_placeholder(&notice.message)
    } else {
        formatting::prompt_watermark(refresh_in)
    }
}

fn integrity_status_marker(status: IntegrityStatus) -> (String, usize) {
    match status {
        IntegrityStatus::Checking => formatting::integrity_checking_marker(),
        IntegrityStatus::Checked { duration } => {
            formatting::integrity_checked_marker(&format_integrity_duration(duration))
        }
        IntegrityStatus::Failed => formatting::integrity_failed_marker(),
    }
}

fn integrity_status_from_verification(
    verification: &sync::integrity::IntegrityVerification,
    duration: std::time::Duration,
) -> IntegrityStatus {
    match verification {
        sync::integrity::IntegrityVerification::Checked { .. } => {
            IntegrityStatus::Checked { duration }
        }
        sync::integrity::IntegrityVerification::Failed { .. } => IntegrityStatus::Failed,
    }
}

fn format_integrity_duration(duration: std::time::Duration) -> String {
    if duration.as_millis() < 1_000 {
        format!("{} ms", duration.as_millis())
    } else {
        format!("{:.2} s", duration.as_secs_f64())
    }
}

fn monitor_sync_interval(monitor_interval: std::time::Duration) -> std::time::Duration {
    (monitor_interval / 3).max(std::time::Duration::from_secs(60))
}

fn monitor_deadlines_after_interval_change(
    now: std::time::Instant,
    monitor_interval_seconds: u64,
) -> (std::time::Instant, std::time::Instant) {
    let monitor_interval = std::time::Duration::from_secs(monitor_interval_seconds);
    (
        now + monitor_interval,
        now + monitor_sync_interval(monitor_interval),
    )
}

fn auto_update_deadline_after(
    now: std::time::Instant,
    auto_update_interval_seconds: u64,
) -> std::time::Instant {
    now + vibe_usage_updater::normalize_auto_update_interval(auto_update_interval_seconds)
}

fn format_manual_sync_progress(event: &sync::engine::SyncProgress) -> Option<String> {
    match event {
        sync::engine::SyncProgress::UploadPlanned {
            total_records,
            total_batches,
            skipped_records,
        } => {
            if *skipped_records == 0 {
                Some(format!(
                    "sync push: {total_records} records to upload in {total_batches} batches"
                ))
            } else {
                Some(format!(
                    "sync push: {total_records} records to upload in {total_batches} batches, {skipped_records} already synced"
                ))
            }
        }
        sync::engine::SyncProgress::UploadBatchFinished {
            batch_index,
            total_batches,
            uploaded_records,
            total_records,
            accepted,
            ignored,
        } => Some(format!(
            "sync push: [{}] {batch_index}/{total_batches} batches, {uploaded_records}/{total_records} records, accepted {accepted}, ignored {ignored}",
            progress_bar(*batch_index, *total_batches)
        )),
        sync::engine::SyncProgress::UploadFinished {
            uploaded_records,
            total_records,
            accepted,
            ignored,
        } => Some(format!(
            "sync push: complete, {uploaded_records}/{total_records} records, accepted {accepted}, ignored {ignored}"
        )),
        sync::engine::SyncProgress::PullPageFinished {
            page_index,
            pulled_records,
            max_seq,
            ..
        } => Some(format!(
            "sync pull: page {page_index}, {pulled_records} records pulled, latest seq {max_seq}"
        )),
        sync::engine::SyncProgress::PullFinished {
            pages,
            pulled_records,
            max_seq,
        } => Some(format!(
            "sync pull: complete, {pages} pages, {pulled_records} records pulled, latest seq {max_seq}"
        )),
        sync::engine::SyncProgress::IntegrityUnsupported => {
            Some("sync integrity: server does not support integrity reports".to_string())
        }
        sync::engine::SyncProgress::IntegrityReportSubmitted {
            record_count,
            range_end_utc,
        } => Some(format!(
            "sync integrity: submitted {record_count} records through {range_end_utc}"
        )),
        sync::engine::SyncProgress::IntegrityCheckFinished { verification } => Some(format!(
            "sync integrity: {}",
            format_integrity_verification(verification)
        )),
    }
}

fn format_http_progress(event: &sync::client::HttpProgress) -> String {
    match event {
        sync::client::HttpProgress::RateLimited {
            attempt,
            retry_after,
        } => format!(
            "sync: rate limited, retrying in {} (attempt {attempt})",
            format_retry_duration(*retry_after)
        ),
    }
}

fn format_monitor_sync_status(stats: &sync::worker::SyncStats) -> Option<String> {
    format_monitor_sync_status_at(stats, chrono::Utc::now())
}

fn format_monitor_sync_status_at(
    stats: &sync::worker::SyncStats,
    now: DateTime<chrono::Utc>,
) -> Option<String> {
    if stats.running {
        return Some(match stats.progress.as_ref() {
            Some(progress) => format!("Sync: {}", format_monitor_worker_progress(progress)),
            None => "Sync: running".to_string(),
        });
    }

    if let Some(error) = stats.last_error.as_ref() {
        return Some(format!("Sync: error: {error}"));
    }

    if stats.success_count > 0 {
        let finished = stats
            .last_finished_at
            .as_deref()
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
            .map(|value| value.with_timezone(&chrono::Utc));
        if let Some(finished) = finished {
            return Some(format!(
                "Sync: checked {}",
                format_sync_age(now.signed_duration_since(finished))
            ));
        }
        return Some("Sync: checked".to_string());
    }

    None
}

fn format_monitor_worker_progress(progress: &sync::worker::SyncWorkerProgress) -> String {
    match progress {
        sync::worker::SyncWorkerProgress::Sync(event) => match event {
            sync::engine::SyncProgress::UploadPlanned {
                total_records,
                total_batches,
                ..
            } => format!("push queued {total_records} records in {total_batches} batches"),
            sync::engine::SyncProgress::UploadBatchFinished {
                batch_index,
                total_batches,
                uploaded_records,
                total_records,
                ..
            } => format!(
                "push {batch_index}/{total_batches} batches, {uploaded_records}/{total_records} records"
            ),
            sync::engine::SyncProgress::UploadFinished {
                uploaded_records,
                total_records,
                ..
            } => format!("push complete, {uploaded_records}/{total_records} records"),
            sync::engine::SyncProgress::PullPageFinished {
                page_index,
                pulled_records,
                max_seq,
                ..
            } => format!(
                "pull page {page_index}, {pulled_records} records pulled, latest seq {max_seq}"
            ),
            sync::engine::SyncProgress::PullFinished {
                pages,
                pulled_records,
                max_seq,
            } => format!(
                "pull complete, {pages} pages, {pulled_records} records pulled, latest seq {max_seq}"
            ),
            sync::engine::SyncProgress::IntegrityUnsupported => "integrity unsupported".to_string(),
            sync::engine::SyncProgress::IntegrityReportSubmitted {
                record_count,
                range_end_utc,
            } => format!("integrity submitted {record_count} records through {range_end_utc}"),
            sync::engine::SyncProgress::IntegrityCheckFinished { verification } => {
                format!("integrity {}", format_integrity_verification(verification))
            }
        },
        sync::worker::SyncWorkerProgress::Http(event) => match event {
            sync::client::HttpProgress::RateLimited {
                attempt,
                retry_after,
            } => format!(
                "rate limited, retrying in {} (attempt {attempt})",
                format_retry_duration(*retry_after)
            ),
        },
    }
}

fn format_integrity_verification(verification: &sync::integrity::IntegrityVerification) -> String {
    match verification {
        sync::integrity::IntegrityVerification::Checked { checked_hosts } => {
            format!("checked {checked_hosts} hosts")
        }
        sync::integrity::IntegrityVerification::Failed { failures } => {
            format!("failed {} hosts", failures.len())
        }
    }
}

fn format_sync_age(age: Duration) -> String {
    let seconds = age.num_seconds().max(0);
    if seconds < 60 {
        "just now".to_string()
    } else if seconds < 3600 {
        format!("{} min ago", seconds / 60)
    } else if seconds < 86_400 {
        format!("{} h ago", seconds / 3600)
    } else {
        format!("{} d ago", seconds / 86_400)
    }
}

fn progress_bar(done: usize, total: usize) -> String {
    const WIDTH: usize = 20;
    let filled = if total == 0 {
        WIDTH
    } else {
        done.saturating_mul(WIDTH).min(total.saturating_mul(WIDTH)) / total
    };
    format!("{}{}", "#".repeat(filled), "-".repeat(WIDTH - filled))
}

fn format_retry_duration(duration: std::time::Duration) -> String {
    if duration.as_millis() == 0 {
        "0s".to_string()
    } else if duration.as_secs() < 10 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}s", duration.as_secs())
    }
}

fn prompt_notice_expired(notice: &Option<PromptNotice>, now: std::time::Instant) -> bool {
    notice
        .as_ref()
        .map(|notice| now >= notice.expires_at)
        .unwrap_or(false)
}

#[derive(Clone, Copy)]
enum PromptBlockMode {
    Append,
    Replace,
}

fn truncate_chars(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

fn format_monitor_status_line(left: &str, right: Option<&str>, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let Some(right) = right.filter(|text| !text.is_empty()) else {
        return truncate_chars(left, width);
    };
    let right_width = right.chars().count();
    if right_width >= width {
        return truncate_chars(right, width);
    }

    let left_width = left.chars().count();
    if left_width + right_width < width {
        return format!(
            "{}{}{}",
            left,
            " ".repeat(width - left_width - right_width),
            right
        );
    }

    let left_limit = width.saturating_sub(right_width + 1);
    format!("{} {}", truncate_chars(left, left_limit), right)
}

fn monitor_status_title(state: &mut AppState) -> String {
    if state.vendor == "all" {
        state
            .all_vendor_prompt
            .clone()
            .unwrap_or_else(|| "All Vendors Comparison".to_string())
    } else {
        let vendor = state.vendor.clone();
        get_version(state, &vendor)
    }
}

fn render_monitor_prompt_block(
    state: &mut AppState,
    input: &InputLine,
    too_small: bool,
    notice: Option<&PromptNotice>,
    sync_status: Option<&str>,
    mode: PromptBlockMode,
    refresh_in: Option<std::time::Duration>,
) -> bool {
    if too_small {
        return false;
    }

    let (width, _) = get_terminal_size();
    let width = width as usize;
    let title = monitor_status_title(state);
    let status_line = format_monitor_status_line(&title, sync_status, width);

    match mode {
        PromptBlockMode::Append => {
            println!("\n\r{}\r", status_line);
        }
        PromptBlockMode::Replace => {
            print!("\x1b[2A\r\x1b[K{}\r\n", status_line);
        }
    }
    println!("\r{}\r", "-".repeat(width));
    render_prompt_line(input, too_small, notice, state.integrity_status, refresh_in);
    true
}

fn render_prompt_line(
    input: &InputLine,
    too_small: bool,
    notice: Option<&PromptNotice>,
    integrity_status: IntegrityStatus,
    refresh_in: Option<std::time::Duration>,
) {
    if too_small {
        return;
    }

    let (width, _) = get_terminal_size();
    let width = width as usize;
    let input_visible = 2 + input.char_count();
    let (status, status_visible) = integrity_status_marker(integrity_status);
    let status_fits = width >= input_visible + 2 + status_visible;

    print!("\r\x1b[K> {}", input.snapshot());

    if input.is_empty() {
        let (mark, mark_visible) = active_prompt_placeholder(notice, refresh_in);
        let watermark_fits = if status_fits {
            width >= 2 + mark_visible + 2 + status_visible
        } else {
            width >= 2 + mark_visible
        };
        if watermark_fits {
            print!("{}", mark);
        }
    }

    if status_fits {
        let status_col = width.saturating_sub(status_visible) + 1;
        print!("\x1b[{}G{}", status_col, status);
    }

    let cursor_col = (3 + input.cursor_chars()).min(width.max(1));
    print!("\x1b[{}G", cursor_col);
    io::stdout().flush().unwrap();
}

fn host_label(host: Option<&str>) -> &str {
    host.unwrap_or("all")
}

fn parse_host_selection(selection: &str) -> Result<Option<String>, &'static str> {
    let selection = selection.trim();
    if selection == "all" {
        return Ok(None);
    }
    if vibe_usage_proto::is_valid_host_id(selection) {
        Ok(Some(selection.to_string()))
    } else {
        Err("Usage: host [all|HOST], where HOST matches [a-z0-9_-]{1,64}")
    }
}

fn known_host_ids(local_host_id: Option<&str>) -> Vec<String> {
    let cache_root = data::cache::default_cache_dir();
    let mut hosts = HashSet::new();
    if let Some(local_host_id) = local_host_id {
        hosts.insert(local_host_id.to_string());
    }
    for record in data::cache::load_remote_entries(&cache_root, None) {
        if let Some(host_id) = record.entry.host_id {
            hosts.insert(host_id);
        }
    }
    let mut hosts = hosts.into_iter().collect::<Vec<_>>();
    hosts.sort();
    hosts
}

/// Loaded and filtered data for all vendors.
struct AllVendorData {
    claude: Vec<UsageEntry>,
    codex: Vec<UsageEntry>,
    gemini: Vec<UsageEntry>,
    omp: Vec<UsageEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowDataState {
    NoSourceData,
    EmptyWindow,
    Populated,
}

fn classify_window_data(has_source_data: bool, has_window_data: bool) -> WindowDataState {
    match (has_source_data, has_window_data) {
        (false, _) => WindowDataState::NoSourceData,
        (true, false) => WindowDataState::EmptyWindow,
        (true, true) => WindowDataState::Populated,
    }
}

fn raw_cache_has_any_vendor_data(cache: &RawDataCache) -> bool {
    !cache.claude.is_empty()
        || !cache.codex.is_empty()
        || !cache.gemini.is_empty()
        || !cache.omp.is_empty()
}

fn all_vendor_data_has_window_data(all_data: &AllVendorData) -> bool {
    !all_data.claude.is_empty()
        || !all_data.codex.is_empty()
        || !all_data.gemini.is_empty()
        || !all_data.omp.is_empty()
}

fn load_all_vendor_data(state: &mut AppState, now: DateTime<Local>) -> AllVendorData {
    let horizon = compute_required_horizon(&state.time_window, now);
    let window = state.time_window.clone();
    let cache = ensure_raw_cache(state, horizon);
    AllVendorData {
        claude: data::filter_usage_data_by_window(&cache.claude, &window, now),
        codex: data::filter_usage_data_by_window(&cache.codex, &window, now),
        gemini: data::filter_usage_data_by_window(&cache.gemini, &window, now),
        omp: data::filter_usage_data_by_window(&cache.omp, &window, now),
    }
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
                    entry.usage.input_tokens
                        + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens
                        + entry.usage.reasoning_output_tokens
                }
                _ => {
                    entry.usage.input_tokens
                        + entry.usage.output_tokens
                        + entry.usage.cache_read_input_tokens
                        + entry.usage.cache_creation_input_tokens
                }
            } as f64;

            let parsed = entry
                .parsed_timestamp
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
    if !all_data.omp.is_empty() {
        process_data(&all_data.omp, "Oh My Pi", &mut time_series);
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
    if !all_data.omp.is_empty() {
        all_stats.extend(stats::calculate_omp_model_breakdown(&all_data.omp, pricing));
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
        ("omp", &all_data.omp, 0.0),
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

            if let Some(costs) = entry.costs {
                api_cost += costs.input + costs.output + costs.cache_read + costs.cache_creation;
                continue;
            }

            let pricing_vendor = stats::pricing_vendor_for_entry(vendor, entry);
            let p = pricing.pricing_for_entry(pricing_vendor, &entry.model, entry.fast_tier);
            api_cost +=
                ModelPricing::tier_cost(entry.usage.input_tokens, p.input, p.input_above_200k);
            api_cost +=
                ModelPricing::tier_cost(entry.usage.output_tokens, p.output, p.output_above_200k);
            api_cost += ModelPricing::tier_cost(
                entry.usage.cache_read_input_tokens,
                p.cache_input,
                p.cache_input_above_200k,
            );
            api_cost += match (vendor, pricing_vendor) {
                ("omp", _) => ModelPricing::tier_cost(
                    entry.usage.cache_creation_input_tokens,
                    p.cache_output,
                    p.cache_output_above_200k,
                ),
                (_, "codex") => ModelPricing::tier_cost(
                    entry.usage.reasoning_output_tokens,
                    p.output,
                    p.output_above_200k,
                ),
                (_, "gemini") => ModelPricing::tier_cost(
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
    if let Some(cached) = state.version_cache.get(vendor) {
        return cached.version_str.clone();
    }

    let (cmd, display_name) = match vendor {
        "claude" => ("claude", "Claude Code"),
        "codex" => ("codex", "Codex"),
        "gemini" => ("gemini", "Gemini CLI"),
        _ => return String::new(),
    };

    let current_version = ProcessCommand::new(cmd)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
                s.split_whitespace()
                    .next()
                    .and_then(|v| {
                        if v.chars()
                            .next()
                            .map(|c| c.is_ascii_digit())
                            .unwrap_or(false)
                        {
                            Some(v.to_string())
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        let re_match: String = s
                            .chars()
                            .skip_while(|c| !c.is_ascii_digit())
                            .take_while(|c| c.is_ascii_digit() || *c == '.')
                            .collect();
                        if re_match.is_empty() {
                            None
                        } else {
                            Some(re_match)
                        }
                    })
            } else {
                None
            }
        });

    let version_str = format_local_version_label(display_name, current_version);

    state.version_cache.insert(
        vendor.to_string(),
        VersionCacheEntry {
            version_str: version_str.clone(),
        },
    );

    version_str
}

fn format_local_version_label(display_name: &str, current_version: Option<String>) -> String {
    match current_version {
        Some(ver) => format!("{} ({})", display_name, ver),
        None => display_name.to_string(),
    }
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

    let data_points =
        time_utils::generate_interval_times(&start_rounded, range_end, interval_minutes).len();

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

fn parse_time_window_command(
    command: &str,
    current_days: i64,
) -> Option<Result<TimeWindow, String>> {
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

    // Cache feeds both the raw-emptiness check and the filtered slice.
    let horizon = compute_required_horizon(&state.time_window, now);
    let _ = ensure_raw_cache(state, horizon);
    let cache = state.raw_cache.as_ref().expect("cache populated");
    let vendor = state.vendor.clone();
    let raw_for_vendor: &[UsageEntry] = match vendor.as_str() {
        "claude" => &cache.claude,
        "codex" => &cache.codex,
        "gemini" => &cache.gemini,
        "omp" => &cache.omp,
        _ => &cache.claude,
    };
    let filtered = data::filter_usage_data_by_window(raw_for_vendor, &state.time_window, now);
    if classify_window_data(!raw_for_vendor.is_empty(), !filtered.is_empty())
        == WindowDataState::NoSourceData
    {
        if !once {
            print!("\x1b[2J\x1b[H");
        }
        println!("No usage data found.");
        return Some(false);
    }
    let vendor = &vendor;
    let model_stats = match vendor.as_str() {
        "codex" => stats::calculate_codex_model_breakdown(&filtered, &state.pricing),
        "gemini" => stats::calculate_gemini_model_breakdown(&filtered, &state.pricing),
        "omp" => stats::calculate_omp_model_breakdown(&filtered, &state.pricing),
        _ => stats::calculate_claude_model_breakdown(&filtered, &state.pricing),
    };

    // Pre-check whether table will be displayed and total height fits
    let table_mode = formatting::get_table_display_mode(width, height, model_stats.len());
    let mut will_print_table = table_mode != "hidden";

    let target_width = get_chart_target_width();
    let (mut chart_height, mut fits) =
        calculate_chart_height(!once, will_print_table, model_stats.len(), false);
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

    let vendor_name = match vendor.as_str() {
        "codex" => "Codex",
        "gemini" => "Gemini CLI",
        "omp" => "Oh My Pi",
        _ => "Claude Code",
    };

    println!("Calculating {} usage...", vendor_name);
    println!("{}", showing_data_line(&state.time_window, now));
    if !once {
        println!(
            "Monitor mode: Refreshing every {} seconds (Press Ctrl+C to exit)",
            state.monitor_interval
        );
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

    let granularity = display_chart_granularity(&range_start, &range_end);
    let optimal =
        calculate_optimal_interval_minutes(&range_start, &range_end, target_width, granularity);
    let interval_minutes = round_to_nice_interval(optimal);

    let model_ts = match vendor.as_str() {
        "codex" => {
            stats::calculate_codex_model_token_breakdown_time_series(&filtered, interval_minutes)
        }
        "gemini" => {
            stats::calculate_gemini_model_token_breakdown_time_series(&filtered, interval_minutes)
        }
        "omp" => {
            stats::calculate_omp_model_token_breakdown_time_series(&filtered, interval_minutes)
        }
        _ => stats::calculate_claude_model_token_breakdown_time_series(&filtered, interval_minutes),
    };

    let included_models: HashSet<String> = model_stats.iter().map(|s| s.model.clone()).collect();

    let table_w = if will_print_table {
        formatting::get_table_width(formatting::get_table_display_mode(
            width,
            height,
            model_stats.len(),
        ))
    } else {
        0
    };
    let table_pad = formatting::center_pad(width as usize, table_w);

    print_time_span_info(
        &range_start,
        &range_end,
        interval_minutes,
        width,
        &table_pad,
    );

    charts::print_multi_line_chart(
        &model_ts,
        chart_height,
        &range_start,
        &range_end,
        "io",
        false,
        Some(target_width),
        interval_minutes,
        granularity,
        vendor,
        Some(&included_models),
        true,
        Some(width as usize),
    );
    charts::print_multi_line_chart(
        &model_ts,
        chart_height,
        &range_start,
        &range_end,
        "cache",
        true,
        Some(target_width),
        interval_minutes,
        granularity,
        vendor,
        Some(&included_models),
        true,
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
    let granularity = display_chart_granularity(&range_start, &range_end);
    let optimal =
        calculate_optimal_interval_minutes(&range_start, &range_end, target_width, granularity);
    let interval_minutes = round_to_nice_interval(optimal);

    let all_data = load_all_vendor_data(state, now);

    // Compute and cache the weighted cost prompt for show_prompt reuse
    let (weighted_cost, total_savings) = calculate_weighted_cost_per_mtok(
        &all_data,
        projection_days,
        &state.pricing,
        &state.subscription_fees,
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
    let has_source_data = state
        .raw_cache
        .as_ref()
        .is_some_and(raw_cache_has_any_vendor_data);
    let data_state =
        classify_window_data(has_source_data, all_vendor_data_has_window_data(&all_data));
    if data_state == WindowDataState::NoSourceData {
        println!("No usage data found from any vendor.");
        return Some(false);
    }

    // Pre-check whether table will be displayed
    let table_mode = formatting::get_table_display_mode(width, height, all_model_stats.len());
    let mut will_print_table = table_mode != "hidden";

    // Check total height fits before printing anything
    let (mut chart_height, mut fits) =
        calculate_chart_height(!once, will_print_table, all_model_stats.len(), true);
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

    println!("Calculating usage across all vendors...");
    println!("{}", showing_data_line(&state.time_window, now));
    if !once {
        println!(
            "Monitor mode: Refreshing every {} seconds (Press Ctrl+C to exit)",
            state.monitor_interval
        );
    }

    let effective_height = if will_print_table { height } else { 0 };
    print_model_breakdown(
        &all_model_stats,
        projection_days,
        Some(width),
        Some(effective_height),
        "all",
        &state.subscription_fees,
    );

    let table_w = if will_print_table {
        formatting::get_table_width(formatting::get_table_display_mode(
            width,
            height,
            all_model_stats.len(),
        ))
    } else {
        0
    };
    let table_pad = formatting::center_pad(width as usize, table_w);

    print_time_span_info(
        &range_start,
        &range_end,
        interval_minutes,
        width,
        &table_pad,
    );

    charts::print_vendor_comparison_chart(
        &vendor_time_series,
        chart_height,
        &range_start,
        &range_end,
        Some(target_width),
        interval_minutes,
        granularity,
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

fn run_cli_command(command: CliCommand, sync_config: sync::config::SyncConfig) -> i32 {
    match command {
        CliCommand::Sync { command } => run_sync_command(command, sync_config),
    }
}

fn run_sync_command(command: SyncCommand, sync_config: sync::config::SyncConfig) -> i32 {
    match command {
        SyncCommand::Status => {
            print_sync_status(&sync_config);
            0
        }
        SyncCommand::Init { force } => match sync::config::init_sync_config(force) {
            Ok(path) => {
                println!("wrote sync config template: {}", path.display());
                println!("edit the template, then set sync.enabled to true");
                0
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                eprintln!("vibe-usage: sync config already exists; pass --force to replace it");
                1
            }
            Err(err) => {
                eprintln!("vibe-usage: sync init failed: {err}");
                1
            }
        },
        SyncCommand::Push | SyncCommand::Pull | SyncCommand::Clean => {
            let sync::config::SyncConfig::Enabled(config) = sync_config else {
                eprintln!("vibe-usage: sync is disabled");
                return 1;
            };
            let cache_root = data::cache::default_cache_dir();
            let client = sync::client::SyncHttpClient::new_with_progress(config.clone(), |event| {
                eprintln!("{}", format_http_progress(event));
            });
            let result = match command {
                SyncCommand::Push => {
                    eprintln!("sync push: refreshing local cache");
                    let _ = refresh_all_vendor_raw_full(Some(&config.machine_id));
                    let mut on_progress = |event: &sync::engine::SyncProgress| {
                        if let Some(message) = format_manual_sync_progress(event) {
                            eprintln!("{message}");
                        }
                    };
                    sync::engine::run_upload_once_with_progress(
                        &cache_root,
                        &config,
                        &client,
                        &mut on_progress,
                    )
                    .and_then(|()| {
                        sync::engine::run_integrity_once_with_repair(
                            &cache_root,
                            &config,
                            &client,
                            &mut on_progress,
                        )
                    })
                }
                SyncCommand::Pull => {
                    let mut on_progress = |event: &sync::engine::SyncProgress| {
                        if let Some(message) = format_manual_sync_progress(event) {
                            eprintln!("{message}");
                        }
                    };
                    sync::engine::run_pull_and_integrity_once_with_progress(
                        &cache_root,
                        &config,
                        &client,
                        &mut on_progress,
                    )
                }
                SyncCommand::Clean => run_sync_clean(&cache_root, &config, &client),
                SyncCommand::Status => unreachable!("status handled above"),
                SyncCommand::Init { .. } => unreachable!("init handled above"),
            };
            match result {
                Ok(()) => {
                    println!("sync {} complete", sync_command_name(command));
                    0
                }
                Err(err) => {
                    eprintln!(
                        "vibe-usage: sync {} failed: {}",
                        sync_command_name(command),
                        err
                    );
                    1
                }
            }
        }
    }
}

fn run_sync_clean(
    cache_root: &Path,
    config: &sync::config::EnabledSyncConfig,
    client: &sync::client::SyncHttpClient,
) -> Result<(), sync::engine::SyncError> {
    let removed_files = data::cache::clear_remote_cache(cache_root)?;
    let removed_state = sync::state::clear_sync_state(cache_root)?;
    eprintln!(
        "sync clean: cleared {removed_files} cached remote file(s); sync cursor {}",
        if removed_state {
            "reset"
        } else {
            "already absent"
        }
    );
    eprintln!("sync clean: refetching records from server");
    let mut on_progress = |event: &sync::engine::SyncProgress| {
        if let Some(message) = format_manual_sync_progress(event) {
            eprintln!("{message}");
        }
    };
    sync::engine::run_pull_and_integrity_once_with_progress(
        cache_root,
        config,
        client,
        &mut on_progress,
    )
}

fn print_sync_status(sync_config: &sync::config::SyncConfig) {
    match sync_config {
        sync::config::SyncConfig::Disabled => {
            println!("sync: disabled");
        }
        sync::config::SyncConfig::Enabled(config) => {
            let state = sync::state::load_sync_state(&data::cache::default_cache_dir());
            println!("sync: enabled");
            println!("server_url: {}", config.server_url);
            println!("machine_id: {}", config.machine_id);
            println!("last_seen_seq: {}", state.last_seen_seq);
            println!(
                "last_successful_sync: {}",
                state.last_successful_sync.as_deref().unwrap_or("never")
            );
            println!(
                "last_error: {}",
                state.last_error.as_deref().unwrap_or("none")
            );
            let client = sync::client::SyncHttpClient::new(config.clone());
            match client.machines() {
                Ok(list) => {
                    println!("machines:");
                    if list.machines.is_empty() {
                        println!("  none");
                    } else {
                        for machine in list.machines {
                            println!(
                                "  {} last_seen={} record_count={}",
                                machine.host_id, machine.last_seen, machine.record_count
                            );
                        }
                    }
                }
                Err(err) => {
                    println!("machines: unavailable ({err})");
                }
            }
        }
    }
}

fn sync_command_name(command: SyncCommand) -> &'static str {
    match command {
        SyncCommand::Push => "push",
        SyncCommand::Pull => "pull",
        SyncCommand::Status => "status",
        SyncCommand::Init { .. } => "init",
        SyncCommand::Clean => "clean",
    }
}

fn main() {
    let args = Args::parse();

    let sync_config = sync::config::load_sync_config(false);
    if let Some(command) = args.command {
        let code = run_cli_command(command, sync_config);
        std::process::exit(code);
    }

    let pricing = pricing::load_layered();
    let subscription_fees = load_subscription_fees().unwrap_or_else(prompt_subscription_fees);
    let local_host_id = match &sync_config {
        sync::config::SyncConfig::Enabled(config) => Some(config.machine_id.clone()),
        sync::config::SyncConfig::Disabled => None,
    };

    // Validate vendor data directory on startup (matches Python behavior)
    if let Some(data_dir) = get_vendor_data_dir(&args.vendor)
        && !data_dir.exists()
    {
        eprintln!("Error: Data directory not found at {}", data_dir.display());
        std::process::exit(1);
    }

    let mut state = AppState {
        vendor: args.vendor.clone(),
        host: args.host.clone(),
        local_host_id,
        days: args.days,
        time_window: TimeWindow::rolling_days(args.days),
        monitor_interval: 3600,
        pricing,
        subscription_fees,
        version_cache: HashMap::new(),
        all_vendor_prompt: None,
        raw_cache: None,
        raw_refresh: None,
        integrity_status: IntegrityStatus::Checking,
        integrity_started_at: None,
    };

    if args.once {
        let mut refreshed = refresh_all_vendor_raw_full(state.local_host_id.as_deref());
        if !include_local_for_host_filter(state.host.as_deref(), state.local_host_id.as_deref()) {
            clear_local_raw_cache(&mut refreshed);
        }
        let cache_root = data::cache::default_cache_dir();
        let host_set = state
            .host
            .as_ref()
            .map(|host| HashSet::from([host.clone()]));
        let remote_records = data::cache::load_remote_entries(&cache_root, host_set.as_ref());
        merge_remote_records_into_raw_cache(&mut refreshed, remote_records);
        state.raw_cache = Some(refreshed);
        let result = print_stats(&mut state, true);
        match result {
            None => std::process::exit(1),
            Some(false) => std::process::exit(0),
            Some(true) => {}
        }
    } else {
        // Monitor mode
        let (width, height) = get_terminal_size();
        let load_started = std::time::Instant::now();
        print_loading_data_screen(width, height);
        state.raw_cache = Some(read_all_vendor_cached_snapshot_for_hosts(
            state.host.as_deref(),
            state.local_host_id.as_deref(),
        ));
        let mut sync_worker = match &sync_config {
            sync::config::SyncConfig::Enabled(config) => Some(sync::worker::SyncWorker::spawn(
                data::cache::default_cache_dir(),
                config.clone(),
            )),
            sync::config::SyncConfig::Disabled => None,
        };
        let mut observed_sync_revision = 0_u64;
        let mut prompt_notice = Some(data_loaded_notice(load_started.elapsed()));
        start_background_raw_refresh(&mut state);

        // Helper: disable raw mode, render the next frame off-screen into a
        // buffer, then write it atomically with per-line clears so the new
        // content fully overwrites the previous frame (no residue, no flash).
        let refresh_display = |state: &mut AppState| -> Option<bool> {
            crossterm::terminal::disable_raw_mode().ok();
            let mut result: Option<Option<bool>> = None;
            let captured = render_frame(|| {
                result = Some(print_stats(state, false));
            });
            if let Err(err) = captured {
                eprintln!("render_frame failed: {err}. Falling back to direct draw.");
                // Fall back so the user still sees something even if the
                // pipe-capture path errors out for some reason.
                result = Some(print_stats(state, false));
            }
            crossterm::terminal::enable_raw_mode().ok();
            result.flatten()
        };

        // Enable raw mode for non-blocking input
        crossterm::terminal::enable_raw_mode().ok();

        let result = refresh_display(&mut state);
        let mut terminal_too_small = result.is_none();
        let mut last_size = get_terminal_size();

        let mut input = InputLine::new();
        let mut history = CommandHistory::new();
        let mut pending_events = VecDeque::new();

        // Auto-refresh deadline kept behind a `Cell` so the prompt-rendering
        // closures below can read the live value when drawing the watermark's
        // "refresh in HH:MM:SS" countdown, while the monitor loop keeps moving
        // the deadline forward after each refresh.
        let next_refresh = std::cell::Cell::new(
            std::time::Instant::now() + std::time::Duration::from_secs(state.monitor_interval),
        );
        let refresh_remaining = || {
            Some(
                next_refresh
                    .get()
                    .saturating_duration_since(std::time::Instant::now()),
            )
        };

        let show_prompt = |state: &mut AppState,
                           input: &InputLine,
                           too_small: bool,
                           notice: Option<&PromptNotice>,
                           sync_status: Option<String>,
                           mode: PromptBlockMode| {
            render_monitor_prompt_block(
                state,
                input,
                too_small,
                notice,
                sync_status.as_deref(),
                mode,
                refresh_remaining(),
            )
        };

        let mut prompt_block_visible = show_prompt(
            &mut state,
            &input,
            terminal_too_small,
            prompt_notice.as_ref(),
            current_sync_status(sync_worker.as_ref()),
            PromptBlockMode::Append,
        );

        let mut next_sync = std::time::Instant::now()
            + monitor_sync_interval(std::time::Duration::from_secs(state.monitor_interval));
        let mut next_auto_update = args.auto_update.then(std::time::Instant::now);

        // Redraw the prompt line in place: clears it, reprints "> {buf}",
        // restores the dimmed watermark when empty, and finally moves the
        // terminal cursor to match `input`'s logical position so left/right
        // arrows feel like a real shell.
        let render_input = |input: &InputLine,
                            too_small: bool,
                            notice: Option<&PromptNotice>,
                            integrity_status: IntegrityStatus| {
            render_prompt_line(
                input,
                too_small,
                notice,
                integrity_status,
                refresh_remaining(),
            );
        };

        let cleanup_and_break = |msg: &str| {
            crossterm::terminal::disable_raw_mode().ok();
            let (width, _) = get_terminal_size();
            println!("\n\r{}\r", "-".repeat(width as usize));
            println!("\r{}", msg);
        };

        'monitor: loop {
            if poll_background_raw_refresh(&mut state) {
                if let Some(worker) = sync_worker.as_ref() {
                    worker.request_sync();
                    next_sync = std::time::Instant::now()
                        + monitor_sync_interval(std::time::Duration::from_secs(
                            state.monitor_interval,
                        ));
                }
                let result = refresh_display(&mut state);
                terminal_too_small = result.is_none();
                prompt_block_visible = show_prompt(
                    &mut state,
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    current_sync_status(sync_worker.as_ref()),
                    PromptBlockMode::Append,
                );
                render_input(
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    state.integrity_status,
                );
            }

            if let Some(sync_stats) =
                poll_sync_worker_status(sync_worker.as_ref(), &mut observed_sync_revision)
            {
                let mut redrew_display = false;
                if sync_stats.running {
                    state.integrity_status = IntegrityStatus::Checking;
                    state
                        .integrity_started_at
                        .get_or_insert_with(std::time::Instant::now);
                }
                if let Some(verification) = sync_stats.integrity_verification.as_ref() {
                    let duration = state
                        .integrity_started_at
                        .take()
                        .map(|started_at| started_at.elapsed())
                        .unwrap_or_default();
                    state.integrity_status =
                        integrity_status_from_verification(verification, duration);
                }
                if !sync_stats.running
                    && sync_stats.last_error.is_none()
                    && sync_stats.success_count > 0
                {
                    state.raw_cache = Some(read_all_vendor_cached_snapshot_for_hosts(
                        state.host.as_deref(),
                        state.local_host_id.as_deref(),
                    ));
                    let result = refresh_display(&mut state);
                    terminal_too_small = result.is_none();
                    redrew_display = true;
                }
                let mode = if redrew_display || !prompt_block_visible {
                    PromptBlockMode::Append
                } else {
                    PromptBlockMode::Replace
                };
                prompt_block_visible = show_prompt(
                    &mut state,
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    format_monitor_sync_status(&sync_stats),
                    mode,
                );
                render_input(
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    state.integrity_status,
                );
            }

            if prompt_notice_expired(&prompt_notice, std::time::Instant::now()) {
                prompt_notice = None;
                render_input(
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    state.integrity_status,
                );
            }

            if let Some(deadline) = next_auto_update
                && std::time::Instant::now() >= deadline
            {
                crossterm::terminal::disable_raw_mode().ok();
                println!("\r");
                let result = updater::run_update(|message| {
                    println!("auto-update: {message}\r");
                });
                match result {
                    Ok(updater::UpdateOutcome::AlreadyLatest { current, latest }) => {
                        println!(
                            "auto-update: already on latest version: v{current} (remote: v{latest}).\r"
                        );
                    }
                    Err(err) => {
                        println!("auto-update: update failed: {err}\r");
                    }
                }
                crossterm::terminal::enable_raw_mode().ok();
                next_auto_update = Some(auto_update_deadline_after(
                    std::time::Instant::now(),
                    args.auto_update_interval_seconds,
                ));
                prompt_block_visible = show_prompt(
                    &mut state,
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    current_sync_status(sync_worker.as_ref()),
                    PromptBlockMode::Append,
                );
                render_input(
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    state.integrity_status,
                );
            }

            // Check terminal resize
            let current_size = get_terminal_size();
            if current_size != last_size {
                last_size = current_size;
                if !terminal_too_small {
                    let (width, _) = current_size;
                    println!("\r{}\r", " ".repeat(width as usize + 2));
                    println!("{}\r", "-".repeat(width as usize));
                    println!("\n\r{}\r", "=".repeat(width as usize));
                    println!(
                        "TERMINAL RESIZED (width: {}, height: {})\r",
                        current_size.0, current_size.1
                    );
                    println!("{}\n\r", "=".repeat(width as usize));
                }
                let result = refresh_display(&mut state);
                terminal_too_small = result.is_none();
                next_refresh.set(
                    std::time::Instant::now()
                        + std::time::Duration::from_secs(state.monitor_interval),
                );
                next_sync = std::time::Instant::now()
                    + monitor_sync_interval(std::time::Duration::from_secs(state.monitor_interval));
                prompt_block_visible = show_prompt(
                    &mut state,
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    current_sync_status(sync_worker.as_ref()),
                    PromptBlockMode::Append,
                );
            }

            // Check auto-refresh
            if std::time::Instant::now() >= next_refresh.get() {
                if !terminal_too_small {
                    let (width, _) = get_terminal_size();
                    println!("\r{}\r", " ".repeat(width as usize + 2));
                    println!("{}\r", "-".repeat(width as usize));
                    println!("\n\r{}\r", "=".repeat(width as usize));
                    println!("AUTO-REFRESH\r");
                    println!("{}\n\r", "=".repeat(width as usize));
                }
                start_background_raw_refresh(&mut state);
                let result = refresh_display(&mut state);
                terminal_too_small = result.is_none();
                next_refresh.set(
                    std::time::Instant::now()
                        + std::time::Duration::from_secs(state.monitor_interval),
                );
                prompt_block_visible = show_prompt(
                    &mut state,
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    current_sync_status(sync_worker.as_ref()),
                    PromptBlockMode::Append,
                );
            }

            if let Some(worker) = sync_worker.as_ref()
                && std::time::Instant::now() >= next_sync
            {
                worker.request_sync();
                next_sync = std::time::Instant::now()
                    + monitor_sync_interval(std::time::Duration::from_secs(state.monitor_interval));
                prompt_block_visible = show_prompt(
                    &mut state,
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    current_sync_status(sync_worker.as_ref()),
                    if prompt_block_visible {
                        PromptBlockMode::Replace
                    } else {
                        PromptBlockMode::Append
                    },
                );
            }

            // Poll for input with 1s timeout
            let mut timeout = std::time::Duration::from_secs(1).min(
                next_refresh
                    .get()
                    .saturating_duration_since(std::time::Instant::now()),
            );
            if sync_worker.is_some() {
                timeout =
                    timeout.min(next_sync.saturating_duration_since(std::time::Instant::now()));
            }
            if let Some(deadline) = next_auto_update {
                timeout =
                    timeout.min(deadline.saturating_duration_since(std::time::Instant::now()));
            }
            let next_event = if let Some(event) = pending_events.pop_front() {
                Some(event)
            } else if crossterm::event::poll(timeout).unwrap_or(false) {
                crossterm::event::read().ok()
            } else {
                None
            };
            if let Some(event) = next_event {
                match event {
                    Event::Key(KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers,
                        ..
                    }) if modifiers.contains(KeyModifiers::CONTROL) => {
                        cleanup_and_break("Monitoring stopped.");
                        break 'monitor;
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Char('d'),
                        modifiers,
                        ..
                    }) if modifiers.contains(KeyModifiers::CONTROL) => {
                        cleanup_and_break("Exiting monitor mode...");
                        break 'monitor;
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Enter,
                        ..
                    }) => {
                        prompt_notice = None;
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
                                start_background_raw_refresh(&mut state);
                                let result = refresh_display(&mut state);
                                terminal_too_small = result.is_none();
                                did_refresh = true;
                            }
                            "n" => {
                                let rotation = ["all", "claude", "codex", "gemini", "omp"];
                                let idx = rotation
                                    .iter()
                                    .position(|&v| v == state.vendor)
                                    .unwrap_or(0);
                                let mut new_vendor = rotation[(idx + 1) % rotation.len()];
                                // Validate directory; skip missing vendors
                                for _ in 0..rotation.len() {
                                    if let Some(dir) = get_vendor_data_dir(new_vendor)
                                        && !dir.exists()
                                    {
                                        println!("Skipping {} (no data dir)...\r", new_vendor);
                                        let skip_idx = rotation
                                            .iter()
                                            .position(|&v| v == new_vendor)
                                            .unwrap_or(0);
                                        new_vendor = rotation[(skip_idx + 1) % rotation.len()];
                                        continue;
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
                                println!(
                                    "  v, vendor [X]    - Switch vendor (claude|codex|gemini|omp|all)\r"
                                );
                                println!("  host [X]         - Switch host (all or machine id)\r");
                                println!("  n                - Rotate to next vendor\r");
                                println!("  a                - Jump to vendor=all\r");
                                println!("  d, day, days [N] - Change days (default: 1 if no N)\r");
                                println!("  w, week          - Week mode (7 days)\r");
                                println!("  m, month         - Month mode (30 days)\r");
                                println!("  date YYYY-MM-DD  - Show one complete local day\r");
                                println!(
                                    "  range A B        - Show inclusive local date span (any order)\r"
                                );
                                println!("  latest           - Return to rolling days window\r");
                                println!(
                                    "  i, interval <N>  - Change refresh interval (seconds)\r"
                                );
                                println!(
                                    "  PgUp / PgDn      - Slide the time window back / forward by its width\r"
                                );
                                println!(
                                    "                     (PgDn snaps to the present once you reach it)\r"
                                );
                                println!(
                                    "  Left / Right     - Empty prompt: newer / older by interval; text: move cursor\r"
                                );
                                println!(
                                    "  + / -            - Empty prompt: zoom the time window in / out\r"
                                );
                                println!(
                                    "  update           - Download the latest GitHub release and restart\r"
                                );
                                println!("  e, exit          - Exit monitor mode\r");
                                println!("{}\r", "-".repeat(width as usize));
                                println!(
                                    "Current: vendor={}, host={}, window={}, interval={}s\r",
                                    state.vendor,
                                    host_label(state.host.as_deref()),
                                    state.time_window.display_label(Local::now()),
                                    state.monitor_interval
                                );
                            }
                            "host" => {
                                let known_hosts = known_host_ids(state.local_host_id.as_deref());
                                println!("Current host: {}\r", host_label(state.host.as_deref()));
                                println!("Usage: host [all|HOST]\r");
                                if !known_hosts.is_empty() {
                                    println!("Known hosts: {}\r", known_hosts.join(", "));
                                }
                            }
                            "e" | "exit" => {
                                cleanup_and_break("Exiting monitor mode...");
                                break 'monitor;
                            }
                            "update" | "upgrade" => {
                                crossterm::terminal::disable_raw_mode().ok();
                                println!("\r");
                                let result = updater::run_update(|msg| {
                                    println!("{msg}\r");
                                });
                                match result {
                                    Ok(updater::UpdateOutcome::AlreadyLatest {
                                        current,
                                        latest,
                                    }) => {
                                        println!(
                                            "Already on latest version: v{current} (remote: v{latest}).\r"
                                        );
                                    }
                                    Err(e) => {
                                        println!("Update failed: {e}\r");
                                    }
                                }
                                crossterm::terminal::enable_raw_mode().ok();
                            }
                            _ => {
                                if let Some(parsed) =
                                    parse_time_window_command(&command, state.days)
                                {
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
                                            if ["claude", "codex", "gemini", "omp", "all"]
                                                .contains(&nv)
                                            {
                                                // Validate directory before switching
                                                if let Some(dir) = get_vendor_data_dir(nv)
                                                    && !dir.exists()
                                                {
                                                    println!(
                                                        "Error: Data directory not found at {}\r",
                                                        dir.display()
                                                    );
                                                    prompt_block_visible = show_prompt(
                                                        &mut state,
                                                        &input,
                                                        terminal_too_small,
                                                        prompt_notice.as_ref(),
                                                        current_sync_status(sync_worker.as_ref()),
                                                        PromptBlockMode::Append,
                                                    );
                                                    continue 'monitor;
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
                                                println!(
                                                    "Usage: v, vendor [claude|codex|gemini|omp|all]\r"
                                                );
                                            }
                                        }
                                        "host" if parts.len() == 2 => {
                                            match parse_host_selection(parts[1]) {
                                                Ok(new_host) => {
                                                    if state.host == new_host {
                                                        println!(
                                                            "Already showing host {}.\r",
                                                            host_label(state.host.as_deref())
                                                        );
                                                    } else {
                                                        state.host = new_host;
                                                        state.raw_cache = None;
                                                        state.raw_refresh = None;
                                                        println!(
                                                            "{}\r",
                                                            "-".repeat(width as usize)
                                                        );
                                                        println!(
                                                            "\n\r{}\r",
                                                            "=".repeat(width as usize)
                                                        );
                                                        println!(
                                                            "SWITCHED TO HOST {}\r",
                                                            host_label(state.host.as_deref())
                                                        );
                                                        println!(
                                                            "{}\n\r",
                                                            "=".repeat(width as usize)
                                                        );
                                                        start_background_raw_refresh(&mut state);
                                                        let result = refresh_display(&mut state);
                                                        terminal_too_small = result.is_none();
                                                        did_refresh = true;
                                                    }
                                                }
                                                Err(err) => println!("{err}\r"),
                                            }
                                        }
                                        "d" | "day" | "days" if parts.len() == 2 => {
                                            if let Ok(n) = parts[1].parse::<i64>() {
                                                if n >= 1 {
                                                    state.days = n;
                                                    state.time_window = TimeWindow::rolling_days(n);
                                                    println!("{}\r", "-".repeat(width as usize));
                                                    println!(
                                                        "\n\r{}\r",
                                                        "=".repeat(width as usize)
                                                    );
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
                                                    let (refresh_at, sync_at) =
                                                        monitor_deadlines_after_interval_change(
                                                            std::time::Instant::now(),
                                                            n,
                                                        );
                                                    next_refresh.set(refresh_at);
                                                    next_sync = sync_at;
                                                    println!(
                                                        "Refresh interval changed to {} seconds.\r",
                                                        n
                                                    );
                                                } else {
                                                    println!(
                                                        "Interval must be at least 1 second.\r"
                                                    );
                                                }
                                            } else {
                                                println!("Invalid interval value.\r");
                                            }
                                        }
                                        "v" | "vendor" => {
                                            println!("Current vendor: {}\r", state.vendor);
                                            println!(
                                                "Usage: v, vendor [claude|codex|gemini|omp|all]\r"
                                            );
                                        }
                                        "host" => {
                                            let known_hosts =
                                                known_host_ids(state.local_host_id.as_deref());
                                            println!(
                                                "Current host: {}\r",
                                                host_label(state.host.as_deref())
                                            );
                                            println!("Usage: host [all|HOST]\r");
                                            if !known_hosts.is_empty() {
                                                println!(
                                                    "Known hosts: {}\r",
                                                    known_hosts.join(", ")
                                                );
                                            }
                                        }
                                        "i" | "interval" => {
                                            println!(
                                                "Current interval: {} seconds\r",
                                                state.monitor_interval
                                            );
                                            println!("Usage: i <N> or interval <N>\r");
                                        }
                                        _ => {
                                            println!(
                                                "Unknown command: '{}'. Type h for help.\r",
                                                command
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        if did_refresh {
                            next_refresh.set(
                                std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval),
                            );
                        }
                        prompt_block_visible = show_prompt(
                            &mut state,
                            &input,
                            terminal_too_small,
                            prompt_notice.as_ref(),
                            current_sync_status(sync_worker.as_ref()),
                            PromptBlockMode::Append,
                        );
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Up, ..
                    }) => {
                        if let Some(recalled) = history.navigate_up(input.snapshot()) {
                            prompt_notice = None;
                            input.replace(recalled);
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Down,
                        ..
                    }) => {
                        if let Some(recalled) = history.navigate_down() {
                            prompt_notice = None;
                            input.replace(recalled);
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::PageUp,
                        ..
                    }) => {
                        let now = Local::now();
                        if let Some(new_window) = state.time_window.slide_back(now) {
                            state.time_window = new_window;
                            let result = refresh_display(&mut state);
                            terminal_too_small = result.is_none();
                            next_refresh.set(
                                std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval),
                            );
                            prompt_block_visible = show_prompt(
                                &mut state,
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                current_sync_status(sync_worker.as_ref()),
                                PromptBlockMode::Append,
                            );
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::PageDown,
                        ..
                    }) => {
                        let now = Local::now();
                        if let Some(new_window) = state.time_window.slide_forward(now) {
                            state.time_window = new_window;
                            let result = refresh_display(&mut state);
                            terminal_too_small = result.is_none();
                            next_refresh.set(
                                std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval),
                            );
                            prompt_block_visible = show_prompt(
                                &mut state,
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                current_sync_status(sync_worker.as_ref()),
                                PromptBlockMode::Append,
                            );
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Left,
                        ..
                    }) => {
                        if input.is_empty() {
                            let now = Local::now();
                            let directions = collect_interval_slide_directions(
                                &mut pending_events,
                                IntervalSlideDirection::Newer,
                            );
                            if let Some(new_window) = apply_interval_slide_directions(
                                &state.time_window,
                                now,
                                get_chart_target_width(),
                                directions,
                            ) {
                                state.time_window = new_window;
                                let result = refresh_display(&mut state);
                                terminal_too_small = result.is_none();
                                next_refresh.set(
                                    std::time::Instant::now()
                                        + std::time::Duration::from_secs(state.monitor_interval),
                                );
                                prompt_block_visible = show_prompt(
                                    &mut state,
                                    &input,
                                    terminal_too_small,
                                    prompt_notice.as_ref(),
                                    current_sync_status(sync_worker.as_ref()),
                                    PromptBlockMode::Append,
                                );
                                render_input(
                                    &input,
                                    terminal_too_small,
                                    prompt_notice.as_ref(),
                                    state.integrity_status,
                                );
                            }
                        } else if input.move_left() {
                            print!("\x1b[D");
                            io::stdout().flush().unwrap();
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Right,
                        ..
                    }) => {
                        if input.is_empty() {
                            let now = Local::now();
                            let directions = collect_interval_slide_directions(
                                &mut pending_events,
                                IntervalSlideDirection::Older,
                            );
                            if let Some(new_window) = apply_interval_slide_directions(
                                &state.time_window,
                                now,
                                get_chart_target_width(),
                                directions,
                            ) {
                                state.time_window = new_window;
                                let result = refresh_display(&mut state);
                                terminal_too_small = result.is_none();
                                next_refresh.set(
                                    std::time::Instant::now()
                                        + std::time::Duration::from_secs(state.monitor_interval),
                                );
                                prompt_block_visible = show_prompt(
                                    &mut state,
                                    &input,
                                    terminal_too_small,
                                    prompt_notice.as_ref(),
                                    current_sync_status(sync_worker.as_ref()),
                                    PromptBlockMode::Append,
                                );
                                render_input(
                                    &input,
                                    terminal_too_small,
                                    prompt_notice.as_ref(),
                                    state.integrity_status,
                                );
                            }
                        } else if input.move_right() {
                            print!("\x1b[C");
                            io::stdout().flush().unwrap();
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Backspace,
                        ..
                    }) => {
                        if input.backspace() {
                            prompt_notice = None;
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Char('+'),
                        modifiers,
                        ..
                    }) if input.is_empty() && !modifiers.contains(KeyModifiers::CONTROL) => {
                        let now = Local::now();
                        if let Some(new_window) = state.time_window.zoom_in(now) {
                            state.time_window = new_window;
                            let result = refresh_display(&mut state);
                            terminal_too_small = result.is_none();
                            next_refresh.set(
                                std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval),
                            );
                            prompt_block_visible = show_prompt(
                                &mut state,
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                current_sync_status(sync_worker.as_ref()),
                                PromptBlockMode::Append,
                            );
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Char('-'),
                        modifiers,
                        ..
                    }) if input.is_empty() && !modifiers.contains(KeyModifiers::CONTROL) => {
                        let now = Local::now();
                        if let Some(new_window) = state.time_window.zoom_out(now) {
                            state.time_window = new_window;
                            let result = refresh_display(&mut state);
                            terminal_too_small = result.is_none();
                            next_refresh.set(
                                std::time::Instant::now()
                                    + std::time::Duration::from_secs(state.monitor_interval),
                            );
                            prompt_block_visible = show_prompt(
                                &mut state,
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                current_sync_status(sync_worker.as_ref()),
                                PromptBlockMode::Append,
                            );
                            render_input(
                                &input,
                                terminal_too_small,
                                prompt_notice.as_ref(),
                                state.integrity_status,
                            );
                        }
                    }
                    Event::Key(KeyEvent {
                        code: KeyCode::Char(c),
                        modifiers,
                        ..
                    }) if !modifiers.contains(KeyModifiers::CONTROL) => {
                        prompt_notice = None;
                        input.insert_char(c);
                        render_input(
                            &input,
                            terminal_too_small,
                            prompt_notice.as_ref(),
                            state.integrity_status,
                        );
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
                        next_refresh.set(
                            std::time::Instant::now()
                                + std::time::Duration::from_secs(state.monitor_interval),
                        );
                        prompt_block_visible = show_prompt(
                            &mut state,
                            &input,
                            terminal_too_small,
                            prompt_notice.as_ref(),
                            current_sync_status(sync_worker.as_ref()),
                            PromptBlockMode::Append,
                        );
                    }
                    _ => {}
                }
            } else if !terminal_too_small
                && prompt_block_visible
                && input.is_empty()
                && prompt_notice.is_none()
            {
                // Idle poll timeout (~1s elapsed with no input): redraw the
                // prompt line in place so the watermark's "refresh in HH:MM:SS"
                // countdown ticks down toward the next auto-refresh.
                render_input(
                    &input,
                    terminal_too_small,
                    prompt_notice.as_ref(),
                    state.integrity_status,
                );
            }
        }
        crossterm::terminal::disable_raw_mode().ok();
        if let Some(worker) = sync_worker.as_mut() {
            worker.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn date_command_selects_single_inclusive_day() {
        let command = parse_time_window_command("date 2026-05-07", 3)
            .expect("recognized command")
            .expect("valid date");
        let TimeWindow::ExplicitRange {
            start,
            end,
            projection_days,
            ..
        } = command
        else {
            panic!("date command should create an explicit window");
        };

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-07 00:00:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-07 23:59:59"
        );
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
    fn horizon_prefetches_multiple_page_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        assert!(compute_required_horizon(&window, now) >= 29);
    }

    #[test]
    fn local_version_label_does_not_include_remote_update_state() {
        assert_eq!(
            format_local_version_label("Codex", Some("1.2.3".to_string())),
            "Codex (1.2.3)"
        );
        assert_eq!(format_local_version_label("Codex", None), "Codex");
    }

    #[test]
    fn data_loaded_notice_formats_milliseconds_and_expires() {
        let notice = data_loaded_notice(std::time::Duration::from_millis(42));

        assert_eq!(notice.message, "Data loaded in 42 ms");
        assert!(prompt_notice_expired(
            &Some(notice),
            std::time::Instant::now() + std::time::Duration::from_secs(4)
        ));
    }

    #[test]
    fn integrity_status_marker_uses_expected_text_width_and_color() {
        let (checking, checking_visible) = integrity_status_marker(IntegrityStatus::Checking);
        assert!(checking.contains("\x1b[38;5;143mIntegrity Checking\x1b[0m"));
        assert_eq!(checking_visible, "Integrity Checking".chars().count());

        let (checked_ms, checked_ms_visible) = integrity_status_marker(IntegrityStatus::Checked {
            duration: std::time::Duration::from_millis(842),
        });
        assert!(checked_ms.contains("\x1b[38;5;108mIntegrity Checked in 842 ms\x1b[0m"));
        assert_eq!(
            checked_ms_visible,
            "Integrity Checked in 842 ms".chars().count()
        );

        let (checked_s, checked_s_visible) = integrity_status_marker(IntegrityStatus::Checked {
            duration: std::time::Duration::from_millis(1_234),
        });
        assert!(checked_s.contains("\x1b[38;5;108mIntegrity Checked in 1.23 s\x1b[0m"));
        assert_eq!(
            checked_s_visible,
            "Integrity Checked in 1.23 s".chars().count()
        );

        let (failed, failed_visible) = integrity_status_marker(IntegrityStatus::Failed);
        assert!(failed.contains("\x1b[38;5;203mIntegrity Failed\x1b[0m"));
        assert_eq!(failed_visible, "Integrity Failed".chars().count());
    }

    #[test]
    fn sync_integrity_verification_maps_to_prompt_status() {
        let duration = std::time::Duration::from_millis(12);
        assert_eq!(
            integrity_status_from_verification(
                &sync::integrity::IntegrityVerification::Checked { checked_hosts: 2 },
                duration,
            ),
            IntegrityStatus::Checked { duration }
        );

        assert_eq!(
            integrity_status_from_verification(
                &sync::integrity::IntegrityVerification::Failed {
                    failures: vec![sync::integrity::IntegrityFailure {
                        host_id: "laptop".to_string(),
                        range_end_utc: "2026-06-01T00:00:00Z".to_string(),
                        expected_record_count: 1,
                        actual_record_count: 0,
                        expected_digest_sha256: "a".repeat(64),
                        actual_digest_sha256: "b".repeat(64),
                    }],
                },
                duration,
            ),
            IntegrityStatus::Failed
        );
    }

    #[test]
    fn sync_interval_is_one_third_of_monitor_interval_with_minimum() {
        assert_eq!(
            monitor_sync_interval(std::time::Duration::from_secs(3600)),
            std::time::Duration::from_secs(1200)
        );
        assert_eq!(
            monitor_sync_interval(std::time::Duration::from_secs(180)),
            std::time::Duration::from_secs(60)
        );
        assert_eq!(
            monitor_sync_interval(std::time::Duration::from_secs(30)),
            std::time::Duration::from_secs(60)
        );
    }

    #[test]
    fn interval_change_reschedules_refresh_and_sync_deadlines() {
        let now = std::time::Instant::now();

        let (next_refresh, next_sync) = monitor_deadlines_after_interval_change(now, 3600);

        assert_eq!(next_refresh, now + std::time::Duration::from_secs(3600));
        assert_eq!(next_sync, now + std::time::Duration::from_secs(1200));
    }

    #[test]
    fn manual_sync_progress_formats_push_pull_and_retry_events() {
        assert_eq!(
            format_manual_sync_progress(&sync::engine::SyncProgress::UploadPlanned {
                total_records: 1001,
                total_batches: 2,
                skipped_records: 7,
            }),
            Some("sync push: 1001 records to upload in 2 batches, 7 already synced".to_string())
        );
        assert_eq!(
            format_manual_sync_progress(&sync::engine::SyncProgress::UploadBatchFinished {
                batch_index: 1,
                total_batches: 2,
                uploaded_records: 1000,
                total_records: 1001,
                accepted: 997,
                ignored: 3,
            }),
            Some(
                "sync push: [##########----------] 1/2 batches, 1000/1001 records, accepted 997, ignored 3"
                    .to_string()
            )
        );
        assert_eq!(
            format_manual_sync_progress(&sync::engine::SyncProgress::PullPageFinished {
                page_index: 2,
                page_records: 5000,
                pulled_records: 10000,
                max_seq: 123,
                truncated: true,
            }),
            Some("sync pull: page 2, 10000 records pulled, latest seq 123".to_string())
        );
        assert_eq!(
            format_http_progress(&sync::client::HttpProgress::RateLimited {
                attempt: 2,
                retry_after: std::time::Duration::from_millis(1100),
            }),
            "sync: rate limited, retrying in 1.1s (attempt 2)"
        );
    }

    #[test]
    fn monitor_sync_status_formats_running_progress_and_completion() {
        let running = sync::worker::SyncStats {
            running: true,
            progress: Some(sync::worker::SyncWorkerProgress::Sync(
                sync::engine::SyncProgress::PullPageFinished {
                    page_index: 2,
                    page_records: 5000,
                    pulled_records: 10000,
                    max_seq: 123,
                    truncated: true,
                },
            )),
            ..sync::worker::SyncStats::default()
        };
        assert_eq!(
            format_monitor_sync_status_at(
                &running,
                chrono::DateTime::parse_from_rfc3339("2026-05-19T12:00:00Z")
                    .expect("timestamp")
                    .with_timezone(&chrono::Utc),
            ),
            Some("Sync: pull page 2, 10000 records pulled, latest seq 123".to_string())
        );

        let completed = sync::worker::SyncStats {
            success_count: 1,
            last_finished_at: Some("2026-05-19T11:42:00Z".to_string()),
            ..sync::worker::SyncStats::default()
        };
        assert_eq!(
            format_monitor_sync_status_at(
                &completed,
                chrono::DateTime::parse_from_rfc3339("2026-05-19T12:00:00Z")
                    .expect("timestamp")
                    .with_timezone(&chrono::Utc),
            ),
            Some("Sync: checked 18 min ago".to_string())
        );
    }

    #[test]
    fn window_data_state_distinguishes_empty_window_from_missing_source_data() {
        assert_eq!(
            classify_window_data(false, false),
            WindowDataState::NoSourceData
        );
        assert_eq!(
            classify_window_data(true, false),
            WindowDataState::EmptyWindow
        );
        assert_eq!(classify_window_data(true, true), WindowDataState::Populated);
    }

    #[test]
    fn interval_slide_older_moves_rolling_window_by_display_interval() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        let slid =
            slide_window_by_display_interval(&window, now, 160, IntervalSlideDirection::Older)
                .expect("slide older");
        let (start, end) = slid.bounds(now);

        assert_eq!(end, now - chrono::Duration::hours(1));
        assert_eq!(
            start,
            now - chrono::Duration::days(3) - chrono::Duration::hours(1)
        );
        assert_eq!(slid.page_step(), chrono::Duration::days(3));
    }

    #[test]
    fn interval_slide_newer_clamps_to_present() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);
        let older =
            slide_window_by_display_interval(&window, now, 160, IntervalSlideDirection::Older)
                .expect("slide older");

        let newer =
            slide_window_by_display_interval(&older, now, 160, IntervalSlideDirection::Newer)
                .expect("slide newer");
        let (start, end) = newer.bounds(now);

        assert_eq!(end, now);
        assert_eq!(start, now - chrono::Duration::days(3));
    }

    #[test]
    fn interval_slide_newer_on_current_window_is_noop() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        assert!(
            slide_window_by_display_interval(&window, now, 160, IntervalSlideDirection::Newer)
                .is_none()
        );
    }

    #[test]
    fn batched_interval_slide_matches_repeated_right_arrow_slides() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);
        let target_width = 160;
        let directions = [IntervalSlideDirection::Older; 5];

        let batched = apply_interval_slide_directions(&window, now, target_width, directions)
            .expect("batched slide");
        let mut sequential = window.clone();
        for _ in 0..directions.len() {
            sequential = slide_window_by_display_interval(
                &sequential,
                now,
                target_width,
                IntervalSlideDirection::Older,
            )
            .expect("sequential slide");
        }

        assert_eq!(batched.bounds(now), sequential.bounds(now));
        assert_eq!(batched.page_step(), sequential.page_step());
    }

    #[test]
    fn batched_interval_slide_preserves_noop_then_right_arrow_slide() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);
        let target_width = 160;
        let directions = [IntervalSlideDirection::Newer, IntervalSlideDirection::Older];

        let batched = apply_interval_slide_directions(&window, now, target_width, directions)
            .expect("batched slide");
        let expected = slide_window_by_display_interval(
            &window,
            now,
            target_width,
            IntervalSlideDirection::Older,
        )
        .expect("right arrow slide");

        assert_eq!(batched.bounds(now), expected.bounds(now));
        assert_eq!(batched.page_step(), expected.page_step());
    }

    #[test]
    fn display_interval_scales_past_daily_for_multi_month_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2025-11-03", "2026-05-14").expect("range");

        let interval = display_interval_minutes_for_window(&window, now, 160);

        assert!(interval > 1440);
        assert_eq!(interval, 2880);
    }

    #[test]
    fn display_interval_uses_month_granularity_for_quarter_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2026-02-14", "2026-05-14").expect("range");
        let (range_start, range_end) = window.bounds(now);

        let granularity = display_chart_granularity(&range_start, &range_end);
        let interval = display_interval_minutes_for_window(&window, now, 160);

        assert_eq!(granularity, charts::ChartGranularity::Month);
        assert_eq!(interval, 1440);
    }

    #[test]
    fn display_interval_scales_past_daily_for_year_granularity_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2024-01-01", "2026-05-14").expect("range");

        let interval = display_interval_minutes_for_window(&window, now, 160);

        assert!(interval > 1440);
        assert_eq!(interval, 20160);
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

    #[test]
    fn sync_subcommands_parse() {
        let push = Args::try_parse_from(["vibe-usage", "sync", "push"]).expect("push parses");
        assert!(matches!(
            push.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Push
            })
        ));

        let pull = Args::try_parse_from(["vibe-usage", "sync", "pull"]).expect("pull parses");
        assert!(matches!(
            pull.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Pull
            })
        ));

        let status = Args::try_parse_from(["vibe-usage", "sync", "status"]).expect("status parses");
        assert!(matches!(
            status.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Status
            })
        ));

        let init =
            Args::try_parse_from(["vibe-usage", "sync", "init", "--force"]).expect("init parses");
        assert!(matches!(
            init.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Init { force: true }
            })
        ));

        let clean = Args::try_parse_from(["vibe-usage", "sync", "clean"]).expect("clean parses");
        assert!(matches!(
            clean.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Clean
            })
        ));
    }

    #[test]
    fn host_arg_parses() {
        let args = Args::try_parse_from(["vibe-usage", "--host", "laptop"]).expect("host parses");
        assert_eq!(args.host.as_deref(), Some("laptop"));
    }

    #[test]
    fn auto_update_flag_defaults_off() {
        let args = Args::try_parse_from(["vibe-usage"]).expect("args parse");

        assert!(!args.auto_update);
        assert_eq!(args.auto_update_interval_seconds, 3600);
    }

    #[test]
    fn auto_update_flag_parses_interval() {
        let args = Args::try_parse_from([
            "vibe-usage",
            "--auto-update",
            "--auto-update-interval-seconds",
            "7200",
        ])
        .expect("args parse");

        assert!(args.auto_update);
        assert_eq!(args.auto_update_interval_seconds, 7200);
    }

    #[test]
    fn version_flag_prints_cargo_pkg_version() {
        let err = Args::try_parse_from(["vibe-usage", "--version"])
            .expect_err("--version should short-circuit parsing");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        let rendered = err.to_string();
        assert!(
            rendered.contains(env!("CARGO_PKG_VERSION")),
            "version output should contain crate version: {rendered}"
        );
    }

    #[test]
    fn monitor_status_line_places_sync_on_same_line() {
        let line = format_monitor_status_line(
            "All Vendors Comparison",
            Some("Sync: checked just now"),
            60,
        );

        assert_eq!(
            line,
            format!(
                "{}{}{}",
                "All Vendors Comparison",
                " ".repeat(16),
                "Sync: checked just now"
            )
        );
        assert_eq!(line.lines().count(), 1);
    }

    #[test]
    fn monitor_status_line_truncates_left_before_sync() {
        let line = format_monitor_status_line(
            "All Vendors Comparison",
            Some("Sync: checked just now"),
            28,
        );

        assert_eq!(line, "All V Sync: checked just now");
    }

    #[test]
    fn host_command_selection_parses_all_and_machine_ids() {
        assert_eq!(parse_host_selection("all"), Ok(None));
        assert_eq!(
            parse_host_selection("workstation-home"),
            Ok(Some("workstation-home".to_string()))
        );
        assert!(parse_host_selection("Workstation").is_err());
    }

    #[test]
    fn host_filter_includes_local_only_when_machine_matches() {
        assert!(include_local_for_host_filter(None, None));
        assert!(include_local_for_host_filter(
            Some("workstation"),
            Some("workstation")
        ));
        assert!(!include_local_for_host_filter(
            Some("laptop"),
            Some("workstation")
        ));
        assert!(!include_local_for_host_filter(Some("laptop"), None));
    }

    #[test]
    fn remote_records_merge_into_vendor_buckets() {
        let timestamp = "2026-05-18T12:00:00Z";
        let mut cache = RawDataCache {
            claude: Vec::new(),
            codex: Vec::new(),
            gemini: Vec::new(),
            omp: Vec::new(),
            horizon_days: FULL_CACHE_HORIZON,
            local_host_id: None,
            local_record_keys: HashSet::new(),
        };
        merge_remote_records_into_raw_cache(
            &mut cache,
            vec![
                data::cache::RemoteUsageRecord {
                    vendor: "claude".to_string(),
                    dedup_key: "a".to_string(),
                    entry: UsageEntry {
                        host_id: Some("laptop".to_string()),
                        timestamp: timestamp.to_string(),
                        parsed_timestamp: time_utils::parse_timestamp(timestamp),
                        session_start_time: timestamp.to_string(),
                        session_end_time: timestamp.to_string(),
                        model: "model-a".to_string(),
                        effort: None,
                        fast_tier: -1,
                        usage: data::TokenUsage::default(),
                        costs: None,
                    },
                },
                data::cache::RemoteUsageRecord {
                    vendor: "codex".to_string(),
                    dedup_key: "b".to_string(),
                    entry: UsageEntry {
                        host_id: Some("laptop".to_string()),
                        timestamp: timestamp.to_string(),
                        parsed_timestamp: time_utils::parse_timestamp(timestamp),
                        session_start_time: timestamp.to_string(),
                        session_end_time: timestamp.to_string(),
                        model: "model-b".to_string(),
                        effort: None,
                        fast_tier: -1,
                        usage: data::TokenUsage::default(),
                        costs: None,
                    },
                },
            ],
        );

        assert_eq!(cache.claude.len(), 1);
        assert_eq!(cache.codex.len(), 1);
        assert!(cache.gemini.is_empty());
    }

    #[test]
    fn remote_records_from_local_host_fill_missing_keys_without_duplicates() {
        let timestamp = "2026-05-18T12:00:00Z";
        let mut cache = RawDataCache {
            claude: vec![UsageEntry {
                host_id: None,
                timestamp: timestamp.to_string(),
                parsed_timestamp: time_utils::parse_timestamp(timestamp),
                session_start_time: timestamp.to_string(),
                session_end_time: timestamp.to_string(),
                model: "local-model".to_string(),
                effort: None,
                fast_tier: -1,
                usage: data::TokenUsage::default(),
                costs: None,
            }],
            codex: Vec::new(),
            gemini: Vec::new(),
            omp: Vec::new(),
            horizon_days: FULL_CACHE_HORIZON,
            local_host_id: Some("laptop".to_string()),
            local_record_keys: HashSet::from([("claude".to_string(), "a".to_string())]),
        };
        merge_remote_records_into_raw_cache(
            &mut cache,
            vec![
                data::cache::RemoteUsageRecord {
                    vendor: "claude".to_string(),
                    dedup_key: "a".to_string(),
                    entry: UsageEntry {
                        host_id: Some("laptop".to_string()),
                        timestamp: timestamp.to_string(),
                        parsed_timestamp: time_utils::parse_timestamp(timestamp),
                        session_start_time: timestamp.to_string(),
                        session_end_time: timestamp.to_string(),
                        model: "duplicate-model".to_string(),
                        effort: None,
                        fast_tier: -1,
                        usage: data::TokenUsage::default(),
                        costs: None,
                    },
                },
                data::cache::RemoteUsageRecord {
                    vendor: "claude".to_string(),
                    dedup_key: "b".to_string(),
                    entry: UsageEntry {
                        host_id: Some("laptop".to_string()),
                        timestamp: timestamp.to_string(),
                        parsed_timestamp: time_utils::parse_timestamp(timestamp),
                        session_start_time: timestamp.to_string(),
                        session_end_time: timestamp.to_string(),
                        model: "missing-model".to_string(),
                        effort: None,
                        fast_tier: -1,
                        usage: data::TokenUsage::default(),
                        costs: None,
                    },
                },
            ],
        );

        assert_eq!(cache.claude.len(), 2);
        assert!(
            cache
                .claude
                .iter()
                .any(|entry| entry.model == "local-model")
        );
        assert!(
            cache
                .claude
                .iter()
                .any(|entry| entry.model == "missing-model")
        );
        assert!(
            !cache
                .claude
                .iter()
                .any(|entry| entry.model == "duplicate-model")
        );
    }
}
