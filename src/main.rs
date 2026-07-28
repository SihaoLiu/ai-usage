mod charts;
mod constants;
mod data;
mod formatting;
mod model_id;
mod model_overrides;
mod pricing;
mod raw_data;
mod stats;
mod sync;
mod sync_status;
mod table_view;
mod time_utils;
mod tool;
mod tui;
mod updater;
mod window_nav;

use std::collections::{HashMap, HashSet};

use std::path::Path;
use std::process::Command as ProcessCommand;
use std::sync::mpsc;

use chrono::{DateTime, Local};
use clap::{Parser, Subcommand};
use crossterm::terminal;

use constants::{AllPricing, SubscriptionFees, load_subscription_fees, prompt_subscription_fees};
use data::UsageEntry;
use formatting::print_model_breakdown;
use table_view::TableView;
use time_utils::TimeWindow;
use tool::Tool;

pub(crate) use raw_data::*;
pub(crate) use sync_status::*;
pub(crate) use window_nav::*;

const MIN_TERMINAL_WIDTH: u16 = 60;
const MIN_TERMINAL_HEIGHT: u16 = 35;
const DAY_PRESET_DAYS: i64 = 1;
const WEEK_PRESET_DAYS: i64 = 7;
const MONTH_PRESET_DAYS: i64 = 30;
const YEAR_PRESET_DAYS: i64 = 365;

struct VersionCacheEntry {
    version_str: String,
    receiver: Option<mpsc::Receiver<String>>,
}

impl VersionCacheEntry {
    fn poll(&mut self) -> bool {
        let Some(receiver) = self.receiver.take() else {
            return false;
        };
        match receiver.try_recv() {
            Ok(version_str) => {
                let changed = self.version_str != version_str;
                self.version_str = version_str;
                changed
            }
            Err(mpsc::TryRecvError::Empty) => {
                self.receiver = Some(receiver);
                false
            }
            Err(mpsc::TryRecvError::Disconnected) => false,
        }
    }
}

struct AppState {
    tool: String,
    table_view: TableView,
    host: Option<String>,
    session_id: Option<String>,
    local_host_id: Option<String>,
    days: i64,
    time_window: TimeWindow,
    monitor_interval: u64,
    pricing: AllPricing,
    subscription_fees: SubscriptionFees,
    version_cache: HashMap<String, VersionCacheEntry>,
    all_tool_prompt: Option<String>,
    raw_cache: Option<RawDataCache>,
    raw_cache_last_used_at: Option<std::time::Instant>,
    raw_refresh: Option<mpsc::Receiver<BackgroundRawLoad>>,
    integrity_status: IntegrityStatus,
    integrity_started_at: Option<std::time::Instant>,
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

    /// Tool to collect statistics from
    #[arg(long, default_value = "all", value_parser = ["claude", "codex", "gemini", "kimi", "omp", "all"])]
    tool: String,

    /// Breakdown table shape: flat (Vendor/Model/Harness columns),
    /// vendor (grouped by vendor), or model (merged across harnesses)
    #[arg(long, default_value = "flat", value_parser = ["flat", "vendor", "model"])]
    view: String,

    /// Filter usage to a single machine id
    #[arg(long)]
    host: Option<String>,

    /// Track one conversation by its harness session id
    #[arg(long)]
    session: Option<String>,

    /// Check GitHub releases periodically in monitor mode and restart into a newer binary
    #[arg(long)]
    auto_update: bool,

    /// Seconds between automatic release checks when --auto-update is enabled
    #[arg(
        long,
        default_value_t = ai_usage_updater::DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS
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

fn host_label(host: Option<&str>) -> &str {
    host.unwrap_or("all")
}

fn needs_session_metadata(session_id: Option<&str>, cache: &RawDataCache) -> bool {
    session_id.is_some() && !cache.local_session_metadata_current
}

fn parse_host_selection(selection: &str) -> Result<Option<String>, &'static str> {
    let selection = selection.trim();
    if selection == "all" {
        return Ok(None);
    }
    if ai_usage_proto::is_valid_host_id(selection) {
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
    hosts.extend(data::cache::remote_host_ids(&cache_root));
    let mut hosts = hosts.into_iter().collect::<Vec<_>>();
    hosts.sort();
    hosts
}

fn get_version(state: &mut AppState, tool: &str) -> String {
    if let Some(cached) = state.version_cache.get_mut(tool) {
        cached.poll();
        return cached.version_str.clone();
    }

    let (cmd, display_name) = match Tool::from_key(tool) {
        Some(Tool::Claude) => ("claude", Tool::Claude.display_name()),
        Some(Tool::Codex) => ("codex", Tool::Codex.display_name()),
        Some(Tool::Gemini) => ("gemini", Tool::Gemini.display_name()),
        Some(Tool::Kimi) => ("kimi", Tool::Kimi.display_name()),
        _ => return String::new(),
    };

    let fallback = format_local_version_label(display_name, None);
    let (tx, rx) = mpsc::channel();
    state.version_cache.insert(
        tool.to_string(),
        VersionCacheEntry {
            version_str: fallback.clone(),
            receiver: Some(rx),
        },
    );
    let _ = std::thread::Builder::new()
        .name(format!("{tool}-version"))
        .spawn(move || {
            let _ = tx.send(resolve_local_version(cmd, display_name));
        });

    fallback
}

fn resolve_local_version(cmd: &str, display_name: &str) -> String {
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

    format_local_version_label(display_name, current_version)
}

fn poll_version_cache(state: &mut AppState) -> bool {
    let mut changed = false;
    for entry in state.version_cache.values_mut() {
        changed |= entry.poll();
    }
    changed
}

fn format_local_version_label(display_name: &str, current_version: Option<String>) -> String {
    match current_version {
        Some(ver) => format!("{} ({})", display_name, ver),
        None => display_name.to_string(),
    }
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
    let range = raw_cache_visible_range(&state.time_window, now);
    let _ = ensure_raw_cache(state, range);
    let cache = state.raw_cache.as_ref().expect("cache populated");
    let tool = state.tool.clone();
    let raw_for_tool: &[UsageEntry] = match tool.as_str() {
        "claude" => &cache.claude,
        "codex" => &cache.codex,
        "gemini" => &cache.gemini,
        "kimi" => &cache.kimi,
        "omp" => &cache.omp,
        _ => &cache.claude,
    };
    let filtered = data::filter_usage_data_by_window_and_session(
        raw_for_tool,
        &state.time_window,
        state.session_id.as_deref(),
        now,
    );
    if state.session_id.is_some() && filtered.is_empty() {
        println!(
            "No usage data found for session {}.",
            state.session_id.as_deref().unwrap_or_default()
        );
        return Some(false);
    }
    if classify_window_data(cache.has_source_data, !filtered.is_empty())
        == WindowDataState::NoSourceData
    {
        if !once {
            print!("\x1b[2J\x1b[H");
        }
        println!("No usage data found.");
        return Some(false);
    }
    let tool = &tool;
    let model_stats = match tool.as_str() {
        "codex" => stats::calculate_codex_model_breakdown(&filtered, &state.pricing),
        "gemini" => stats::calculate_gemini_model_breakdown(&filtered, &state.pricing),
        "kimi" => stats::calculate_kimi_model_breakdown(&filtered, &state.pricing),
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

    let tool_name = Tool::from_key(tool)
        .map(Tool::display_name)
        .unwrap_or("Claude Code");

    println!("Calculating {} usage...", tool_name);
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
        tool,
        &state.subscription_fees,
        state.table_view,
    );

    let granularity = display_chart_granularity(&range_start, &range_end);
    let optimal =
        calculate_optimal_interval_minutes(&range_start, &range_end, target_width, granularity);
    let interval_minutes = round_to_nice_interval(optimal);

    let model_ts = match tool.as_str() {
        "codex" => {
            stats::calculate_codex_model_token_breakdown_time_series(&filtered, interval_minutes)
        }
        "gemini" => {
            stats::calculate_gemini_model_token_breakdown_time_series(&filtered, interval_minutes)
        }
        "kimi" => {
            stats::calculate_kimi_model_token_breakdown_time_series(&filtered, interval_minutes)
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
        tool,
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
        tool,
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

    let all_data = load_all_tool_data(state, now);

    if state.session_id.is_some() && !all_tool_data_has_window_data(&all_data) {
        println!(
            "No usage data found for session {}.",
            state.session_id.as_deref().unwrap_or_default()
        );
        return Some(false);
    }

    let all_model_stats = calculate_all_model_breakdown(&all_data, &state.pricing);

    // Compute and cache the weighted cost prompt for show_prompt reuse
    let (weighted_cost, total_savings) = calculate_weighted_cost_per_mtok(
        &all_model_stats,
        projection_days,
        &state.subscription_fees,
    );
    state.all_tool_prompt = if weighted_cost > 0.0 {
        Some(format!(
            "All Tools Comparison, {} / MTok, Monthly Saving ${:.2}",
            formatting::format_cost_per_mtok(weighted_cost),
            total_savings,
        ))
    } else {
        None
    };

    let tool_time_series = calculate_tool_aggregate_time_series(&all_data, interval_minutes);
    let has_source_data = state
        .raw_cache
        .as_ref()
        .is_some_and(|cache| cache.has_source_data);
    let data_state =
        classify_window_data(has_source_data, all_tool_data_has_window_data(&all_data));
    if data_state == WindowDataState::NoSourceData {
        println!("No usage data found from any tool.");
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

    println!("Calculating usage across all tools...");
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
        state.table_view,
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

    charts::print_tool_comparison_chart(
        &tool_time_series,
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
    if state.tool == "all" {
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
                eprintln!("ai-usage: sync config already exists; pass --force to replace it");
                1
            }
            Err(err) => {
                eprintln!("ai-usage: sync init failed: {err}");
                1
            }
        },
        SyncCommand::Push | SyncCommand::Pull | SyncCommand::Clean => {
            let sync::config::SyncConfig::Enabled(config) = sync_config else {
                eprintln!("ai-usage: sync is disabled");
                return 1;
            };
            let cache_root = data::cache::default_cache_dir();
            let _sync_lock = match sync::lock::SyncLock::try_acquire(&cache_root) {
                Ok(Some(lock)) => lock,
                Ok(None) => {
                    eprintln!("ai-usage: another sync is already running");
                    return 1;
                }
                Err(err) => {
                    eprintln!("ai-usage: failed to acquire sync lock: {err}");
                    return 1;
                }
            };
            let client = sync::client::SyncHttpClient::new_with_progress(config.clone(), |event| {
                eprintln!("{}", format_http_progress(event));
            });
            let result = match command {
                SyncCommand::Push => {
                    eprintln!("sync push: refreshing local cache");
                    refresh_all_tool_caches();
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
                    .and_then(|outcome| {
                        if !outcome.held_back_vendors.is_empty() {
                            // Integrity digests would count the held-back
                            // records the server cannot distribute yet.
                            return Ok(());
                        }
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
                        "ai-usage: sync {} failed: {}",
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

    // Validate tool data directory on startup.
    if let Some(data_dir) = get_tool_data_dir(&args.tool)
        && !data_dir.exists()
    {
        eprintln!("Error: Data directory not found at {}", data_dir.display());
        std::process::exit(1);
    }

    let mut state = AppState {
        tool: args.tool.clone(),
        table_view: TableView::from_key(&args.view).unwrap_or_default(),
        host: args.host.clone(),
        session_id: args.session.filter(|id| !id.trim().is_empty()),
        local_host_id,
        days: args.days,
        time_window: TimeWindow::rolling_days(args.days),
        monitor_interval: 3600,
        pricing,
        subscription_fees,
        version_cache: HashMap::new(),
        all_tool_prompt: None,
        raw_cache: None,
        raw_cache_last_used_at: None,
        raw_refresh: None,
        integrity_status: initial_integrity_status(matches!(
            &sync_config,
            sync::config::SyncConfig::Enabled(_)
        )),
        integrity_started_at: None,
    };

    if args.once {
        let now = Local::now();
        let required_range = raw_cache_visible_range(&state.time_window, now);
        let mut refreshed = read_cached_raw_data_for_window(
            state.host.as_deref(),
            state.local_host_id.as_deref(),
            required_range,
            now,
        );
        // A missing cache must still bootstrap from the canonical source logs.
        // A session query does the same only when the loaded cache predates
        // session metadata, so its first run after an upgrade is accurate.
        let needs_session_metadata =
            needs_session_metadata(state.session_id.as_deref(), &refreshed);
        if !refreshed.has_source_data || needs_session_metadata {
            refresh_all_tool_caches();
            refreshed = load_persistent_raw_data_for_window(
                state.host.as_deref(),
                state.local_host_id.as_deref(),
                required_range,
                Local::now(),
            );
        }
        state.raw_cache = Some(refreshed);
        let result = print_stats(&mut state, true);
        match result {
            None => std::process::exit(1),
            Some(false) => std::process::exit(0),
            Some(true) => {}
        }
    } else {
        // Monitor mode
        let sync_worker = match &sync_config {
            sync::config::SyncConfig::Enabled(config) => Some(sync::worker::SyncWorker::spawn(
                data::cache::default_cache_dir(),
                config.clone(),
            )),
            sync::config::SyncConfig::Disabled => None,
        };
        tui::run_monitor(
            &mut state,
            sync_worker,
            tui::MonitorConfig {
                auto_update: args.auto_update,
                auto_update_interval_seconds: args.auto_update_interval_seconds,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with_session(session_id: Option<&str>) -> RawDataCache {
        let timestamp = "2026-07-23T00:00:00Z".to_string();
        let center = time_utils::parse_timestamp(&timestamp).expect("test timestamp");
        RawDataCache {
            claude: vec![UsageEntry {
                host_id: None,
                session_id: session_id.map(str::to_string),
                timestamp: timestamp.clone(),
                parsed_timestamp: None,
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model: "test-model".to_string(),
                effort: None,
                fast_tier: data::UNKNOWN_FAST_TIER,
                usage: data::TokenUsage::default(),
                costs: None,
            }],
            codex: Vec::new(),
            gemini: Vec::new(),
            kimi: Vec::new(),
            omp: Vec::new(),
            range: RawDataRange::from_bounds(
                center - chrono::Duration::days(1),
                center + chrono::Duration::days(1),
            ),
            has_source_data: true,
            local_host_id: None,
            local_record_keys: HashMap::new(),
            persistent_generation: String::new(),
            local_session_metadata_current: true,
        }
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
    fn version_cache_poll_never_waits_for_a_pending_lookup() {
        let (tx, rx) = mpsc::channel();
        let mut entry = VersionCacheEntry {
            version_str: "Codex".to_string(),
            receiver: Some(rx),
        };

        assert!(!entry.poll());
        assert_eq!(entry.version_str, "Codex");

        tx.send("Codex (1.2.3)".to_string())
            .expect("send resolved version");
        assert!(entry.poll());
        assert_eq!(entry.version_str, "Codex (1.2.3)");
        assert!(entry.receiver.is_none());
    }

    #[test]
    fn sync_subcommands_parse() {
        let push = Args::try_parse_from(["ai-usage", "sync", "push"]).expect("push parses");
        assert!(matches!(
            push.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Push
            })
        ));

        let pull = Args::try_parse_from(["ai-usage", "sync", "pull"]).expect("pull parses");
        assert!(matches!(
            pull.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Pull
            })
        ));

        let status = Args::try_parse_from(["ai-usage", "sync", "status"]).expect("status parses");
        assert!(matches!(
            status.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Status
            })
        ));

        let init =
            Args::try_parse_from(["ai-usage", "sync", "init", "--force"]).expect("init parses");
        assert!(matches!(
            init.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Init { force: true }
            })
        ));

        let clean = Args::try_parse_from(["ai-usage", "sync", "clean"]).expect("clean parses");
        assert!(matches!(
            clean.command,
            Some(CliCommand::Sync {
                command: SyncCommand::Clean
            })
        ));
    }

    #[test]
    fn host_arg_parses() {
        let args = Args::try_parse_from(["ai-usage", "--host", "laptop"]).expect("host parses");
        assert_eq!(args.host.as_deref(), Some("laptop"));
    }

    #[test]
    fn session_arg_selects_one_conversation() {
        let args = Args::try_parse_from(["ai-usage", "--session", "conversation-42"])
            .expect("session parses");
        assert_eq!(args.session.as_deref(), Some("conversation-42"));
    }

    #[test]
    fn session_query_does_not_refresh_a_cache_with_session_metadata() {
        let cache = cache_with_session(Some("another-conversation"));

        assert!(!needs_session_metadata(
            Some("missing-conversation"),
            &cache
        ));
    }

    #[test]
    fn session_query_refreshes_a_cache_with_stale_metadata() {
        let mut cache = cache_with_session(None);
        cache.local_session_metadata_current = false;

        assert!(needs_session_metadata(Some("conversation-42"), &cache));
    }

    #[test]
    fn auto_update_flag_defaults_off() {
        let args = Args::try_parse_from(["ai-usage"]).expect("args parse");

        assert!(!args.auto_update);
        assert_eq!(args.auto_update_interval_seconds, 3600);
    }

    #[test]
    fn tool_flag_selects_usage_source() {
        let args = Args::try_parse_from(["ai-usage", "--tool", "omp"]).expect("tool flag parses");

        assert_eq!(args.tool, "omp");
    }

    #[test]
    fn vendor_flag_is_not_a_cli_alias() {
        let err = Args::try_parse_from(["ai-usage", "--vendor", "omp"])
            .expect_err("vendor should not be accepted");

        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn auto_update_flag_parses_interval() {
        let args = Args::try_parse_from([
            "ai-usage",
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
        let err = Args::try_parse_from(["ai-usage", "--version"])
            .expect_err("--version should short-circuit parsing");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        let rendered = err.to_string();
        assert!(
            rendered.contains(env!("CARGO_PKG_VERSION")),
            "version output should contain crate version: {rendered}"
        );
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
}
