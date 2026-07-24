//! Raw usage-entry cache: loading, host filtering, background refresh,
//! and cross-tool aggregation.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use chrono::{DateTime, Local};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use std::path::Path;

use crate::constants::{AllPricing, SubscriptionFees};
use crate::data::claude::detect_fast_tier_snapshot as detect_claude_fast_tier_snapshot;
use crate::data::codex::{
    detect_fast_tier_snapshot as detect_codex_fast_tier_snapshot, get_codex_dir,
};
use crate::data::gemini::get_gemini_dir;
use crate::data::kimi::get_kimi_dir;
use crate::data::omp::get_omp_dir;
use crate::data::{self, UsageEntry};
use crate::stats::{self, ModelBreakdownRow, ToolTimeSeries};
use crate::sync_status::IntegrityStatus;
use crate::time_utils::{self, TimeWindow};
use crate::tool::Tool;
use crate::{AppState, FULL_CACHE_HORIZON};

/// Short rolling windows are common enough to keep a compact derived snapshot
/// beside the authoritative per-source caches.
const HOT_CACHE_HORIZON_DAYS: i64 = 8;

fn hot_snapshot_covers(
    horizon_days: i64,
    captured_at: DateTime<Local>,
    required_horizon: i64,
    now: DateTime<Local>,
) -> bool {
    if required_horizon > horizon_days {
        return false;
    }

    let elapsed_seconds = now.signed_duration_since(captured_at).num_seconds();
    elapsed_seconds >= 0
        && elapsed_seconds <= (horizon_days - required_horizon).saturating_mul(86_400)
}

/// In-memory snapshot of raw tool entries, scoped to a known scan
/// horizon (in days back from `now`). PageUp/PageDown reuse this cache so
/// they feel instant; only manual `r` and auto-refresh invalidate it.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RawDataCache {
    pub(crate) claude: Vec<UsageEntry>,
    pub(crate) codex: Vec<UsageEntry>,
    pub(crate) gemini: Vec<UsageEntry>,
    pub(crate) kimi: Vec<UsageEntry>,
    pub(crate) omp: Vec<UsageEntry>,
    pub(crate) horizon_days: i64,
    pub(crate) local_host_id: Option<String>,
    pub(crate) local_record_keys: HashSet<(String, String)>,
    /// True when cached local records were parsed by the session-id-aware
    /// parser. An absent requested id is then a valid result, not a reason to
    /// rescan every source log.
    #[serde(default)]
    pub(crate) local_session_metadata_current: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct HotRawSnapshot {
    captured_at_seconds: i64,
    cache: RawDataCache,
}

fn recent_entries(entries: &[UsageEntry], now: DateTime<Local>) -> Vec<UsageEntry> {
    data::filter_usage_data_by_window(
        entries,
        &TimeWindow::rolling_days(HOT_CACHE_HORIZON_DAYS),
        now,
    )
}

fn write_hot_raw_snapshot(
    cache_root: &Path,
    source: &RawDataCache,
    captured_at: DateTime<Local>,
) -> std::io::Result<()> {
    let cache = RawDataCache {
        claude: recent_entries(&source.claude, captured_at),
        codex: recent_entries(&source.codex, captured_at),
        gemini: recent_entries(&source.gemini, captured_at),
        kimi: recent_entries(&source.kimi, captured_at),
        omp: recent_entries(&source.omp, captured_at),
        horizon_days: HOT_CACHE_HORIZON_DAYS,
        local_host_id: source.local_host_id.clone(),
        // A hot snapshot is never extended in place: a full refresh replaces
        // it, so retaining every historic local dedup key wastes memory.
        local_record_keys: HashSet::new(),
        local_session_metadata_current: source.local_session_metadata_current,
    };
    data::cache::write_hot_snapshot(
        cache_root,
        &HotRawSnapshot {
            captured_at_seconds: captured_at.timestamp(),
            cache,
        },
    )
}

fn load_hot_raw_snapshot(
    cache_root: &Path,
    local_host_id: Option<&str>,
    required_horizon: i64,
    now: DateTime<Local>,
) -> Option<RawDataCache> {
    let snapshot: HotRawSnapshot = data::cache::load_hot_snapshot(cache_root).ok().flatten()?;
    let captured_at =
        DateTime::from_timestamp(snapshot.captured_at_seconds, 0)?.with_timezone(&Local);
    (snapshot.cache.local_host_id.as_deref() == local_host_id
        && snapshot.cache.horizon_days == HOT_CACHE_HORIZON_DAYS
        && hot_snapshot_covers(
            snapshot.cache.horizon_days,
            captured_at,
            required_horizon,
            now,
        ))
    .then_some(snapshot.cache)
}

/// Get the data directory for a tool, or None for "all".
pub(crate) fn get_tool_data_dir(tool: &str) -> Option<PathBuf> {
    match tool {
        "codex" => Some(get_codex_dir().join("sessions")),
        "gemini" => Some(get_gemini_dir().join("tmp")),
        "kimi" => Some(get_kimi_dir().join("sessions")),
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
        _ => None,
    }
}

/// Days of history needed to render the current time window.
///
/// A recent hot snapshot keeps this bounded to the visible window. Once the
/// user navigates beyond that snapshot, the complete cache is loaded.
pub(crate) fn compute_required_horizon(window: &TimeWindow, now: DateTime<Local>) -> i64 {
    let (start, _) = window.bounds(now);
    let days = now.signed_duration_since(start).num_days() + 2;
    days.max(1)
}

pub(crate) fn include_local_for_host_filter(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
) -> bool {
    match host_filter {
        None => true,
        Some(host) => local_host_id == Some(host),
    }
}

pub(crate) fn merge_remote_records_into_raw_cache(
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
            "kimi" => cache.kimi.push(record.entry),
            "omp" => cache.omp.push(record.entry),
            _ => {}
        }
    }
}

/// Entries plus their `(vendor, dedup_key)` identities for one vendor cache.
type LoadedVendorCache = (Vec<UsageEntry>, HashSet<(String, String)>, bool);

pub(crate) fn load_local_tool_cached_records(cache_root: &Path, tool: &str) -> LoadedVendorCache {
    let records = data::cache::load_vendor_cached_records(cache_root, tool);
    let session_metadata_current =
        records.is_empty() || data::cache::vendor_session_metadata_is_current(cache_root, tool);
    let mut entries = Vec::with_capacity(records.len());
    let mut keys = HashSet::with_capacity(records.len());
    for record in records {
        if !record.dedup_key.is_empty() {
            keys.insert((tool.to_string(), record.dedup_key));
        }
        entries.push(record.entry);
    }
    (entries, keys, session_metadata_current)
}

pub(crate) fn local_cached_raw_cache(
    cache_root: &Path,
    include_local: bool,
    local_host_id: Option<&str>,
) -> RawDataCache {
    // The five vendor caches deserialize independently; load them in parallel.
    let mut loaded: Vec<LoadedVendorCache> = ["claude", "codex", "gemini", "kimi", "omp"]
        .par_iter()
        .map(|tool| {
            if include_local {
                load_local_tool_cached_records(cache_root, tool)
            } else {
                (Vec::new(), HashSet::new(), true)
            }
        })
        .collect();

    let mut local_record_keys = HashSet::new();
    let mut take = |slot: &mut LoadedVendorCache| {
        local_record_keys.extend(std::mem::take(&mut slot.1));
        std::mem::take(&mut slot.0)
    };
    let local_session_metadata_current = loaded.iter().all(|slot| slot.2);
    let claude = take(&mut loaded[0]);
    let codex = take(&mut loaded[1]);
    let gemini = take(&mut loaded[2]);
    let kimi = take(&mut loaded[3]);
    let omp = take(&mut loaded[4]);

    RawDataCache {
        claude,
        codex,
        gemini,
        kimi,
        omp,
        horizon_days: FULL_CACHE_HORIZON,
        local_host_id: local_host_id.map(str::to_string),
        local_record_keys,
        local_session_metadata_current,
    }
}

pub(crate) fn clear_local_raw_cache(cache: &mut RawDataCache) {
    cache.claude.clear();
    cache.codex.clear();
    cache.gemini.clear();
    cache.kimi.clear();
    cache.omp.clear();
    cache.local_record_keys.clear();
    cache.local_session_metadata_current = true;
}

pub(crate) fn read_all_tool_cached_snapshot_for_hosts(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
) -> RawDataCache {
    let cache_root = data::cache::default_cache_dir();
    let include_local = include_local_for_host_filter(host_filter, local_host_id);
    let host_set = host_filter.map(|host| HashSet::from([host.to_string()]));
    // Local vendor caches and the remote per-host caches load concurrently.
    let (mut cache, remote_records) = rayon::join(
        || local_cached_raw_cache(&cache_root, include_local, local_host_id),
        || data::cache::load_remote_entries(&cache_root, host_set.as_ref()),
    );
    merge_remote_records_into_raw_cache(&mut cache, remote_records);
    cache
}

/// Load the complete authoritative cache and refresh the derived hot snapshot
/// for ordinary all-host views.
pub(crate) fn read_full_cached_raw_data_for_hosts(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    now: DateTime<Local>,
) -> RawDataCache {
    let cache = read_all_tool_cached_snapshot_for_hosts(host_filter, local_host_id);
    if host_filter.is_none() && raw_cache_has_any_tool_data(&cache) {
        let _ = write_hot_raw_snapshot(&data::cache::default_cache_dir(), &cache, now);
    }
    cache
}

/// Prefer the compact recent snapshot for the common all-host rolling view.
/// It is only accepted while its timestamp range fully covers the request;
/// otherwise the canonical complete cache is loaded and regenerates it.
pub(crate) fn read_cached_raw_data_for_window(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    required_horizon: i64,
    now: DateTime<Local>,
) -> RawDataCache {
    let cache_root = data::cache::default_cache_dir();
    if host_filter.is_none()
        && let Some(cache) =
            load_hot_raw_snapshot(&cache_root, local_host_id, required_horizon, now)
    {
        return cache;
    }
    read_full_cached_raw_data_for_hosts(host_filter, local_host_id, now)
}

pub(crate) fn refresh_all_tool_raw_full(local_host_id: Option<&str>) -> RawDataCache {
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

    let kimi_dir = get_kimi_dir().join("sessions");
    let _ = data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "kimi",
        data::kimi::collect_usage_files(&kimi_dir, None),
        0,
        data::kimi::read_kimi_file_records,
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
pub(crate) fn ensure_raw_cache(state: &mut AppState, required_horizon: i64) -> &RawDataCache {
    let needs_load = match &state.raw_cache {
        None => true,
        Some(cache) => cache.horizon_days < required_horizon,
    };
    if needs_load {
        state.raw_cache = Some(read_cached_raw_data_for_window(
            state.host.as_deref(),
            state.local_host_id.as_deref(),
            required_horizon,
            Local::now(),
        ));
    }
    state.raw_cache.as_ref().unwrap()
}

pub(crate) fn start_background_raw_refresh(state: &mut AppState) {
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
        let mut refreshed = refresh_all_tool_raw_full(local_host_id.as_deref());
        if !include_local_for_host_filter(host_filter.as_deref(), local_host_id.as_deref()) {
            clear_local_raw_cache(&mut refreshed);
        }
        let cache_root = data::cache::default_cache_dir();
        let host_set = host_filter
            .as_ref()
            .map(|host| HashSet::from([host.clone()]));
        let remote_records = data::cache::load_remote_entries(&cache_root, host_set.as_ref());
        merge_remote_records_into_raw_cache(&mut refreshed, remote_records);
        if host_filter.is_none() && raw_cache_has_any_tool_data(&refreshed) {
            let _ = write_hot_raw_snapshot(&cache_root, &refreshed, Local::now());
        }
        let _ = tx.send(refreshed);
    });
}

pub(crate) fn poll_background_raw_refresh(state: &mut AppState) -> bool {
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

/// Loaded and filtered data for all tools.
pub(crate) struct AllToolData {
    pub(crate) claude: Vec<UsageEntry>,
    pub(crate) codex: Vec<UsageEntry>,
    pub(crate) gemini: Vec<UsageEntry>,
    pub(crate) kimi: Vec<UsageEntry>,
    pub(crate) omp: Vec<UsageEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowDataState {
    NoSourceData,
    EmptyWindow,
    Populated,
}

pub(crate) fn classify_window_data(
    has_source_data: bool,
    has_window_data: bool,
) -> WindowDataState {
    match (has_source_data, has_window_data) {
        (false, _) => WindowDataState::NoSourceData,
        (true, false) => WindowDataState::EmptyWindow,
        (true, true) => WindowDataState::Populated,
    }
}

pub(crate) fn raw_cache_has_any_tool_data(cache: &RawDataCache) -> bool {
    !cache.claude.is_empty()
        || !cache.codex.is_empty()
        || !cache.gemini.is_empty()
        || !cache.kimi.is_empty()
        || !cache.omp.is_empty()
}

pub(crate) fn all_tool_data_has_window_data(all_data: &AllToolData) -> bool {
    !all_data.claude.is_empty()
        || !all_data.codex.is_empty()
        || !all_data.gemini.is_empty()
        || !all_data.kimi.is_empty()
        || !all_data.omp.is_empty()
}

pub(crate) fn load_all_tool_data(state: &mut AppState, now: DateTime<Local>) -> AllToolData {
    let horizon = compute_required_horizon(&state.time_window, now);
    let window = state.time_window.clone();
    let session_id = state.session_id.clone();
    let cache = ensure_raw_cache(state, horizon);
    AllToolData {
        claude: data::filter_usage_data_by_window_and_session(
            &cache.claude,
            &window,
            session_id.as_deref(),
            now,
        ),
        codex: data::filter_usage_data_by_window_and_session(
            &cache.codex,
            &window,
            session_id.as_deref(),
            now,
        ),
        gemini: data::filter_usage_data_by_window_and_session(
            &cache.gemini,
            &window,
            session_id.as_deref(),
            now,
        ),
        kimi: data::filter_usage_data_by_window_and_session(
            &cache.kimi,
            &window,
            session_id.as_deref(),
            now,
        ),
        omp: data::filter_usage_data_by_window_and_session(
            &cache.omp,
            &window,
            session_id.as_deref(),
            now,
        ),
    }
}

fn merge_tool_series(mut a: ToolTimeSeries, b: ToolTimeSeries) -> ToolTimeSeries {
    for (interval_time, tools) in b {
        let target = a.entry(interval_time).or_default();
        for (label, total) in tools {
            *target.entry(label).or_insert(0.0) += total;
        }
    }
    a
}

/// Per-interval token totals for one harness, chunk-parallel over its entries.
fn tool_series(entries: &[UsageEntry], tool: Tool, interval_minutes: i64) -> ToolTimeSeries {
    let tool_label = tool.comparison_label();
    entries
        .par_chunks(stats::PAR_CHUNK)
        .fold(HashMap::new, |mut ts: ToolTimeSeries, chunk| {
            for entry in chunk {
                if entry.timestamp.is_empty() {
                    continue;
                }

                let total = match tool {
                    Tool::Codex => {
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
                        .entry(tool_label.to_string())
                        .or_insert(0.0) += total;
                }
            }
            ts
        })
        .reduce(HashMap::new, merge_tool_series)
}

pub(crate) fn calculate_tool_aggregate_time_series(
    all_data: &AllToolData,
    interval_minutes: i64,
) -> ToolTimeSeries {
    let buckets: [(&[UsageEntry], Tool); 5] = [
        (&all_data.claude, Tool::Claude),
        (&all_data.codex, Tool::Codex),
        (&all_data.gemini, Tool::Gemini),
        (&all_data.kimi, Tool::Kimi),
        (&all_data.omp, Tool::Omp),
    ];
    buckets
        .into_iter()
        .filter(|(entries, _)| !entries.is_empty())
        .map(|(entries, tool)| tool_series(entries, tool, interval_minutes))
        .fold(HashMap::new(), merge_tool_series)
}

pub(crate) fn calculate_all_model_breakdown(
    all_data: &AllToolData,
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
    if !all_data.kimi.is_empty() {
        all_stats.extend(stats::calculate_kimi_model_breakdown(
            &all_data.kimi,
            pricing,
        ));
    }
    if !all_data.omp.is_empty() {
        all_stats.extend(stats::calculate_omp_model_breakdown(&all_data.omp, pricing));
    }

    all_stats.sort_by(|a, b| b.count.cmp(&a.count));
    all_stats
}

/// Calculate weighted average cost per MTok and total monthly savings across all tools.
/// Returns (weighted_cost_per_mtok, total_monthly_savings).
pub(crate) fn calculate_weighted_cost_per_mtok(
    model_stats: &[ModelBreakdownRow],
    days: f64,
    subscription_fees: &SubscriptionFees,
) -> (f64, f64) {
    let mut by_tool: HashMap<&str, (i64, f64)> = HashMap::new();
    for row in model_stats {
        let totals = by_tool.entry(row.tool.as_str()).or_default();
        totals.0 += row.total_with_cache;
        totals.1 +=
            row.input_cost + row.output_cost + row.cache_read_cost + row.cache_creation_cost;
    }

    let grand_total: i64 = by_tool.values().map(|(tokens, _)| tokens).sum();
    if grand_total == 0 || days <= 0.0 {
        return (0.0, 0.0);
    }

    let mut weighted_cost = 0.0;
    let mut total_savings = 0.0;

    for tool in ["claude", "codex", "gemini", "kimi", "omp"] {
        let Some(&(tokens, api_cost)) = by_tool.get(tool) else {
            continue;
        };
        if tokens == 0 {
            continue;
        }
        let sub_price = subscription_fees.get(tool);
        let percentage = tokens as f64 / grand_total as f64;
        let monthly_tokens = (tokens as f64 / days) * 30.0;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use chrono::{Duration, TimeZone};

    #[test]
    fn weighted_cost_reuses_model_breakdown_totals() {
        let row = |tool: &str, tokens: i64, cost: f64| ModelBreakdownRow {
            model: format!("{tool}-model"),
            tool: tool.to_string(),
            count: 1,
            input: tokens,
            output: 0,
            cache_creation: 0,
            cache_read: 0,
            reasoning: 0,
            thinking: 0,
            total: tokens,
            total_with_cache: tokens,
            input_cost: cost,
            output_cost: 0.0,
            cache_read_cost: 0.0,
            cache_creation_cost: 0.0,
        };
        let rows = [
            row("claude", 1_000_000, 10.0),
            row("codex", 3_000_000, 30.0),
        ];
        let fees = SubscriptionFees {
            claude: 20.0,
            codex: 40.0,
            ..Default::default()
        };

        let (weighted_cost, savings) = calculate_weighted_cost_per_mtok(&rows, 10.0, &fees);

        assert!((weighted_cost - 5.0).abs() < 1e-9);
        assert!((savings - 60.0).abs() < 1e-9);
    }

    #[test]
    fn hot_snapshot_only_covers_windows_inside_its_recent_history() {
        let now = Local
            .with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
            .single()
            .expect("fixed now");
        let captured_at = now - Duration::days(2);

        assert!(hot_snapshot_covers(
            HOT_CACHE_HORIZON_DAYS,
            captured_at,
            3,
            now
        ));
        assert!(!hot_snapshot_covers(
            HOT_CACHE_HORIZON_DAYS,
            captured_at,
            7,
            now
        ));
    }

    #[test]
    fn hot_snapshot_round_trip_keeps_only_recent_entries() {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = Local
            .with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
            .single()
            .expect("fixed now");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let cache_root = std::env::temp_dir().join(format!("ai-usage-hot-cache-{stamp}"));
        fs::create_dir_all(&cache_root).expect("create cache root");

        let entry = |timestamp: DateTime<Local>, input_tokens| UsageEntry {
            host_id: None,
            session_id: None,
            timestamp: timestamp.to_rfc3339(),
            parsed_timestamp: Some(timestamp),
            session_start_time: String::new(),
            session_end_time: String::new(),
            model: "test-model".to_string(),
            effort: None,
            fast_tier: -1,
            usage: data::TokenUsage {
                input_tokens,
                ..Default::default()
            },
            costs: None,
        };
        let source = RawDataCache {
            claude: vec![
                entry(now - Duration::days(1), 11),
                entry(now - Duration::days(HOT_CACHE_HORIZON_DAYS + 1), 99),
            ],
            codex: vec![entry(now - Duration::days(2), 22)],
            gemini: Vec::new(),
            kimi: Vec::new(),
            omp: Vec::new(),
            horizon_days: FULL_CACHE_HORIZON,
            local_host_id: Some("workstation".to_string()),
            local_record_keys: HashSet::new(),
            local_session_metadata_current: true,
        };

        write_hot_raw_snapshot(&cache_root, &source, now).expect("write hot snapshot");
        let snapshot = load_hot_raw_snapshot(&cache_root, Some("workstation"), 3, now)
            .expect("load hot snapshot");

        assert_eq!(snapshot.claude.len(), 1);
        assert_eq!(snapshot.claude[0].usage.input_tokens, 11);
        assert_eq!(snapshot.codex.len(), 1);
        assert_eq!(snapshot.codex[0].usage.input_tokens, 22);
        assert_eq!(snapshot.horizon_days, HOT_CACHE_HORIZON_DAYS);
        assert!(snapshot.local_session_metadata_current);
        assert!(load_hot_raw_snapshot(&cache_root, Some("other-host"), 3, now).is_none());

        fs::remove_dir_all(cache_root).expect("remove cache root");
    }

    #[test]
    fn horizon_covers_the_current_window() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        assert_eq!(compute_required_horizon(&window, now), 5);
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
    fn remote_records_merge_into_tool_buckets() {
        let timestamp = "2026-05-18T12:00:00Z";
        let mut cache = RawDataCache {
            claude: Vec::new(),
            codex: Vec::new(),
            gemini: Vec::new(),
            kimi: Vec::new(),
            omp: Vec::new(),
            horizon_days: FULL_CACHE_HORIZON,
            local_host_id: None,
            local_record_keys: HashSet::new(),
            local_session_metadata_current: true,
        };
        merge_remote_records_into_raw_cache(
            &mut cache,
            vec![
                data::cache::RemoteUsageRecord {
                    vendor: "claude".to_string(),
                    dedup_key: "a".to_string(),
                    entry: UsageEntry {
                        host_id: Some("laptop".to_string()),
                        session_id: None,
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
                        session_id: None,
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
                session_id: None,
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
            kimi: Vec::new(),
            omp: Vec::new(),
            horizon_days: FULL_CACHE_HORIZON,
            local_host_id: Some("laptop".to_string()),
            local_record_keys: HashSet::from([("claude".to_string(), "a".to_string())]),
            local_session_metadata_current: true,
        };
        merge_remote_records_into_raw_cache(
            &mut cache,
            vec![
                data::cache::RemoteUsageRecord {
                    vendor: "claude".to_string(),
                    dedup_key: "a".to_string(),
                    entry: UsageEntry {
                        host_id: Some("laptop".to_string()),
                        session_id: None,
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
                        session_id: None,
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
