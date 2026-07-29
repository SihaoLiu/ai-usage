//! Raw usage-entry cache: loading, host filtering, background refresh,
//! and cross-tool aggregation.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{OnceLock, mpsc};
use std::thread;
use std::time::{Duration as StdDuration, Instant};

use chrono::{DateTime, Duration, Local};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use std::path::Path;

use crate::AppState;
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
use crate::time_utils::{self, TimeWindow};
use crate::tool::Tool;

/// Short rolling windows are common enough to keep a compact derived snapshot
/// beside the authoritative per-source caches.
const HOT_CACHE_HORIZON_DAYS: i64 = 8;
const NAVIGATION_PREFETCH_WINDOWS: i64 = 8;
const LARGE_WINDOW_PREFETCH_SPANS: i64 = 4;
const MIN_NAVIGATION_PREFETCH_DAYS: i64 = 14;
const MAX_SCALED_NAVIGATION_PREFETCH_DAYS: i64 = 365;
pub(crate) const RAW_CACHE_IDLE_TTL: StdDuration = StdDuration::from_secs(5 * 60);
#[cfg(test)]
static IDLE_RECLAIM_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn background_refresh_parallelism(available: usize) -> usize {
    (available / 4).clamp(1, 4)
}

fn background_refresh_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let available = thread::available_parallelism().map_or(1, usize::from);
        rayon::ThreadPoolBuilder::new()
            .num_threads(background_refresh_parallelism(available))
            .thread_name(|index| format!("usage-refresh-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

fn refresh_all_tool_caches_in_background() {
    match background_refresh_pool() {
        Some(pool) => pool.install(refresh_all_tool_caches),
        None => refresh_all_tool_caches(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RawDataRange {
    start_seconds: i64,
    start_nanos: u32,
    end_seconds: i64,
    end_nanos: u32,
}

impl RawDataRange {
    pub(crate) fn from_bounds(start: DateTime<Local>, end: DateTime<Local>) -> Self {
        Self {
            start_seconds: start.timestamp(),
            start_nanos: start.timestamp_subsec_nanos(),
            end_seconds: end.timestamp(),
            end_nanos: end.timestamp_subsec_nanos(),
        }
    }

    pub(crate) fn start(self) -> DateTime<Local> {
        DateTime::from_timestamp(self.start_seconds, self.start_nanos)
            .expect("cached range start timestamp")
            .with_timezone(&Local)
    }

    pub(crate) fn end(self) -> DateTime<Local> {
        DateTime::from_timestamp(self.end_seconds, self.end_nanos)
            .expect("cached range end timestamp")
            .with_timezone(&Local)
    }

    pub(crate) fn covers(self, other: Self) -> bool {
        (self.start_seconds, self.start_nanos) <= (other.start_seconds, other.start_nanos)
            && (self.end_seconds, self.end_nanos) >= (other.end_seconds, other.end_nanos)
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let start = self.start().max(other.start());
        let end = self.end().min(other.end());
        (start <= end).then(|| Self::from_bounds(start, end))
    }
}

/// In-memory snapshot of raw tool entries scoped to a bounded time range.
/// Installed tool vectors are ordered by timestamp so navigation can borrow
/// each visible slice with two binary searches.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RawDataCache {
    pub(crate) claude: Vec<UsageEntry>,
    pub(crate) codex: Vec<UsageEntry>,
    pub(crate) gemini: Vec<UsageEntry>,
    pub(crate) kimi: Vec<UsageEntry>,
    pub(crate) omp: Vec<UsageEntry>,
    pub(crate) range: RawDataRange,
    #[serde(default)]
    pub(crate) has_source_data: bool,
    pub(crate) local_host_id: Option<String>,
    #[serde(skip)]
    pub(crate) local_record_keys: HashMap<Tool, HashSet<String>>,
    #[serde(default)]
    pub(crate) persistent_generation: String,
    /// True when cached local records were parsed by the session-id-aware
    /// parser. An absent requested id is then a valid result, not a reason to
    /// rescan every source log.
    #[serde(default)]
    pub(crate) local_session_metadata_current: bool,
}

pub(crate) enum BackgroundRawLoad {
    Refreshed(Box<RawDataCache>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BackgroundSourceRefresh {
    pub(crate) changed: bool,
    pub(crate) generation: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct HotRawSnapshot {
    cache: RawDataCache,
}

#[derive(Serialize)]
struct HotRawSnapshotRef<'a> {
    cache: &'a RawDataCache,
}

fn sorted_window_slice<'a>(
    entries: &'a [UsageEntry],
    window: &TimeWindow,
    now: DateTime<Local>,
) -> &'a [UsageEntry] {
    let (start, end) = window.bounds(now);
    sorted_range_slice(entries, RawDataRange::from_bounds(start, end))
}

fn sorted_range_slice(entries: &[UsageEntry], range: RawDataRange) -> &[UsageEntry] {
    let start = range.start();
    let end = range.end();
    let first = entries
        .partition_point(|entry| entry_timestamp(entry).is_none_or(|timestamp| timestamp < start));
    let last = first
        + entries[first..].partition_point(|entry| {
            entry_timestamp(entry).is_some_and(|timestamp| timestamp <= end)
        });
    &entries[first..last]
}

fn recent_entries(entries: &[UsageEntry], now: DateTime<Local>) -> Vec<UsageEntry> {
    sorted_range_slice(entries, hot_cache_range(now)).to_vec()
}

fn prepare_hot_raw_snapshot(source: &RawDataCache, captured_at: DateTime<Local>) -> HotRawSnapshot {
    let range = hot_cache_range(captured_at);
    HotRawSnapshot {
        cache: RawDataCache {
            claude: recent_entries(&source.claude, captured_at),
            codex: recent_entries(&source.codex, captured_at),
            gemini: recent_entries(&source.gemini, captured_at),
            kimi: recent_entries(&source.kimi, captured_at),
            omp: recent_entries(&source.omp, captured_at),
            range,
            has_source_data: source.has_source_data,
            local_host_id: source.local_host_id.clone(),
            local_record_keys: HashMap::new(),
            persistent_generation: source.persistent_generation.clone(),
            local_session_metadata_current: source.local_session_metadata_current,
        },
    }
}

fn write_hot_raw_snapshot(
    cache_root: &Path,
    source: &RawDataCache,
    captured_at: DateTime<Local>,
) -> std::io::Result<()> {
    if source.range == hot_cache_range(captured_at) && source.local_record_keys.is_empty() {
        return data::cache::write_hot_snapshot(cache_root, &HotRawSnapshotRef { cache: source });
    }
    data::cache::write_hot_snapshot(cache_root, &prepare_hot_raw_snapshot(source, captured_at))
}

fn load_hot_raw_snapshot(
    cache_root: &Path,
    local_host_id: Option<&str>,
    required_range: RawDataRange,
) -> Option<RawDataCache> {
    let snapshot: HotRawSnapshot = data::cache::load_hot_snapshot(cache_root).ok().flatten()?;
    if !(snapshot.cache.local_host_id.as_deref() == local_host_id
        && snapshot.cache.persistent_generation
            == crate::sync::cache_generation::raw_data_generation(cache_root)
        && snapshot.cache.range.covers(required_range))
    {
        return None;
    }
    let mut cache = snapshot.cache;
    sort_raw_cache(&mut cache);
    Some(cache)
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

fn hot_cache_range(now: DateTime<Local>) -> RawDataRange {
    RawDataRange::from_bounds(
        now - Duration::days(HOT_CACHE_HORIZON_DAYS),
        now + Duration::days(HOT_CACHE_HORIZON_DAYS),
    )
}

fn navigation_prefetch_days(window: &TimeWindow, now: DateTime<Local>) -> i64 {
    let (start, end) = window.bounds(now);
    let span_seconds = end.signed_duration_since(start).num_seconds().max(1);
    let span_days = span_seconds.saturating_add(86_399) / 86_400;
    // Refill at half capacity so a held navigation key has several complete
    // viewports of headroom while the next slice streams from persistent data.
    span_days.saturating_mul(LARGE_WINDOW_PREFETCH_SPANS).max(
        span_days.saturating_mul(NAVIGATION_PREFETCH_WINDOWS).clamp(
            MIN_NAVIGATION_PREFETCH_DAYS,
            MAX_SCALED_NAVIGATION_PREFETCH_DAYS,
        ),
    )
}

pub(crate) fn raw_cache_visible_range(window: &TimeWindow, now: DateTime<Local>) -> RawDataRange {
    let (start, end) = window.bounds(now);
    RawDataRange::from_bounds(start, end)
}

fn expanded_navigation_range(
    window: &TimeWindow,
    now: DateTime<Local>,
    margin_days: i64,
) -> RawDataRange {
    let visible = raw_cache_visible_range(window, now);
    let unaligned_start = visible.start() - Duration::days(margin_days);
    let unaligned_end = visible.end() + Duration::days(margin_days);
    let start_seconds = unaligned_start
        .timestamp()
        .div_euclid(86_400)
        .saturating_mul(86_400);
    let end_seconds = unaligned_end
        .timestamp()
        .div_euclid(86_400)
        .saturating_add(1)
        .saturating_mul(86_400)
        .saturating_sub(1);
    let start = DateTime::from_timestamp(start_seconds, 0)
        .expect("prefetch range start")
        .with_timezone(&Local);
    let end = DateTime::from_timestamp(end_seconds, 999_999_999)
        .expect("prefetch range end")
        .with_timezone(&Local);
    RawDataRange::from_bounds(start, end)
}

pub(crate) fn raw_cache_target_range(window: &TimeWindow, now: DateTime<Local>) -> RawDataRange {
    expanded_navigation_range(window, now, navigation_prefetch_days(window, now))
}

fn raw_cache_trigger_range(window: &TimeWindow, now: DateTime<Local>) -> RawDataRange {
    let margin = navigation_prefetch_days(window, now);
    expanded_navigation_range(window, now, margin / 2)
}

pub(crate) fn raw_cache_covers_window(state: &AppState, now: DateTime<Local>) -> bool {
    let required_range = raw_cache_visible_range(&state.time_window, now);
    state
        .raw_cache
        .as_ref()
        .is_some_and(|cache| cache.range.covers(required_range))
}

pub(crate) fn raw_cache_needs_prefetch(state: &AppState, now: DateTime<Local>) -> bool {
    let trigger_range = raw_cache_trigger_range(&state.time_window, now);
    state
        .raw_cache
        .as_ref()
        .is_none_or(|cache| !cache.range.covers(trigger_range))
}

fn entry_timestamp(entry: &UsageEntry) -> Option<DateTime<Local>> {
    entry
        .parsed_timestamp
        .or_else(|| time_utils::parse_timestamp(&entry.timestamp))
}

fn sort_raw_cache(cache: &mut RawDataCache) {
    let sort = |entries: &mut Vec<UsageEntry>| {
        entries.sort_unstable_by_key(entry_timestamp);
    };
    sort(&mut cache.claude);
    sort(&mut cache.codex);
    sort(&mut cache.gemini);
    sort(&mut cache.kimi);
    sort(&mut cache.omp);
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
        let Some(tool) = Tool::from_key(&record.vendor) else {
            continue;
        };
        if cache.local_host_id.as_deref() == Some(host_id)
            && cache
                .local_record_keys
                .get(&tool)
                .is_some_and(|keys| keys.contains(&record.dedup_key))
        {
            continue;
        }
        match tool {
            Tool::Claude => cache.claude.push(record.entry),
            Tool::Codex => cache.codex.push(record.entry),
            Tool::Gemini => cache.gemini.push(record.entry),
            Tool::Kimi => cache.kimi.push(record.entry),
            Tool::Omp => cache.omp.push(record.entry),
            Tool::All => {}
        }
    }
}

/// Entries, dedup identities, metadata currency, and global presence for one
/// vendor cache.
type LoadedVendorCache = (Vec<UsageEntry>, HashSet<String>, bool, bool);

pub(crate) fn load_local_tool_cached_records(
    cache_root: &Path,
    tool: &str,
    range: RawDataRange,
) -> LoadedVendorCache {
    let (records, has_cached_records) = data::cache::load_vendor_cached_records_in_range(
        cache_root,
        tool,
        range.start(),
        range.end(),
    );
    let session_metadata_current =
        !has_cached_records || data::cache::vendor_session_metadata_is_current(cache_root, tool);
    let mut entries = Vec::with_capacity(records.len());
    let mut keys = HashSet::with_capacity(records.len());
    for (dedup_key, entry) in records {
        if !dedup_key.is_empty() {
            keys.insert(dedup_key);
        }
        entries.push(entry);
    }
    (entries, keys, session_metadata_current, has_cached_records)
}

pub(crate) fn local_cached_raw_cache(
    cache_root: &Path,
    include_local: bool,
    local_host_id: Option<&str>,
    range: RawDataRange,
) -> RawDataCache {
    // Decode one cache at a time so a large history cannot multiply its peak
    // memory use across all vendors.
    let mut loaded: Vec<LoadedVendorCache> = ["claude", "codex", "gemini", "kimi", "omp"]
        .iter()
        .map(|tool| {
            if include_local {
                load_local_tool_cached_records(cache_root, tool, range)
            } else {
                (Vec::new(), HashSet::new(), true, false)
            }
        })
        .collect();

    let mut local_record_keys = HashMap::new();
    let mut take = |tool, slot: &mut LoadedVendorCache| {
        let keys = std::mem::take(&mut slot.1);
        if !keys.is_empty() {
            local_record_keys.insert(tool, keys);
        }
        std::mem::take(&mut slot.0)
    };
    let local_session_metadata_current = loaded.iter().all(|slot| slot.2);
    let has_source_data = loaded.iter().any(|slot| slot.3);
    let claude = take(Tool::Claude, &mut loaded[0]);
    let codex = take(Tool::Codex, &mut loaded[1]);
    let gemini = take(Tool::Gemini, &mut loaded[2]);
    let kimi = take(Tool::Kimi, &mut loaded[3]);
    let omp = take(Tool::Omp, &mut loaded[4]);

    RawDataCache {
        claude,
        codex,
        gemini,
        kimi,
        omp,
        range,
        has_source_data,
        local_host_id: local_host_id.map(str::to_string),
        local_record_keys,
        persistent_generation: String::new(),
        local_session_metadata_current,
    }
}

fn load_scoped_persistent_raw_data_from(
    cache_root: &Path,
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    range: RawDataRange,
) -> RawDataCache {
    let persistent_generation = crate::sync::cache_generation::raw_data_generation(cache_root);
    let include_local = include_local_for_host_filter(host_filter, local_host_id);
    let host_set = host_filter.map(|host| HashSet::from([host.to_string()]));
    let mut cache = local_cached_raw_cache(cache_root, include_local, local_host_id, range);
    let (remote_records, has_remote_source_data) = data::cache::load_remote_entries_in_range(
        cache_root,
        host_set.as_ref(),
        range.start(),
        range.end(),
    );
    cache.has_source_data |= has_remote_source_data;
    merge_remote_records_into_raw_cache(&mut cache, remote_records);
    cache.local_record_keys = HashMap::new();
    cache.persistent_generation = persistent_generation;
    sort_raw_cache(&mut cache);
    cache
}

/// Load the requested slice of the authoritative caches and refresh the
/// derived hot snapshot for ordinary all-host views.
pub(crate) fn load_persistent_raw_data_for_window(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    range: RawDataRange,
    now: DateTime<Local>,
) -> RawDataCache {
    let cache_root = data::cache::default_cache_dir();
    load_persistent_raw_data_for_window_from(&cache_root, host_filter, local_host_id, range, now)
}

fn load_persistent_raw_data_for_window_from(
    cache_root: &Path,
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    range: RawDataRange,
    now: DateTime<Local>,
) -> RawDataCache {
    let cache = load_scoped_persistent_raw_data_from(cache_root, host_filter, local_host_id, range);
    if host_filter.is_none() && cache.range.covers(hot_cache_range(now)) && cache.has_source_data {
        let _ = write_hot_raw_snapshot(cache_root, &cache, now);
    }
    cache
}

/// Prefer the compact recent snapshot for the common all-host rolling view.
/// It is only accepted while its timestamp range fully covers the request;
/// otherwise the requested slice of the canonical caches regenerates it.
pub(crate) fn read_cached_raw_data_for_window(
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    range: RawDataRange,
    now: DateTime<Local>,
) -> RawDataCache {
    let cache_root = data::cache::default_cache_dir();
    let cache =
        read_cached_raw_data_for_window_from(&cache_root, host_filter, local_host_id, range);
    if host_filter.is_none() && cache.range.covers(hot_cache_range(now)) && cache.has_source_data {
        let _ = write_hot_raw_snapshot(&cache_root, &cache, now);
    }
    cache
}

fn read_cached_raw_data_for_window_from(
    cache_root: &Path,
    host_filter: Option<&str>,
    local_host_id: Option<&str>,
    range: RawDataRange,
) -> RawDataCache {
    if host_filter.is_none()
        && hot_cache_range(Local::now()).covers(range)
        && let Some(cache) = load_hot_raw_snapshot(cache_root, local_host_id, range)
    {
        return cache;
    }
    load_scoped_persistent_raw_data_from(cache_root, host_filter, local_host_id, range)
}

pub(crate) fn refresh_all_tool_caches() {
    let cache_root = data::cache::default_cache_dir();
    let claude_fast_tier = detect_claude_fast_tier_snapshot();
    data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "claude",
        data::claude::collect_usage_files(None),
        claude_fast_tier,
        data::claude::read_jsonl_file_records,
    );

    let codex_dir = get_codex_dir().join("sessions");
    let codex_fast_tier = detect_codex_fast_tier_snapshot();
    data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "codex",
        data::codex::collect_usage_files(&codex_dir, None),
        codex_fast_tier,
        data::codex::read_codex_file_records,
    );

    let gemini_dir = get_gemini_dir().join("tmp");
    data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "gemini",
        data::gemini::collect_usage_files(&gemini_dir, None),
        0,
        data::gemini::read_gemini_file_records,
    );

    let kimi_dir = get_kimi_dir().join("sessions");
    data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "kimi",
        data::kimi::collect_usage_files(&kimi_dir, None),
        0,
        data::kimi::read_kimi_file_records,
    );

    let omp_dir = get_omp_dir().join("agent").join("sessions");
    data::cache::refresh_retaining_vendor_cache(
        &cache_root,
        "omp",
        data::omp::collect_usage_files(&omp_dir, None),
        0,
        data::omp::read_omp_file_records,
    );
}

/// Ensure `state.raw_cache` covers the requested range. Returns a reference
/// to the populated cache so callers can filter without another disk read.
pub(crate) fn ensure_raw_cache(state: &mut AppState, range: RawDataRange) -> &RawDataCache {
    let needs_load = match &state.raw_cache {
        None => true,
        Some(cache) => !cache.range.covers(range),
    };
    if needs_load {
        state.raw_cache = Some(read_cached_raw_data_for_window(
            state.host.as_deref(),
            state.local_host_id.as_deref(),
            range,
            Local::now(),
        ));
    }
    state.raw_cache.as_ref().unwrap()
}

pub(crate) fn start_background_raw_reload(state: &mut AppState, range: RawDataRange) {
    start_background_raw_load(state, range);
}

pub(crate) fn start_background_raw_prefetch(state: &mut AppState, range: RawDataRange) {
    start_background_raw_load(state, range);
}

fn start_background_raw_load(state: &mut AppState, range: RawDataRange) {
    if state.raw_refresh.is_some() {
        return;
    }
    let (tx, rx) = mpsc::sync_channel(0);
    let cache_root = data::cache::default_cache_dir();
    let host_filter = state.host.clone();
    let local_host_id = state.local_host_id.clone();
    state.raw_refresh = Some(rx);
    let _ = thread::Builder::new()
        .name("usage-cache-load".to_string())
        .spawn(move || {
            run_background_raw_load(cache_root, host_filter, local_host_id, range, tx);
        });
}

fn run_background_raw_load(
    cache_root: PathBuf,
    host_filter: Option<String>,
    local_host_id: Option<String>,
    range: RawDataRange,
    tx: mpsc::SyncSender<BackgroundRawLoad>,
) {
    let now = Local::now();
    let refreshed = read_cached_raw_data_for_window_from(
        &cache_root,
        host_filter.as_deref(),
        local_host_id.as_deref(),
        range,
    );
    let hot_snapshot = (host_filter.is_none()
        && refreshed.range.covers(hot_cache_range(now))
        && refreshed.has_source_data)
        .then(|| prepare_hot_raw_snapshot(&refreshed, now));
    if tx
        .send(BackgroundRawLoad::Refreshed(Box::new(refreshed)))
        .is_ok()
        && let Some(snapshot) = hot_snapshot
    {
        let _ = data::cache::write_hot_snapshot(&cache_root, &snapshot);
    }
}

pub(crate) fn start_background_source_refresh(
    loaded_generation: Option<String>,
) -> mpsc::Receiver<BackgroundSourceRefresh> {
    let (tx, rx) = mpsc::channel();
    let _ = thread::Builder::new()
        .name("usage-source-refresh".to_string())
        .spawn(move || {
            refresh_all_tool_caches_in_background();
            crate::process_usage::release_unused_memory();
            let current_generation = crate::sync::cache_generation::raw_data_generation(
                &data::cache::default_cache_dir(),
            );
            let result = BackgroundSourceRefresh {
                changed: loaded_generation.as_deref() != Some(current_generation.as_str()),
                generation: current_generation,
            };
            let _ = tx.send(result);
        });
    rx
}

pub(crate) fn poll_background_raw_refresh(state: &mut AppState) -> bool {
    let Some(rx) = state.raw_refresh.take() else {
        return false;
    };
    match rx.try_recv() {
        Ok(BackgroundRawLoad::Refreshed(cache)) => {
            let cache = *cache;
            let visible_range = raw_cache_visible_range(&state.time_window, Local::now());
            let keep_resident = state.raw_cache.as_ref().is_some_and(|resident| {
                resident.range.covers(visible_range) && !cache.range.covers(visible_range)
            });
            let retired = if keep_resident {
                Some(cache)
            } else {
                state.raw_cache.replace(cache)
            };
            if state.raw_cache_last_used_at.is_none() {
                let desired = idle_raw_cache_retention_range(&state.time_window, Local::now());
                compact_resident_raw_cache(state, desired, retired);
            } else if let Some(retired) = retired {
                retire_raw_cache(retired);
            }
            true
        }
        Err(mpsc::TryRecvError::Empty) => {
            state.raw_refresh = Some(rx);
            false
        }
        Err(mpsc::TryRecvError::Disconnected) => false,
    }
}

pub(crate) fn retire_raw_cache(cache: RawDataCache) {
    retire_in_background(cache);
}

fn retire_in_background<T: Send + 'static>(value: T) {
    let _ = thread::Builder::new()
        .name("usage-cache-retire".to_string())
        .spawn(move || drop(value));
}

fn retire_idle_value<T: Send + 'static>(value: T) {
    let _ = thread::Builder::new()
        .name("usage-cache-retire".to_string())
        .spawn(move || {
            drop(value);
            #[cfg(test)]
            IDLE_RECLAIM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::process_usage::release_unused_memory();
        });
}

pub(crate) fn touch_raw_cache_at(state: &mut AppState, accessed_at: Instant) {
    state.raw_cache_last_used_at = Some(accessed_at);
}

pub(crate) fn idle_raw_cache_retention_range(
    window: &TimeWindow,
    now: DateTime<Local>,
) -> RawDataRange {
    let visible = raw_cache_visible_range(window, now);
    let page_step = window.page_step();
    let end = if matches!(window, TimeWindow::Latest { .. }) {
        visible.end()
    } else {
        visible.end() + page_step
    };
    RawDataRange::from_bounds(visible.start() - page_step, end)
}

fn take_entries_outside_range(
    entries: &mut Vec<UsageEntry>,
    range: RawDataRange,
) -> Vec<UsageEntry> {
    let start = range.start();
    let end = range.end();
    let first = entries
        .partition_point(|entry| entry_timestamp(entry).is_none_or(|timestamp| timestamp < start));
    let last = first
        + entries[first..].partition_point(|entry| {
            entry_timestamp(entry).is_some_and(|timestamp| timestamp <= end)
        });
    if first == 0 && last == entries.len() {
        return Vec::new();
    }
    let retained = entries.drain(first..last).collect();
    std::mem::replace(entries, retained)
}

pub(crate) fn retire_idle_raw_cache_at(
    state: &mut AppState,
    idle_now: Instant,
    window_now: DateTime<Local>,
) -> bool {
    let expired = state.raw_cache_last_used_at.is_some_and(|accessed_at| {
        idle_now.saturating_duration_since(accessed_at) >= RAW_CACHE_IDLE_TTL
    });
    if !expired {
        return false;
    }
    state.raw_cache_last_used_at = None;
    let desired = idle_raw_cache_retention_range(&state.time_window, window_now);
    compact_resident_raw_cache(state, desired, None)
}

fn compact_resident_raw_cache(
    state: &mut AppState,
    desired: RawDataRange,
    additional_retired: Option<RawDataCache>,
) -> bool {
    let Some(resident_range) = state.raw_cache.as_ref().map(|cache| cache.range) else {
        if let Some(retired) = additional_retired {
            retire_idle_value(retired);
        }
        return false;
    };
    let Some(retained_range) = resident_range.intersection(desired) else {
        retire_idle_value((
            additional_retired,
            state.raw_cache.take().expect("resident cache"),
        ));
        return true;
    };
    let cache = state.raw_cache.as_mut().expect("resident cache");
    let retired = [
        take_entries_outside_range(&mut cache.claude, retained_range),
        take_entries_outside_range(&mut cache.codex, retained_range),
        take_entries_outside_range(&mut cache.gemini, retained_range),
        take_entries_outside_range(&mut cache.kimi, retained_range),
        take_entries_outside_range(&mut cache.omp, retained_range),
    ];
    cache.range = retained_range;
    if additional_retired.is_some() || retired.iter().any(|entries| !entries.is_empty()) {
        retire_idle_value((additional_retired, retired));
    }
    true
}

/// Loaded and filtered data for all tools.
pub(crate) struct AllToolData<'a> {
    pub(crate) claude: Cow<'a, [UsageEntry]>,
    pub(crate) codex: Cow<'a, [UsageEntry]>,
    pub(crate) gemini: Cow<'a, [UsageEntry]>,
    pub(crate) kimi: Cow<'a, [UsageEntry]>,
    pub(crate) omp: Cow<'a, [UsageEntry]>,
}

impl Default for AllToolData<'_> {
    fn default() -> Self {
        Self {
            claude: Cow::Borrowed(&[]),
            codex: Cow::Borrowed(&[]),
            gemini: Cow::Borrowed(&[]),
            kimi: Cow::Borrowed(&[]),
            omp: Cow::Borrowed(&[]),
        }
    }
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

#[cfg(test)]
pub(crate) fn raw_cache_has_any_tool_data(cache: &RawDataCache) -> bool {
    !cache.claude.is_empty()
        || !cache.codex.is_empty()
        || !cache.gemini.is_empty()
        || !cache.kimi.is_empty()
        || !cache.omp.is_empty()
}

pub(crate) fn all_tool_data_has_window_data(all_data: &AllToolData<'_>) -> bool {
    !all_data.claude.is_empty()
        || !all_data.codex.is_empty()
        || !all_data.gemini.is_empty()
        || !all_data.kimi.is_empty()
        || !all_data.omp.is_empty()
}

pub(crate) fn load_all_tool_data(
    state: &mut AppState,
    now: DateTime<Local>,
) -> AllToolData<'static> {
    let range = raw_cache_visible_range(&state.time_window, now);
    let window = state.time_window.clone();
    let session_id = state.session_id.clone();
    let cache = ensure_raw_cache(state, range);
    filter_all_tool_data_owned(cache, &window, session_id.as_deref(), now)
}

pub(crate) fn load_resident_all_tool_data<'a>(
    state: &'a AppState,
    now: DateTime<Local>,
) -> AllToolData<'a> {
    let Some(cache) = state.raw_cache.as_ref() else {
        return AllToolData::default();
    };
    filter_all_tool_data_borrowed(cache, &state.time_window, state.session_id.as_deref(), now)
}

fn filter_all_tool_data_owned(
    cache: &RawDataCache,
    window: &TimeWindow,
    session_id: Option<&str>,
    now: DateTime<Local>,
) -> AllToolData<'static> {
    AllToolData {
        claude: Cow::Owned(data::filter_usage_data_by_window_and_session(
            &cache.claude,
            window,
            session_id,
            now,
        )),
        codex: Cow::Owned(data::filter_usage_data_by_window_and_session(
            &cache.codex,
            window,
            session_id,
            now,
        )),
        gemini: Cow::Owned(data::filter_usage_data_by_window_and_session(
            &cache.gemini,
            window,
            session_id,
            now,
        )),
        kimi: Cow::Owned(data::filter_usage_data_by_window_and_session(
            &cache.kimi,
            window,
            session_id,
            now,
        )),
        omp: Cow::Owned(data::filter_usage_data_by_window_and_session(
            &cache.omp, window, session_id, now,
        )),
    }
}

pub(crate) fn filter_all_tool_data_borrowed<'a>(
    cache: &'a RawDataCache,
    window: &TimeWindow,
    session_id: Option<&str>,
    now: DateTime<Local>,
) -> AllToolData<'a> {
    AllToolData {
        claude: cached_window(&cache.claude, window, session_id, now),
        codex: cached_window(&cache.codex, window, session_id, now),
        gemini: cached_window(&cache.gemini, window, session_id, now),
        kimi: cached_window(&cache.kimi, window, session_id, now),
        omp: cached_window(&cache.omp, window, session_id, now),
    }
}

fn cached_window<'a>(
    entries: &'a [UsageEntry],
    window: &TimeWindow,
    session_id: Option<&str>,
    now: DateTime<Local>,
) -> Cow<'a, [UsageEntry]> {
    let visible = sorted_window_slice(entries, window, now);
    match session_id {
        None => Cow::Borrowed(visible),
        Some(selected) => Cow::Owned(
            visible
                .iter()
                .filter(|entry| entry.session_id.as_deref() == Some(selected))
                .cloned()
                .collect(),
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
    all_data: &AllToolData<'_>,
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
    all_data: &AllToolData<'_>,
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

pub(crate) fn calculate_all_dashboard_data(
    all_data: &AllToolData<'_>,
    pricing: &AllPricing,
    interval_minutes: i64,
) -> (Vec<ModelBreakdownRow>, ToolTimeSeries) {
    let buckets: [(&[UsageEntry], Tool); 5] = [
        (&all_data.claude, Tool::Claude),
        (&all_data.codex, Tool::Codex),
        (&all_data.gemini, Tool::Gemini),
        (&all_data.kimi, Tool::Kimi),
        (&all_data.omp, Tool::Omp),
    ];
    let mut model_stats = Vec::new();
    let mut tool_time_series = HashMap::new();
    for (entries, tool) in buckets {
        if entries.is_empty() {
            continue;
        }
        let (rows, time_series) = stats::calculate_comparison_dashboard_data(
            entries,
            interval_minutes,
            tool.key(),
            tool.comparison_label(),
            pricing,
        );
        model_stats.extend(rows);
        tool_time_series = merge_tool_series(tool_time_series, time_series);
    }
    model_stats.sort_by(|a, b| b.count.cmp(&a.count));
    (model_stats, tool_time_series)
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
mod tests;
