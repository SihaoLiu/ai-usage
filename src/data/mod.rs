pub mod cache;
pub mod claude;
pub mod codex;
pub mod gemini;
pub mod kimi;
pub mod omp;

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::time_utils::{TimeWindow, parse_timestamp};
use chrono::{DateTime, Local};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

pub const UNKNOWN_FAST_TIER: i8 = -1;

/// Recursively collect files under `dir` whose path satisfies `matches`,
/// keeping only files modified within the last `max_age_days` (with one day
/// of slack; files whose mtime cannot be read are kept). Results are sorted.
/// Shared by every vendor's usage-file collector; only the name predicate
/// differs per vendor.
pub(crate) fn collect_recent_files(
    dir: &Path,
    max_age_days: Option<i64>,
    matches: impl Fn(&Path) -> bool,
) -> Vec<PathBuf> {
    if !dir.exists() {
        return Vec::new();
    }

    let cutoff = max_age_days
        .map(|days| SystemTime::now() - std::time::Duration::from_secs((days as u64 + 1) * 86400));

    let mut files: Vec<PathBuf> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            if !e.file_type().is_file() {
                return false;
            }
            if !matches(e.path()) {
                return false;
            }
            if let Some(cutoff_time) = cutoff
                && let Ok(meta) = e.metadata()
                && let Ok(mtime) = meta.modified()
            {
                return mtime >= cutoff_time;
            }
            true
        })
        .map(|e| e.path().to_path_buf())
        .collect();
    files.sort();
    files
}

/// Path predicate for the common `*.jsonl` session-log layout.
pub(crate) fn has_jsonl_extension(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "jsonl")
}

/// Resolve a vendor's config directory: the env override wins, else
/// `$HOME/<dir_name>`, else a literal `~/<dir_name>` placeholder.
pub(crate) fn config_dir(env_var: &str, dir_name: &str) -> PathBuf {
    std::env::var(env_var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(dir_name))
                .unwrap_or_else(|_| PathBuf::from(format!("~/{dir_name}")))
        })
}

/// Fetch an integer field from a JSON object, defaulting to 0.
pub(crate) fn as_i64(value: &serde_json::Value, key: &str) -> i64 {
    value.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
}

/// Normalized usage entry shared across all vendors.
/// All vendor-specific data is normalized into this common format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEntry {
    pub host_id: Option<String>,
    /// Stable identifier of the source conversation when the harness exposes one.
    #[serde(default)]
    pub session_id: Option<String>,
    pub timestamp: String,
    pub parsed_timestamp: Option<DateTime<Local>>,
    pub session_start_time: String,
    pub session_end_time: String,
    pub model: String,
    pub effort: Option<String>,
    pub fast_tier: i8,
    pub usage: TokenUsage,
    pub costs: Option<UsageCost>,
}

/// Token usage counts for a single entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub reasoning_output_tokens: i64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_creation: f64,
}

/// Normalized usage entry paired with a stable source-level deduplication key.
#[derive(Debug, Clone)]
pub struct SourceUsageRecord {
    pub dedup_key: String,
    pub entry: UsageEntry,
}

/// Filter usage data to the selected local time window.
pub fn filter_usage_data_by_window(
    usage_data: &[UsageEntry],
    window: &TimeWindow,
    now: DateTime<Local>,
) -> Vec<UsageEntry> {
    filter_usage_data_by_window_and_session(usage_data, window, None, now)
}

/// Filter usage data to a time window and, when selected, one conversation.
pub fn filter_usage_data_by_window_and_session(
    usage_data: &[UsageEntry],
    window: &TimeWindow,
    session_id: Option<&str>,
    now: DateTime<Local>,
) -> Vec<UsageEntry> {
    if usage_data.is_empty() {
        return Vec::new();
    }

    let (start, end) = window.bounds(now);
    usage_data
        .par_iter()
        .with_min_len(4096)
        .filter_map(|entry| {
            let ts = entry
                .parsed_timestamp
                .or_else(|| parse_timestamp(&entry.timestamp))?;
            (ts >= start
                && ts <= end
                && session_id.is_none_or(|selected| entry.session_id.as_deref() == Some(selected)))
                .then(|| entry.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn entry_at(timestamp: DateTime<Local>, input_tokens: i64) -> UsageEntry {
        let timestamp = timestamp.to_rfc3339();
        UsageEntry {
            host_id: None,
            session_id: None,
            timestamp: timestamp.clone(),
            parsed_timestamp: parse_timestamp(&timestamp),
            session_start_time: timestamp.clone(),
            session_end_time: timestamp,
            model: "test-model".to_string(),
            effort: None,
            fast_tier: UNKNOWN_FAST_TIER,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 0,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_output_tokens: 0,
            },
            costs: None,
        }
    }

    #[test]
    fn fixed_window_filter_keeps_entries_inside_inclusive_bounds() {
        let window = TimeWindow::from_range("2026-05-01", "2026-05-07").expect("valid range");
        let (start, end) = window.bounds(Local::now());
        let usage = vec![
            entry_at(start - Duration::nanoseconds(1), 10),
            entry_at(start, 20),
            entry_at(end, 30),
            entry_at(end + Duration::nanoseconds(1), 40),
        ];

        let filtered = filter_usage_data_by_window(&usage, &window, Local::now());
        let tokens: Vec<i64> = filtered
            .iter()
            .map(|entry| entry.usage.input_tokens)
            .collect();

        assert_eq!(tokens, vec![20, 30]);
    }

    #[test]
    fn session_filter_keeps_only_the_requested_session_inside_the_window() {
        let window = TimeWindow::from_range("2026-05-01", "2026-05-07").expect("valid range");
        let (start, _) = window.bounds(Local::now());
        let mut first = entry_at(start, 10);
        first.session_id = Some("session-a".to_string());
        let mut second = entry_at(start, 20);
        second.session_id = Some("session-b".to_string());

        let filtered = filter_usage_data_by_window_and_session(
            &[first, second],
            &window,
            Some("session-b"),
            Local::now(),
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].usage.input_tokens, 20);
    }
}
