pub mod cache;
pub mod claude;
pub mod codex;
pub mod gemini;

use crate::time_utils::{TimeWindow, parse_timestamp};
use chrono::{DateTime, Local};

/// Normalized usage entry shared across all vendors.
/// All vendor-specific data is normalized into this common format.
#[derive(Debug, Clone)]
pub struct UsageEntry {
    pub timestamp: String,
    pub parsed_timestamp: Option<DateTime<Local>>,
    pub session_start_time: String,
    pub session_end_time: String,
    pub model: String,
    pub effort: Option<String>,
    pub usage: TokenUsage,
}

/// Token usage counts for a single entry.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub reasoning_output_tokens: i64,
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
    if usage_data.is_empty() {
        return Vec::new();
    }

    let (start, end) = window.bounds(now);
    usage_data
        .iter()
        .filter_map(|entry| {
            let ts = entry
                .parsed_timestamp
                .or_else(|| parse_timestamp(&entry.timestamp))?;
            (ts >= start && ts <= end).then(|| entry.clone())
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
            timestamp: timestamp.clone(),
            parsed_timestamp: parse_timestamp(&timestamp),
            session_start_time: timestamp.clone(),
            session_end_time: timestamp,
            model: "test-model".to_string(),
            effort: None,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 0,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_output_tokens: 0,
            },
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
}
