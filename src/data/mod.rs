pub mod claude;
pub mod codex;
pub mod gemini;

use chrono::{DateTime, Duration, Local};
use crate::time_utils::parse_timestamp;

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
    pub vendor: &'static str,
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

impl TokenUsage {
    /// Total I/O tokens (input + output, no cache).
    pub fn io_total(&self) -> i64 {
        self.input_tokens + self.output_tokens
    }
}

/// Filter usage data to only include entries from the last N days.
/// Uses the latest timestamp in the data as the reference point.
pub fn filter_usage_data_by_days(usage_data: &[UsageEntry], days_back: i64) -> Vec<UsageEntry> {
    if usage_data.is_empty() {
        return Vec::new();
    }

    let mut latest_time: Option<DateTime<Local>> = None;
    let mut entries_with_ts: Vec<(&UsageEntry, DateTime<Local>)> = Vec::new();

    for entry in usage_data {
        let ts = entry
            .parsed_timestamp
            .or_else(|| parse_timestamp(&entry.timestamp));

        if let Some(ts) = ts {
            entries_with_ts.push((entry, ts));
            if latest_time.is_none() || ts > latest_time.unwrap() {
                latest_time = Some(ts);
            }
        }
    }

    let latest_time = match latest_time {
        Some(t) => t,
        None => return usage_data.to_vec(),
    };

    let start_time = latest_time - Duration::days(days_back);

    entries_with_ts
        .into_iter()
        .filter(|(_, ts)| *ts >= start_time)
        .map(|(entry, _)| entry.clone())
        .collect()
}
