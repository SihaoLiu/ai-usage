use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use walkdir::WalkDir;

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry};
use crate::time_utils::parse_timestamp;

/// Get the Gemini configuration directory.
pub fn get_gemini_dir() -> PathBuf {
    std::env::var("GEMINI_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".gemini"))
                .unwrap_or_else(|_| PathBuf::from("~/.gemini"))
        })
}

/// Read a single Gemini session JSON file.
fn read_single_gemini_file(path: &Path) -> Vec<UsageEntry> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let data: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let messages = match data.get("messages").and_then(|v| v.as_array()) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let mut entries = Vec::new();

    for msg in messages {
        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if msg_type != "gemini" {
            continue;
        }

        let tokens = match msg.get("tokens") {
            Some(t) => t,
            None => continue,
        };

        let timestamp = match msg.get("timestamp").and_then(|v| v.as_str()) {
            Some(ts) => ts.to_string(),
            None => continue,
        };

        let model = msg
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let total_input = tokens.get("input").and_then(|v| v.as_i64()).unwrap_or(0);
        let cached_input = tokens.get("cached").and_then(|v| v.as_i64()).unwrap_or(0);
        let output_tokens = tokens.get("output").and_then(|v| v.as_i64()).unwrap_or(0);
        let thoughts_tokens = tokens.get("thoughts").and_then(|v| v.as_i64()).unwrap_or(0);

        let non_cached_input = total_input - cached_input;
        let parsed_ts = parse_timestamp(&timestamp);

        entries.push(UsageEntry {
            host_id: None,
            timestamp: timestamp.clone(),
            parsed_timestamp: parsed_ts,
            session_start_time: timestamp.clone(),
            session_end_time: timestamp,
            model,
            effort: None,
            fast_tier: UNKNOWN_FAST_TIER,
            usage: TokenUsage {
                input_tokens: non_cached_input,
                output_tokens,
                cache_read_input_tokens: cached_input,
                cache_creation_input_tokens: thoughts_tokens,
                reasoning_output_tokens: 0,
            },
        });
    }

    entries
}

pub fn collect_usage_files(tmp_dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    if !tmp_dir.exists() {
        return Vec::new();
    }

    let cutoff = max_age_days
        .map(|days| SystemTime::now() - std::time::Duration::from_secs((days as u64 + 1) * 86400));

    let mut files: Vec<PathBuf> = WalkDir::new(tmp_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            if !e.file_type().is_file() {
                return false;
            }
            let path = e.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let parent_name = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if !(parent_name == "chats"
                && file_name.starts_with("session-")
                && file_name.ends_with(".json"))
            {
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

pub fn read_gemini_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    read_single_gemini_file(path)
        .into_iter()
        .map(|entry| SourceUsageRecord {
            dedup_key: String::new(),
            entry,
        })
        .collect()
}
