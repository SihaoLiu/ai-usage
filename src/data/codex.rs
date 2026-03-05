use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rayon::prelude::*;
use walkdir::WalkDir;

use crate::data::{TokenUsage, UsageEntry};
use crate::time_utils::parse_timestamp;

/// Get the Codex configuration directory.
pub fn get_codex_dir() -> PathBuf {
    std::env::var("CODEX_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".codex"))
                .unwrap_or_else(|_| PathBuf::from("~/.codex"))
        })
}

/// Read a single Codex JSONL file with two-pass parsing.
fn read_single_codex_file(path: &Path) -> Vec<UsageEntry> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    // First pass: collect all entries and find session start time
    let mut session_start_time: Option<String> = None;
    let mut all_timestamps: Vec<String> = Vec::new();
    let mut file_entries: Vec<serde_json::Value> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let data: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let entry_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let timestamp = data
            .get("timestamp")
            .and_then(|v| v.as_str())
            .map(String::from);

        if let Some(ref ts) = timestamp {
            all_timestamps.push(ts.clone());
        }

        if entry_type == "session_meta" {
            let payload = data.get("payload");
            if let Some(payload_ts) = payload
                .and_then(|p| p.get("timestamp"))
                .and_then(|v| v.as_str())
            {
                session_start_time = Some(payload_ts.to_string());
            } else if let Some(ref ts) = timestamp {
                session_start_time = Some(ts.clone());
            }
        }

        file_entries.push(data);
    }

    // Fallback: use first timestamp
    if session_start_time.is_none() && !all_timestamps.is_empty() {
        session_start_time = all_timestamps.into_iter().min();
    }

    // Second pass: process entries
    let mut current_model = "unknown".to_string();
    let mut current_effort = "unknown".to_string();
    let mut last_token_usage: Option<(i64, i64, i64, i64)> = None;
    let mut result = Vec::new();

    for data in &file_entries {
        let entry_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if entry_type == "turn_context" {
            if let Some(payload) = data.get("payload") {
                if let Some(model) = payload.get("model").and_then(|v| v.as_str()) {
                    current_model = model.to_string();
                }
                if let Some(effort) = payload.get("effort").and_then(|v| v.as_str()) {
                    current_effort = effort.to_string();
                }
            }
        } else if entry_type == "event_msg" {
            if let Some(payload) = data.get("payload") {
                let payload_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if payload_type != "token_count" {
                    continue;
                }

                let info = match payload.get("info") {
                    Some(i) => i,
                    None => continue,
                };
                let token_usage = match info.get("last_token_usage") {
                    Some(t) => t,
                    None => continue,
                };

                let input_tokens = token_usage
                    .get("input_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let cached_input = token_usage
                    .get("cached_input_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let output_tokens = token_usage
                    .get("output_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let reasoning_output = token_usage
                    .get("reasoning_output_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);

                // Skip duplicates
                let usage_key = (input_tokens, cached_input, output_tokens, reasoning_output);
                if Some(usage_key) == last_token_usage {
                    continue;
                }
                last_token_usage = Some(usage_key);

                // Normalize: input_tokens is TOTAL, cached is subset
                let non_cached_input = input_tokens - cached_input;
                let non_reasoning_output = output_tokens - reasoning_output;

                let entry_timestamp = data
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let parsed_ts = parse_timestamp(&entry_timestamp);

                result.push(UsageEntry {
                    timestamp: entry_timestamp.clone(),
                    parsed_timestamp: parsed_ts,
                    session_start_time: entry_timestamp.clone(),
                    session_end_time: entry_timestamp,
                    model: current_model.clone(),
                    effort: Some(current_effort.clone()),
                    vendor: "codex",
                    usage: TokenUsage {
                        input_tokens: non_cached_input,
                        output_tokens: non_reasoning_output,
                        cache_read_input_tokens: cached_input,
                        cache_creation_input_tokens: 0,
                        reasoning_output_tokens: reasoning_output,
                    },
                });
            }
        }
    }

    result
}

/// Read all Codex JSONL files from the sessions directory.
pub fn read_codex_jsonl_files(sessions_dir: &Path, max_age_days: Option<i64>) -> Vec<UsageEntry> {
    if !sessions_dir.exists() {
        return Vec::new();
    }

    let cutoff = max_age_days.map(|days| {
        SystemTime::now() - std::time::Duration::from_secs((days as u64 + 1) * 86400)
    });

    let mut files: Vec<PathBuf> = WalkDir::new(sessions_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            if !e.file_type().is_file() {
                return false;
            }
            if !e.path().extension().is_some_and(|ext| ext == "jsonl") {
                return false;
            }
            if let Some(cutoff_time) = cutoff {
                if let Ok(meta) = e.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        return mtime >= cutoff_time;
                    }
                }
            }
            true
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    files.sort();

    let all_entries: Vec<Vec<UsageEntry>> = files
        .par_iter()
        .map(|path| read_single_codex_file(path))
        .collect();

    all_entries.into_iter().flatten().collect()
}
