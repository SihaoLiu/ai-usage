use std::fs;
use std::path::{Path, PathBuf};

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry};
use crate::time_utils::parse_timestamp;

/// Get Claude configuration directories.
/// Supports CLAUDE_CONFIG_DIR env var (comma-separated) or defaults to
/// ~/.config/claude and ~/.claude.
pub fn get_claude_dirs() -> Vec<PathBuf> {
    if let Ok(env_val) = std::env::var("CLAUDE_CONFIG_DIR") {
        return env_val
            .split(',')
            .map(|p| PathBuf::from(p.trim()))
            .filter(|p| !p.as_os_str().is_empty())
            .collect();
    }

    let home = dirs_home();
    vec![home.join(".config/claude"), home.join(".claude")]
}

pub fn detect_fast_tier_snapshot() -> i8 {
    get_claude_dirs()
        .into_iter()
        .find_map(|dir| {
            parse_fast_mode_setting(&fs::read_to_string(dir.join("settings.json")).ok()?)
        })
        .map(|enabled| if enabled { 1 } else { 0 })
        .unwrap_or(0)
}

fn parse_fast_mode_setting(content: &str) -> Option<bool> {
    let json = serde_json::from_str::<serde_json::Value>(content).ok()?;
    json.get("fastMode").and_then(|value| value.as_bool())
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("~"))
}

/// Collect all JSONL file paths under the given directory,
/// optionally filtering by mtime (files modified within `max_age_days`).
fn collect_jsonl_files(dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    crate::data::collect_recent_files(dir, max_age_days, crate::data::has_jsonl_extension)
}

pub fn collect_usage_files(max_age_days: Option<i64>) -> Vec<PathBuf> {
    let dirs = get_claude_dirs();
    let mut all_files: Vec<PathBuf> = Vec::new();

    for dir in &dirs {
        let projects_dir = dir.join("projects");
        if projects_dir.exists() {
            all_files.extend(collect_jsonl_files(&projects_dir, max_age_days));
        }
    }

    all_files.sort();
    all_files
}

/// Read a single JSONL file and return entries with deduplication keys.
pub fn read_jsonl_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let fallback_session_id = path
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string);
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let data: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let message = match data.get("message") {
            Some(m) => m,
            None => continue,
        };
        let usage = match message.get("usage") {
            Some(u) => u,
            None => continue,
        };

        let timestamp = data
            .get("timestamp")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let model = message
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Build dedup key from message_id:request_id
        let message_id = message.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let request_id = data.get("requestId").and_then(|v| v.as_str()).unwrap_or("");

        let dedup_key = if !message_id.is_empty() && !request_id.is_empty() {
            format!("{}:{}", message_id, request_id)
        } else {
            String::new()
        };

        let parsed_ts = parse_timestamp(&timestamp);
        let session_id = data
            .get("sessionId")
            .or_else(|| data.get("session_id"))
            .and_then(|v| v.as_str())
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .or_else(|| fallback_session_id.clone());

        entries.push(SourceUsageRecord {
            dedup_key,
            entry: UsageEntry {
                host_id: None,
                session_id,
                timestamp: timestamp.clone(),
                parsed_timestamp: parsed_ts,
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model,
                effort: None,
                fast_tier: UNKNOWN_FAST_TIER,
                usage: TokenUsage {
                    input_tokens: usage
                        .get("input_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    output_tokens: usage
                        .get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    cache_read_input_tokens: usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    cache_creation_input_tokens: usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    reasoning_output_tokens: 0,
                },
                costs: None,
            },
        });
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_records_keep_the_claude_session_id() {
        let path = std::env::temp_dir().join("ai-usage-claude-session-test.jsonl");
        fs::write(
            &path,
            r#"{"sessionId":"claude-session","timestamp":"2026-07-23T00:00:00Z","requestId":"request","message":{"id":"message","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
        )
        .expect("write fixture");

        let records = read_jsonl_file_records(&path);
        fs::remove_file(&path).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].entry.session_id.as_deref(),
            Some("claude-session")
        );
    }
}
