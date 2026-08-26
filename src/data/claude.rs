use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::data::{
    SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry, file_fallback_key,
};
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

fn claude_response_key(message_id: &str) -> String {
    format!("claude:message:{message_id}")
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
    let mut response_positions = HashMap::new();
    for (line_index, line) in content.lines().enumerate() {
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

        let message_id = message.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let dedup_key = if message_id.is_empty() {
            file_fallback_key("claude", path, line_index)
        } else {
            claude_response_key(message_id)
        };

        let parsed_ts = parse_timestamp(&timestamp);
        let session_id = data
            .get("sessionId")
            .or_else(|| data.get("session_id"))
            .and_then(|v| v.as_str())
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .or_else(|| fallback_session_id.clone());

        // Claude reports cache creation by retention duration. Older
        // transcript records only expose the aggregate field; retain those as
        // five-minute writes so their cost remains reproducible.
        let cache_creation = usage.get("cache_creation");
        let cache_creation_5m_input_tokens = cache_creation
            .and_then(|value| value.get("ephemeral_5m_input_tokens"))
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        let cache_creation_1h_input_tokens = cache_creation
            .and_then(|value| value.get("ephemeral_1h_input_tokens"))
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        let legacy_cache_creation_input_tokens = usage
            .get("cache_creation_input_tokens")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        let cache_creation_input_tokens = if cache_creation
            .is_some_and(serde_json::Value::is_object)
            && (cache_creation_5m_input_tokens != 0
                || cache_creation_1h_input_tokens != 0
                || legacy_cache_creation_input_tokens == 0)
        {
            cache_creation_5m_input_tokens.saturating_add(cache_creation_1h_input_tokens)
        } else {
            legacy_cache_creation_input_tokens
        };

        let record = SourceUsageRecord {
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
                    cache_creation_input_tokens,
                    cache_creation_5m_input_tokens,
                    cache_creation_1h_input_tokens,
                    reasoning_output_tokens: 0,
                },
                costs: None,
            },
        };
        if message_id.is_empty() {
            entries.push(record);
        } else if let Some(index) = response_positions.get(message_id) {
            entries[*index] = record;
        } else {
            response_positions.insert(message_id.to_string(), entries.len());
            entries.push(record);
        }
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn read_fixture(name: &str, content: &str) -> Vec<SourceUsageRecord> {
        let path = std::env::temp_dir().join(format!(
            "ai-usage-claude-{name}-{}.jsonl",
            std::process::id()
        ));
        fs::write(&path, content).expect("write fixture");

        let records = read_jsonl_file_records(&path);
        fs::remove_file(&path).ok();
        records
    }

    #[test]
    fn usage_records_keep_the_claude_session_id() {
        let records = read_fixture(
            "session",
            r#"{"sessionId":"claude-session","timestamp":"2026-07-23T00:00:00Z","requestId":"request","message":{"id":"message","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
        );

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].entry.session_id.as_deref(),
            Some("claude-session")
        );
    }

    #[test]
    fn repeated_content_blocks_count_one_claude_response() {
        let records = read_fixture(
            "content-blocks",
            concat!(
                r#"{"sessionId":"session-a","timestamp":"2026-08-05T16:43:13Z","message":{"id":"message-a","model":"claude-first","usage":{"input_tokens":1,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":4}}}"#,
                "\n",
                r#"{"sessionId":"session-b","timestamp":"2026-08-05T16:43:14Z","message":{"id":"message-a","model":"claude-middle","usage":{"input_tokens":5,"output_tokens":5,"cache_read_input_tokens":6,"cache_creation_input_tokens":7}}}"#,
                "\n",
                r#"{"sessionId":"session-c","timestamp":"2026-08-05T16:43:15Z","message":{"id":"message-a","model":"claude-final","usage":{"input_tokens":9,"output_tokens":8,"cache_read_input_tokens":10,"cache_creation_input_tokens":11}}}"#,
            ),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dedup_key, "claude:message:message-a");
        assert_eq!(records[0].entry.session_id.as_deref(), Some("session-c"));
        assert_eq!(records[0].entry.timestamp, "2026-08-05T16:43:15Z");
        assert_eq!(records[0].entry.session_start_time, "2026-08-05T16:43:15Z");
        assert_eq!(records[0].entry.session_end_time, "2026-08-05T16:43:15Z");
        assert_eq!(records[0].entry.model, "claude-final");
        assert_eq!(records[0].entry.usage.input_tokens, 9);
        assert_eq!(records[0].entry.usage.output_tokens, 8);
        assert_eq!(records[0].entry.usage.cache_read_input_tokens, 10);
        assert_eq!(records[0].entry.usage.cache_creation_input_tokens, 11);
    }

    #[test]
    fn duration_specific_cache_writes_are_preserved() {
        let records = read_fixture(
            "cache-durations",
            r#"{"timestamp":"2026-08-05T16:43:15Z","message":{"id":"message-a","model":"claude-fable-5","usage":{"input_tokens":1,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":7}}}}"#,
        );

        assert_eq!(records.len(), 1);
        let usage = &records[0].entry.usage;
        assert_eq!(usage.cache_creation_5m_input_tokens, 4);
        assert_eq!(usage.cache_creation_1h_input_tokens, 7);
        assert_eq!(usage.cache_creation_input_tokens, 11);
    }

    #[test]
    fn empty_cache_creation_object_keeps_legacy_aggregate() {
        let records = read_fixture(
            "cache-duration-fallback",
            r#"{"timestamp":"2026-08-05T16:43:15Z","message":{"id":"message-a","model":"claude-fable-5","usage":{"cache_creation_input_tokens":11,"cache_creation":{}}}}"#,
        );

        assert_eq!(records.len(), 1);
        let usage = &records[0].entry.usage;
        assert_eq!(usage.cache_creation_input_tokens, 11);
        assert_eq!(usage.cache_creation_5m_input_tokens, 0);
        assert_eq!(usage.cache_creation_1h_input_tokens, 0);
    }

    #[test]
    fn request_id_does_not_split_one_claude_response() {
        let records = read_fixture(
            "request-id",
            concat!(
                r#"{"timestamp":"2026-08-05T16:43:13Z","requestId":"request-a","message":{"id":"message-a","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
                "\n",
                r#"{"timestamp":"2026-08-05T16:43:14Z","requestId":"request-b","message":{"id":"message-a","model":"claude-test","usage":{"input_tokens":1,"output_tokens":5}}}"#,
                "\n",
                r#"{"timestamp":"2026-08-05T16:43:15Z","message":{"id":"message-a","model":"claude-test","usage":{"input_tokens":1,"output_tokens":8}}}"#,
            ),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dedup_key, "claude:message:message-a");
        assert_eq!(records[0].entry.usage.output_tokens, 8);
    }

    #[test]
    fn repeated_response_keeps_final_timestamp_across_dst_fallback() {
        let records = read_fixture(
            "dst-fallback",
            concat!(
                r#"{"timestamp":"2026-11-01T08:59:59Z","message":{"id":"message-a","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
                "\n",
                r#"{"timestamp":"2026-11-01T09:00:01Z","message":{"id":"message-a","model":"claude-test","usage":{"input_tokens":1,"output_tokens":8}}}"#,
            ),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.timestamp, "2026-11-01T09:00:01Z");
        assert_eq!(records[0].entry.usage.output_tokens, 8);
        assert_eq!(
            records[0]
                .entry
                .parsed_timestamp
                .expect("parsed final timestamp")
                .with_timezone(&Utc),
            chrono::DateTime::parse_from_rfc3339("2026-11-01T09:00:01Z")
                .expect("fixed timestamp")
                .with_timezone(&Utc)
        );
    }

    #[test]
    fn distinct_message_ids_remain_distinct() {
        let records = read_fixture(
            "distinct-ids",
            concat!(
                r#"{"timestamp":"2026-08-05T16:43:13Z","message":{"id":"message-a","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
                "\n",
                r#"{"timestamp":"2026-08-05T16:43:14Z","message":{"id":"message-b","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
            ),
        );

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].dedup_key, "claude:message:message-a");
        assert_eq!(records[1].dedup_key, "claude:message:message-b");
    }

    #[test]
    fn missing_message_ids_use_host_scoped_file_line_keys() {
        let records = read_fixture(
            "missing-ids",
            concat!(
                r#"{"timestamp":"2026-08-05T16:43:13Z","message":{"model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
                "\n",
                r#"{"timestamp":"2026-08-05T16:43:14Z","message":{"id":"","model":"claude-test","usage":{"input_tokens":1,"output_tokens":2}}}"#,
            ),
        );

        assert_eq!(records.len(), 2);
        assert!(records[0].dedup_key.starts_with("claude:file:"));
        assert!(records[0].dedup_key.ends_with(":0"));
        assert!(records[1].dedup_key.starts_with("claude:file:"));
        assert!(records[1].dedup_key.ends_with(":1"));
        assert_ne!(records[0].dedup_key, records[1].dedup_key);
    }
}
