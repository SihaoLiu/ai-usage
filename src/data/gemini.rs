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

/// Read a single Gemini session JSON file, returning each entry with a
/// stable dedup key derived from the session ID and message ID.
fn read_single_gemini_file(path: &Path) -> Vec<SourceUsageRecord> {
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

    let session_id = data
        .get("sessionId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut records = Vec::new();
    let mut message_index = 0usize;

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

        let message_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let dedup_key = build_dedup_key(&session_id, message_id, path, message_index);
        message_index += 1;

        let total_input = tokens.get("input").and_then(|v| v.as_i64()).unwrap_or(0);
        let cached_input = tokens.get("cached").and_then(|v| v.as_i64()).unwrap_or(0);
        let output_tokens = tokens.get("output").and_then(|v| v.as_i64()).unwrap_or(0);
        let thoughts_tokens = tokens.get("thoughts").and_then(|v| v.as_i64()).unwrap_or(0);

        let non_cached_input = total_input - cached_input;
        let parsed_ts = parse_timestamp(&timestamp);

        records.push(SourceUsageRecord {
            dedup_key,
            entry: UsageEntry {
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
            },
        });
    }

    records
}

fn build_dedup_key(
    session_id: &str,
    message_id: &str,
    path: &Path,
    message_index: usize,
) -> String {
    // The session_id + message_id pair is globally unique when both are
    // present (Gemini writes UUIDs for both). When either is missing in
    // an older or malformed session file, fall back to the file path
    // plus the message's position so the key stays stable across reruns
    // of the same machine.
    if !session_id.is_empty() && !message_id.is_empty() {
        format!("gemini:{session_id}:{message_id}")
    } else if !message_id.is_empty() {
        format!("gemini:msg:{message_id}")
    } else {
        let path_repr = path.to_string_lossy();
        format!("gemini:file:{path_repr}:{message_index}")
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vibe-usage-gemini-test-{name}-{stamp}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_session(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, body).expect("write session file");
    }

    #[test]
    fn session_messages_get_session_and_message_id_dedup_keys() {
        let dir = unique_temp_dir("session-ids");
        let path = dir.join("session-x.json");
        write_session(
            &path,
            r#"{
                "sessionId": "sess-uuid-1",
                "messages": [
                    {"id":"u","timestamp":"2026-05-19T00:00:00Z","type":"user","content":"hi"},
                    {"id":"m1","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-2.5","tokens":{"input":10,"cached":2,"output":3,"thoughts":1}},
                    {"id":"m2","timestamp":"2026-05-19T00:00:02Z","type":"gemini","model":"gemini-2.5","tokens":{"input":11,"cached":0,"output":4,"thoughts":0}}
                ]
            }"#,
        );

        let records = read_gemini_file_records(&path);

        let keys: Vec<&str> = records.iter().map(|r| r.dedup_key.as_str()).collect();
        assert_eq!(keys, ["gemini:sess-uuid-1:m1", "gemini:sess-uuid-1:m2"]);
    }

    #[test]
    fn reparsing_unchanged_session_produces_identical_dedup_keys() {
        let dir = unique_temp_dir("stable-keys");
        let path = dir.join("session-y.json");
        write_session(
            &path,
            r#"{
                "sessionId": "sess-uuid-2",
                "messages": [
                    {"id":"a","timestamp":"2026-05-19T00:00:00Z","type":"gemini","model":"g","tokens":{"input":1,"cached":0,"output":1,"thoughts":0}},
                    {"id":"b","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"g","tokens":{"input":2,"cached":0,"output":2,"thoughts":0}}
                ]
            }"#,
        );

        let first: Vec<String> = read_gemini_file_records(&path)
            .iter()
            .map(|r| r.dedup_key.clone())
            .collect();
        let second: Vec<String> = read_gemini_file_records(&path)
            .iter()
            .map(|r| r.dedup_key.clone())
            .collect();

        assert_eq!(first, second);
    }

    #[test]
    fn dedup_keys_are_unique_within_a_session() {
        let dir = unique_temp_dir("unique-keys");
        let path = dir.join("session-z.json");
        write_session(
            &path,
            r#"{
                "sessionId": "sess-uuid-3",
                "messages": [
                    {"id":"a","timestamp":"2026-05-19T00:00:00Z","type":"gemini","model":"g","tokens":{"input":1,"cached":0,"output":1,"thoughts":0}},
                    {"id":"b","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"g","tokens":{"input":2,"cached":0,"output":2,"thoughts":0}}
                ]
            }"#,
        );

        let records = read_gemini_file_records(&path);
        let unique: HashSet<&str> = records.iter().map(|r| r.dedup_key.as_str()).collect();

        assert_eq!(unique.len(), records.len());
    }

    #[test]
    fn missing_message_id_falls_back_to_file_path_and_index() {
        let dir = unique_temp_dir("fallback-key");
        let path = dir.join("session-w.json");
        write_session(
            &path,
            r#"{
                "messages": [
                    {"timestamp":"2026-05-19T00:00:00Z","type":"gemini","model":"g","tokens":{"input":1,"cached":0,"output":1,"thoughts":0}}
                ]
            }"#,
        );

        let records = read_gemini_file_records(&path);

        assert_eq!(records.len(), 1);
        let key = records[0].dedup_key.as_str();
        assert!(key.starts_with("gemini:file:"), "key = {key:?}");
        assert!(key.ends_with(":0"), "key = {key:?}");
        assert!(!key.is_empty());
    }

    #[test]
    fn missing_session_id_still_yields_dedup_key_from_message_id() {
        let dir = unique_temp_dir("msg-only-key");
        let path = dir.join("session-v.json");
        write_session(
            &path,
            r#"{
                "messages": [
                    {"id":"only-msg","timestamp":"2026-05-19T00:00:00Z","type":"gemini","model":"g","tokens":{"input":1,"cached":0,"output":1,"thoughts":0}}
                ]
            }"#,
        );

        let records = read_gemini_file_records(&path);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dedup_key, "gemini:msg:only-msg");
    }
}
