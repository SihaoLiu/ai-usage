use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::data::{
    SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry, file_fallback_key,
    has_jsonl_extension,
};
use crate::time_utils::parse_timestamp;

/// Get the Gemini configuration directory.
pub fn get_gemini_dir() -> PathBuf {
    crate::data::config_dir("GEMINI_CONFIG_DIR", ".gemini")
}

/// Read one legacy session or append log, returning each entry with a stable
/// dedup key derived from the session ID and message ID.
fn read_single_gemini_file(path: &Path) -> Vec<SourceUsageRecord> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    if has_jsonl_extension(path) {
        let (session_id, messages) = decode_append_log(&content);
        let records = normalize_messages(path, &session_id, &messages);
        return retain_last_by_identity(records);
    }

    let (session_id, messages) = match decode_legacy_session(&content) {
        Some(decoded) => decoded,
        None => return Vec::new(),
    };
    normalize_messages(path, &session_id, &messages)
}

type PositionedMessage = (usize, serde_json::Value);

fn decode_legacy_session(content: &str) -> Option<(String, Vec<PositionedMessage>)> {
    let data: serde_json::Value = serde_json::from_str(content).ok()?;

    let session_id = data
        .get("sessionId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let messages = data
        .get("messages")?
        .as_array()?
        .iter()
        .cloned()
        .enumerate()
        .collect();

    Some((session_id, messages))
}

fn decode_append_log(content: &str) -> (String, Vec<PositionedMessage>) {
    let session_id = content
        .lines()
        .next()
        .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .and_then(|header| {
            header
                .get("sessionId")
                .and_then(|id| id.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    let values = content
        .lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            serde_json::from_str(line)
                .ok()
                .map(|value| (line_index, value))
        })
        .collect();

    (session_id, values)
}

fn normalize_messages(
    path: &Path,
    session_id: &str,
    messages: &[PositionedMessage],
) -> Vec<SourceUsageRecord> {
    let mut records = Vec::new();

    for (message_index, msg) in messages {
        if let Some(record) = normalize_message(path, session_id, msg, *message_index) {
            records.push(record);
        }
    }

    records
}

fn normalize_message(
    path: &Path,
    session_id: &str,
    message: &serde_json::Value,
    message_index: usize,
) -> Option<SourceUsageRecord> {
    (message.get("type")?.as_str()? == "gemini").then_some(())?;
    let tokens = message.get("tokens")?;
    let timestamp = message.get("timestamp")?.as_str()?.to_string();
    let model = message
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
        .to_string();
    let message_id = message
        .get("id")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let dedup_key = build_dedup_key(session_id, message_id, path, message_index);

    let total_input = required_token_field(tokens, "input")?;
    let cached_input = optional_token_field(tokens, "cached")?;
    let output_tokens = required_token_field(tokens, "output")?;
    let thoughts_tokens = optional_token_field(tokens, "thoughts")?;
    let tool_tokens = optional_token_field(tokens, "tool")?;
    if cached_input > total_input {
        return None;
    }
    let non_cached_input = total_input
        .checked_sub(cached_input)?
        .checked_add(tool_tokens)?;
    let computed_total = total_input
        .checked_add(output_tokens)?
        .checked_add(thoughts_tokens)?
        .checked_add(tool_tokens)?;
    if let Some(wire_total) = present_token_field(tokens, "total")?
        && wire_total != computed_total
    {
        return None;
    }

    Some(SourceUsageRecord {
        dedup_key,
        entry: UsageEntry {
            host_id: None,
            session_id: (!session_id.is_empty()).then(|| session_id.to_string()),
            timestamp: timestamp.clone(),
            parsed_timestamp: parse_timestamp(&timestamp),
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
                cache_creation_5m_input_tokens: 0,
                cache_creation_1h_input_tokens: 0,
                reasoning_output_tokens: 0,
            },
            costs: None,
        },
    })
}

fn required_token_field(value: &serde_json::Value, name: &str) -> Option<i64> {
    present_token_field(value, name)?.filter(|token_count| *token_count >= 0)
}

fn optional_token_field(value: &serde_json::Value, name: &str) -> Option<i64> {
    present_token_field(value, name).map(|value| value.unwrap_or(0))
}

fn present_token_field(value: &serde_json::Value, name: &str) -> Option<Option<i64>> {
    match value.get(name) {
        Some(value) => value
            .as_i64()
            .filter(|token_count| *token_count >= 0)
            .map(Some),
        None => Some(None),
    }
}

fn retain_last_by_identity(records: Vec<SourceUsageRecord>) -> Vec<SourceUsageRecord> {
    let mut positions = HashMap::new();
    let mut canonical = Vec::new();

    for record in records {
        if let Some(position) = positions.get(&record.dedup_key).copied() {
            canonical[position] = record;
        } else {
            positions.insert(record.dedup_key.clone(), canonical.len());
            canonical.push(record);
        }
    }

    canonical
}

fn build_dedup_key(
    session_id: &str,
    message_id: &str,
    path: &Path,
    message_index: usize,
) -> String {
    // The session_id + message_id pair is globally unique when both are
    // present (Gemini writes UUIDs for both). When either is missing in
    // an older or malformed session file, fall back to the file path digest
    // plus the message's position so the key stays stable across reruns
    // of the same machine.
    if !session_id.is_empty() && !message_id.is_empty() {
        format!("gemini:{session_id}:{message_id}")
    } else {
        file_fallback_key("gemini", path, message_index)
    }
}

pub fn collect_usage_files(tmp_dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    crate::data::collect_recent_files(tmp_dir, max_age_days, |path| {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let parent_name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("");
        parent_name == "chats"
            && file_name.starts_with("session-")
            && (file_name.ends_with(".json") || file_name.ends_with(".jsonl"))
    })
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
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("temp")
            .join(format!("ai-usage-gemini-test-{name}-{stamp}"));
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
    fn collector_discovers_legacy_and_append_log_sessions_only() {
        let dir = unique_temp_dir("collector-formats");
        let chats = dir.join("project").join("chats");
        let legacy = chats.join("session-a.json");
        let append_log = chats.join("session-b.jsonl");

        for path in [
            &legacy,
            &append_log,
            &chats.join("session-c.txt"),
            &chats.join("other.json"),
            &dir.join("project").join("session-d.jsonl"),
        ] {
            write_session(path, "{}");
        }

        assert_eq!(collect_usage_files(&dir, None), vec![legacy, append_log]);
    }

    #[test]
    fn append_log_keeps_last_record_for_each_response_in_first_seen_order() {
        let dir = unique_temp_dir("append-log-supersession");
        let path = dir.join("session-a.jsonl");
        write_session(
            &path,
            concat!(
                r#"{"sessionId":"session-a","projectHash":"project","startTime":"2026-05-19T00:00:00Z","lastUpdated":"2026-05-19T00:00:03Z","kind":"main"}"#,
                "\n",
                r#"{"id":"user-a","timestamp":"2026-05-19T00:00:00Z","type":"user","content":"hello"}"#,
                "\n",
                r#"{"$set":{"lastUpdated":"2026-05-19T00:00:01Z"}}"#,
                "\n",
                r#"{"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-first","tokens":{"input":10,"output":1,"cached":2,"thoughts":3,"tool":0,"total":14}}"#,
                "\n",
                r#"{"id":"response-b","timestamp":"2026-05-19T00:00:02Z","type":"gemini","model":"gemini-middle","tokens":{"input":4,"output":2,"cached":1,"thoughts":0,"tool":0,"total":6}}"#,
                "\n",
                r#"{"$set":{"lastUpdated":"2026-05-19T00:00:03Z"}}"#,
                "\n",
                r#"{"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-final","tokens":{"input":20,"output":4,"cached":5,"thoughts":6,"tool":0,"total":30},"toolCalls":[]}"#,
                "\n",
            ),
        );

        let records = read_gemini_file_records(&path);

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].dedup_key, "gemini:session-a:response-a");
        assert_eq!(records[0].entry.model, "gemini-final");
        assert_eq!(records[0].entry.usage.input_tokens, 15);
        assert_eq!(records[1].dedup_key, "gemini:session-a:response-b");
    }

    #[test]
    fn legacy_and_append_log_share_session_message_identity() {
        let dir = unique_temp_dir("mixed-schema-key");
        let legacy_path = dir.join("session-a.json");
        let append_log_path = dir.join("session-a.jsonl");
        write_session(
            &legacy_path,
            r#"{
                "sessionId": "session-a",
                "messages": [
                    {"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-legacy","tokens":{"input":5,"cached":1,"output":2,"thoughts":0}}
                ]
            }"#,
        );
        write_session(
            &append_log_path,
            concat!(
                r#"{"sessionId":"session-a","projectHash":"project","startTime":"2026-05-19T00:00:00Z","lastUpdated":"2026-05-19T00:00:01Z","kind":"main"}"#,
                "\n",
                r#"{"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-append","tokens":{"input":5,"cached":1,"output":2,"thoughts":0,"tool":0,"total":7}}"#,
                "\n",
            ),
        );

        let legacy = read_gemini_file_records(&legacy_path);
        let append_log = read_gemini_file_records(&append_log_path);

        assert_eq!(legacy.len(), 1);
        assert_eq!(append_log.len(), 1);
        assert_eq!(legacy[0].dedup_key, append_log[0].dedup_key);
    }

    #[test]
    fn tool_tokens_are_non_cached_input_and_preserve_wire_total() {
        let dir = unique_temp_dir("tool-token-accounting");
        let path = dir.join("session-a.jsonl");
        write_session(
            &path,
            concat!(
                r#"{"sessionId":"session-a","projectHash":"project","startTime":"2026-05-19T00:00:00Z","lastUpdated":"2026-05-19T00:00:01Z","kind":"main"}"#,
                "\n",
                r#"{"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-model","tokens":{"input":100,"cached":40,"output":7,"thoughts":11,"tool":13,"total":131}}"#,
                "\n",
            ),
        );

        let records = read_gemini_file_records(&path);

        assert_eq!(records.len(), 1);
        let usage = &records[0].entry.usage;
        assert_eq!(usage.input_tokens, 73);
        assert_eq!(usage.cache_read_input_tokens, 40);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_creation_input_tokens, 11);
        assert_eq!(
            usage.input_tokens
                + usage.cache_read_input_tokens
                + usage.output_tokens
                + usage.cache_creation_input_tokens,
            131
        );
    }

    #[test]
    fn append_log_skips_malformed_lines_and_keeps_missing_id_fallbacks() {
        let dir = unique_temp_dir("append-log-fallbacks");
        let path = dir.join("session-a.jsonl");
        write_session(
            &path,
            concat!(
                r#"{"sessionId":"session-a","projectHash":"project","startTime":"2026-05-19T00:00:00Z","lastUpdated":"2026-05-19T00:00:03Z","kind":"main"}"#,
                "\n",
                "not-json\n",
                r#"{"timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-a","tokens":{"input":1,"cached":0,"output":2,"thoughts":0,"tool":0,"total":3}}"#,
                "\n",
                r#"{"id":"incomplete","type":"gemini","model":"ignored","tokens":{"input":100,"cached":0,"output":0,"thoughts":0,"tool":0,"total":100}}"#,
                "\n",
                r#"{"timestamp":"2026-05-19T00:00:02Z","type":"gemini","model":"gemini-b","tokens":{"input":3,"cached":0,"output":4,"thoughts":0,"tool":0,"total":7}}"#,
                "\n",
            ),
        );

        let first = read_gemini_file_records(&path);
        let second = read_gemini_file_records(&path);

        assert_eq!(first.len(), 2);
        assert_eq!(
            first[0].dedup_key,
            crate::data::file_fallback_key("gemini", &path, 2)
        );
        assert_eq!(
            first[1].dedup_key,
            crate::data::file_fallback_key("gemini", &path, 4)
        );
        assert_eq!(first[0].dedup_key, second[0].dedup_key);
        assert_eq!(first[1].dedup_key, second[1].dedup_key);
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
        assert_eq!(records[0].entry.session_id.as_deref(), Some("sess-uuid-1"));
        assert_eq!(records[0].entry.usage.input_tokens, 8);
        assert_eq!(records[0].entry.usage.cache_read_input_tokens, 2);
        assert_eq!(records[0].entry.usage.output_tokens, 3);
        assert_eq!(records[0].entry.usage.cache_creation_input_tokens, 1);
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
    fn missing_session_id_uses_a_host_scoped_file_key() {
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
        assert_eq!(
            records[0].dedup_key,
            crate::data::file_fallback_key("gemini", &path, 0)
        );
        assert!(!ai_usage_proto::is_globally_stable_usage_key(
            "gemini",
            &records[0].dedup_key
        ));
    }

    #[test]
    fn append_log_only_accepts_session_identity_from_the_first_physical_header() {
        let dir = unique_temp_dir("physical-header");
        let path = dir.join("session-a.jsonl");
        write_session(
            &path,
            concat!(
                r#"{"kind":"main","projectHash":"project"}"#,
                "\n",
                r#"{"sessionId":"late-session","id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"gemini-model","tokens":{"input":5,"cached":1,"output":2,"thoughts":0,"tool":0,"total":7}}"#,
                "\n",
            ),
        );

        let records = read_gemini_file_records(&path);

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].dedup_key,
            crate::data::file_fallback_key("gemini", &path, 1)
        );
        assert_eq!(records[0].entry.session_id, None);
    }

    #[test]
    fn malformed_superseding_tokens_do_not_replace_a_valid_response() {
        let dir = unique_temp_dir("malformed-superseding-tokens");
        let path = dir.join("session-a.jsonl");
        write_session(
            &path,
            concat!(
                r#"{"sessionId":"session-a","kind":"main"}"#,
                "\n",
                r#"{"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"valid-model","tokens":{"input":10,"cached":2,"output":3,"thoughts":1,"tool":0,"total":14}}"#,
                "\n",
                r#"{"id":"response-a","timestamp":"2026-05-19T00:00:01Z","type":"gemini","model":"invalid-model","tokens":{"input":10,"cached":20,"output":3,"thoughts":1,"tool":0,"total":14}}"#,
                "\n",
                r#"{"id":"negative","timestamp":"2026-05-19T00:00:02Z","type":"gemini","model":"invalid-model","tokens":{"input":-1,"cached":0,"output":0,"thoughts":0,"tool":0,"total":-1}}"#,
                "\n",
                r#"{"id":"overflow","timestamp":"2026-05-19T00:00:03Z","type":"gemini","model":"invalid-model","tokens":{"input":9223372036854775808,"cached":0,"output":0,"thoughts":0,"tool":0,"total":0}}"#,
                "\n",
                r#"{"id":"wrong-total","timestamp":"2026-05-19T00:00:04Z","type":"gemini","model":"invalid-model","tokens":{"input":1,"cached":0,"output":2,"thoughts":3,"tool":4,"total":99}}"#,
                "\n",
            ),
        );

        let records = read_gemini_file_records(&path);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dedup_key, "gemini:session-a:response-a");
        assert_eq!(records[0].entry.model, "valid-model");
        assert_eq!(records[0].entry.usage.input_tokens, 8);
    }
}
