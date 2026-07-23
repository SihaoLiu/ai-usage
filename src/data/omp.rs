use std::fs;
use std::path::{Path, PathBuf};

use crate::data::{
    SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageCost, UsageEntry, as_i64,
};
use crate::model_id::normalize_reasoning_effort;
use crate::time_utils::parse_timestamp;

/// Get the Oh My Pi configuration directory.
pub fn get_omp_dir() -> PathBuf {
    crate::data::config_dir("OMP_CONFIG_DIR", ".omp")
}

fn normalize_model(raw: &str) -> (&str, Option<&str>) {
    raw.split_once('/')
        .map_or((raw, None), |(provider, model)| {
            if model.is_empty() {
                (raw, None)
            } else {
                (model, Some(provider))
            }
        })
}

fn non_empty_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
}

fn explicit_reasoning_effort(
    usage: &serde_json::Value,
    message: &serde_json::Value,
) -> Option<String> {
    ["effort", "reasoningEffort", "reasoning_effort"]
        .into_iter()
        .filter_map(|key| non_empty_str(usage, key).or_else(|| non_empty_str(message, key)))
        .find_map(normalize_reasoning_effort)
}

fn read_single_omp_file(path: &Path) -> Vec<SourceUsageRecord> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let session_id = session_id_from_content(&content).or_else(|| {
        path.file_stem()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .map(str::to_string)
    });
    let mut records = Vec::new();
    let mut message_index = 0usize;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let data: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if data.get("type").and_then(|v| v.as_str()) != Some("message") {
            continue;
        }

        let message = match data.get("message") {
            Some(m) => m,
            None => continue,
        };
        if message.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }

        let usage = match message.get("usage") {
            Some(u) => u,
            None => continue,
        };

        let raw_model = usage
            .get("model")
            .or_else(|| message.get("model"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let (model, _) = normalize_model(raw_model);
        let effort = explicit_reasoning_effort(usage, message);

        let input_tokens = as_i64(usage, "input");
        let output_tokens = as_i64(usage, "output");
        let cache_read = as_i64(usage, "cacheRead");
        let cache_write = as_i64(usage, "cacheWrite");
        let costs = usage.get("cost").map(|cost| UsageCost {
            input: cost.get("input").and_then(|v| v.as_f64()).unwrap_or(0.0),
            output: cost.get("output").and_then(|v| v.as_f64()).unwrap_or(0.0),
            cache_read: cost
                .get("cacheRead")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            cache_creation: cost
                .get("cacheWrite")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        });
        if input_tokens == 0 && output_tokens == 0 && cache_read == 0 && cache_write == 0 {
            continue;
        }

        let timestamp = data
            .get("timestamp")
            .or_else(|| message.get("timestamp"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let parsed_timestamp = parse_timestamp(&timestamp);

        let response_id = message
            .get("responseId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let message_id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let dedup_key = build_dedup_key(message_id, response_id, path, message_index);
        message_index += 1;

        records.push(SourceUsageRecord {
            dedup_key,
            entry: UsageEntry {
                host_id: None,
                session_id: session_id.clone(),
                timestamp: timestamp.clone(),
                parsed_timestamp,
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model: model.to_string(),
                effort,
                fast_tier: UNKNOWN_FAST_TIER,
                usage: TokenUsage {
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens: cache_read,
                    cache_creation_input_tokens: cache_write,
                    reasoning_output_tokens: 0,
                },
                costs,
            },
        });
    }

    records
}

fn session_id_from_content(content: &str) -> Option<String> {
    content.lines().find_map(|line| {
        let data: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
        (data.get("type").and_then(|value| value.as_str()) == Some("session"))
            .then_some(&data)
            .and_then(|session| {
                session
                    .get("id")
                    .or_else(|| session.get("sessionId"))
                    .or_else(|| session.get("session_id"))
            })
            .and_then(|id| id.as_str())
            .filter(|id| !id.is_empty())
            .map(str::to_string)
    })
}

fn build_dedup_key(
    message_id: &str,
    response_id: &str,
    path: &Path,
    message_index: usize,
) -> String {
    if !message_id.is_empty() && !response_id.is_empty() {
        format!("omp:message:{message_id}:response:{response_id}")
    } else if !response_id.is_empty() {
        format!("omp:response:{response_id}")
    } else if !message_id.is_empty() {
        format!("omp:message:{message_id}")
    } else {
        format!("omp:file:{}:{message_index}", path.to_string_lossy())
    }
}

pub fn collect_usage_files(sessions_dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    crate::data::collect_recent_files(sessions_dir, max_age_days, crate::data::has_jsonl_extension)
}

pub fn read_omp_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    read_single_omp_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("ai-usage-omp-test-{name}-{stamp}.jsonl"))
    }

    #[test]
    fn parses_assistant_usage_from_message_lines() {
        let path = unique_temp_file("assistant-usage");
        fs::write(
            &path,
            r#"{"type":"session","timestamp":"2026-05-27T08:00:00Z"}
{"type":"message","id":"msg-a","timestamp":"2026-05-27T08:34:02.431Z","message":{"role":"assistant","api":"openai-codex-responses","provider":"openai-codex","model":"openai-codex/gpt-5.5","usage":{"input":917,"output":27,"cacheRead":20224,"cacheWrite":3,"totalTokens":21171,"cost":{"input":0.004585,"output":0.00081,"cacheRead":0.010112,"cacheWrite":0.000001,"total":0.015508}},"responseId":"resp-a"}}
{"type":"message","id":"msg-b","timestamp":"2026-05-27T08:35:02.431Z","message":{"role":"user","content":[]}}
"#,
        )
        .expect("write omp fixture");

        let records = read_omp_file_records(&path);
        fs::remove_file(&path).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.model, "gpt-5.5");
        assert_eq!(records[0].entry.effort.as_deref(), None);
        assert_eq!(records[0].entry.usage.input_tokens, 917);
        assert_eq!(records[0].entry.usage.output_tokens, 27);
        assert_eq!(records[0].entry.usage.cache_read_input_tokens, 20224);
        assert_eq!(records[0].entry.usage.cache_creation_input_tokens, 3);
        let costs = records[0].entry.costs.expect("costs parsed");
        assert!((costs.input - 0.004585).abs() < f64::EPSILON);
        assert!((costs.output - 0.00081).abs() < f64::EPSILON);
        assert!((costs.cache_read - 0.010112).abs() < f64::EPSILON);
        assert!((costs.cache_creation - 0.000001).abs() < f64::EPSILON);
    }

    #[test]
    fn session_metadata_id_is_used_for_usage_records() {
        let content = r#"{"type":"session","id":"omp-session"}
{"type":"message","timestamp":"2026-05-27T08:34:02.431Z","message":{"role":"assistant","model":"gpt-test","usage":{"input":1,"output":2}}}"#;
        assert_eq!(session_id_from_content(content).as_deref(), Some("omp-session"));
    }

    #[test]
    fn missing_ids_fall_back_to_unique_file_position_keys() {
        let path = unique_temp_file("missing-ids");
        fs::write(
            &path,
            r#"{"type":"message","timestamp":"2026-05-27T08:34:02Z","message":{"role":"assistant","provider":"openai-codex","model":"openai-codex/gpt-5.5","usage":{"input":10,"output":2,"cacheRead":3,"cacheWrite":4}}}
{"type":"message","timestamp":"2026-05-27T08:35:02Z","message":{"role":"assistant","provider":"openai-codex","model":"openai-codex/gpt-5.5","usage":{"input":10,"output":2,"cacheRead":3,"cacheWrite":4}}}
"#,
        )
        .expect("write omp fixture");

        let records = read_omp_file_records(&path);
        fs::remove_file(&path).ok();

        assert_eq!(records.len(), 2);
        let keys: HashSet<&str> = records
            .iter()
            .map(|record| record.dedup_key.as_str())
            .collect();
        assert_eq!(keys.len(), records.len());
        assert!(
            records
                .iter()
                .all(|record| record.dedup_key.starts_with("omp:file:"))
        );
    }

    #[test]
    fn provider_fields_do_not_become_effort() {
        let path = unique_temp_file("explicit-provider");
        fs::write(
            &path,
            r#"{"type":"message","id":"msg-a","timestamp":"2026-05-27T08:34:02Z","message":{"role":"assistant","provider":"anthropic","model":"claude-sonnet-4-5-20250929","usage":{"input":10,"output":2,"cacheRead":3,"cacheWrite":4},"responseId":"resp-a"}}
{"type":"message","id":"msg-b","timestamp":"2026-05-27T08:35:02Z","message":{"role":"assistant","model":"gemini-2.5-pro","usage":{"provider":"google","input":20,"output":5,"cacheRead":0,"cacheWrite":0},"responseId":"resp-b"}}
"#,
        )
        .expect("write omp fixture");

        let records = read_omp_file_records(&path);
        fs::remove_file(&path).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].entry.model, "claude-sonnet-4-5-20250929");
        assert_eq!(records[0].entry.effort.as_deref(), None);
        assert_eq!(records[1].entry.model, "gemini-2.5-pro");
        assert_eq!(records[1].entry.effort.as_deref(), None);
    }

    #[test]
    fn explicit_reasoning_effort_survives_endpoint_provider() {
        let path = unique_temp_file("reasoning-effort");
        fs::write(
            &path,
            r#"{"type":"message","id":"msg-a","timestamp":"2026-05-27T08:34:02Z","message":{"role":"assistant","provider":"rust-cat","model":"rust-cat/gpt-5","usage":{"effort":"xhigh","input":10,"output":2,"cacheRead":3,"cacheWrite":4,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"responseId":"resp-a"}}
"#,
        )
        .expect("write omp fixture");

        let records = read_omp_file_records(&path);
        fs::remove_file(&path).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.model, "gpt-5");
        assert_eq!(records[0].entry.effort.as_deref(), Some("xhigh"));
    }
}
