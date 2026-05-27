use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use walkdir::WalkDir;

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageCost, UsageEntry};
use crate::time_utils::parse_timestamp;

/// Get the Oh My Pi configuration directory.
pub fn get_omp_dir() -> PathBuf {
    std::env::var("OMP_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".omp"))
                .unwrap_or_else(|_| PathBuf::from("~/.omp"))
        })
}

fn as_i64(value: &serde_json::Value, key: &str) -> i64 {
    value.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
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

fn read_single_omp_file(path: &Path) -> Vec<SourceUsageRecord> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

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
        let (model, provider_from_model) = normalize_model(raw_model);
        let provider = provider_from_model
            .or_else(|| non_empty_str(usage, "provider"))
            .or_else(|| non_empty_str(message, "provider"));

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
                timestamp: timestamp.clone(),
                parsed_timestamp,
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model: model.to_string(),
                effort: provider.map(str::to_string),
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
    if !sessions_dir.exists() {
        return Vec::new();
    }

    let cutoff = max_age_days
        .map(|days| SystemTime::now() - std::time::Duration::from_secs((days as u64 + 1) * 86400));

    let mut files: Vec<PathBuf> = WalkDir::new(sessions_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            if !e.file_type().is_file() {
                return false;
            }
            if e.path().extension().is_none_or(|ext| ext != "jsonl") {
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
        std::env::temp_dir().join(format!("vibe-usage-omp-test-{name}-{stamp}.jsonl"))
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
        assert_eq!(records[0].entry.effort.as_deref(), Some("openai-codex"));
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
    fn uses_explicit_provider_when_model_has_no_prefix() {
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
        assert_eq!(records[0].entry.effort.as_deref(), Some("anthropic"));
        assert_eq!(records[1].entry.model, "gemini-2.5-pro");
        assert_eq!(records[1].entry.effort.as_deref(), Some("google"));
    }
}
