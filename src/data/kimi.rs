use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, Local};
use walkdir::WalkDir;

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry};
use crate::model_id::normalize_reasoning_effort;

/// Get the Kimi Code configuration directory.
pub fn get_kimi_dir() -> PathBuf {
    std::env::var("KIMI_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".kimi-code"))
                .unwrap_or_else(|_| PathBuf::from("~/.kimi-code"))
        })
}

fn as_i64(value: &serde_json::Value, key: &str) -> i64 {
    value.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
}

/// Strip the `kimi-code/` style provider prefix from a model alias.
fn normalize_model(raw: &str) -> &str {
    raw.split_once('/')
        .map_or(raw, |(_, model)| if model.is_empty() { raw } else { model })
}

/// Session and agent ids from a
/// `.../sessions/<workspace>/<session>/agents/<agent>/wire.jsonl` path.
fn session_agent_ids(path: &Path) -> Option<(String, String)> {
    let agent_dir = path.parent()?;
    let agents_dir = agent_dir.parent()?;
    if agents_dir.file_name()? != "agents" {
        return None;
    }
    let session_dir = agents_dir.parent()?;
    Some((
        session_dir.file_name()?.to_str()?.to_string(),
        agent_dir.file_name()?.to_str()?.to_string(),
    ))
}

fn read_single_kimi_file(path: &Path) -> Vec<SourceUsageRecord> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let session_agent = session_agent_ids(path);
    let mut records = Vec::new();
    let mut record_index = 0usize;
    let mut current_effort: Option<String> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let data: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match data.get("type").and_then(|v| v.as_str()) {
            Some("llm.request") => {
                if let Some(effort) = data
                    .get("thinkingEffort")
                    .and_then(|v| v.as_str())
                    .and_then(normalize_reasoning_effort)
                {
                    current_effort = Some(effort);
                }
                continue;
            }
            Some("usage.record") => {}
            _ => continue,
        }

        let usage = match data.get("usage") {
            Some(u) => u,
            None => continue,
        };
        let input_tokens = as_i64(usage, "inputOther");
        let output_tokens = as_i64(usage, "output");
        let cache_read = as_i64(usage, "inputCacheRead");
        let cache_creation = as_i64(usage, "inputCacheCreation");
        if input_tokens == 0 && output_tokens == 0 && cache_read == 0 && cache_creation == 0 {
            continue;
        }

        let model = data
            .get("model")
            .and_then(|v| v.as_str())
            .map_or("unknown", normalize_model);

        let time_ms = data.get("time").and_then(|v| v.as_i64()).unwrap_or(0);
        let parsed_timestamp =
            DateTime::from_timestamp_millis(time_ms).map(|dt| dt.with_timezone(&Local));
        let timestamp = parsed_timestamp
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default();

        let dedup_key = match &session_agent {
            Some((session, agent)) => {
                format!("kimi:{session}:{agent}:{time_ms}:{record_index}")
            }
            None => format!("kimi:file:{}:{record_index}", path.to_string_lossy()),
        };
        record_index += 1;

        records.push(SourceUsageRecord {
            dedup_key,
            entry: UsageEntry {
                host_id: None,
                timestamp: timestamp.clone(),
                parsed_timestamp,
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model: model.to_string(),
                effort: current_effort.clone(),
                fast_tier: UNKNOWN_FAST_TIER,
                usage: TokenUsage {
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens: cache_read,
                    cache_creation_input_tokens: cache_creation,
                    reasoning_output_tokens: 0,
                },
                costs: None,
            },
        });
    }

    records
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
            if e.file_name() != "wire.jsonl" {
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

pub fn read_kimi_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    read_single_kimi_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("ai-usage-kimi-test-{name}-{stamp}"))
    }

    fn write_wire_file(dir: &Path, content: &str) -> PathBuf {
        fs::create_dir_all(dir).expect("create wire dir");
        let path = dir.join("wire.jsonl");
        fs::write(&path, content).expect("write wire fixture");
        path
    }

    const WIRE_SAMPLE: &str = r#"{"type":"metadata","protocol_version":"1.4","created_at":1784329235246}
{"type":"llm.request","kind":"loop","provider":"kimi","model":"k3","modelAlias":"kimi-code/k3","thinkingEffort":"max","maxTokens":1048576,"turnStep":"0.1","time":1784329646154}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":2715,"output":117,"inputCacheRead":18944,"inputCacheCreation":0},"usageScope":"turn","time":1784329652715}
{"type":"llm.request","kind":"loop","provider":"kimi","model":"k3","modelAlias":"kimi-code/k3","thinkingEffort":"high","maxTokens":1048576,"turnStep":"0.2","time":1784329693000}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":650,"output":435,"inputCacheRead":21504,"inputCacheCreation":3},"usageScope":"turn","time":1784329693636}
"#;

    fn session_wire_path(root: &Path) -> PathBuf {
        root.join("wd_test_1")
            .join("session_abc")
            .join("agents")
            .join("agent-5")
    }

    #[test]
    fn parses_usage_records_with_effort_from_llm_request() {
        let root = unique_temp_dir("parse");
        let path = write_wire_file(&session_wire_path(&root), WIRE_SAMPLE);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);

        let first = &records[0].entry;
        assert_eq!(first.model, "k3");
        assert_eq!(first.effort.as_deref(), Some("max"));
        assert_eq!(first.usage.input_tokens, 2715);
        assert_eq!(first.usage.output_tokens, 117);
        assert_eq!(first.usage.cache_read_input_tokens, 18944);
        assert_eq!(first.usage.cache_creation_input_tokens, 0);
        assert_eq!(first.usage.reasoning_output_tokens, 0);
        assert!(first.costs.is_none());

        let expected = chrono::DateTime::from_timestamp_millis(1784329652715)
            .expect("valid epoch ms")
            .with_timezone(&chrono::Local);
        assert_eq!(first.parsed_timestamp, Some(expected));

        let second = &records[1].entry;
        assert_eq!(second.effort.as_deref(), Some("high"));
        assert_eq!(second.usage.cache_creation_input_tokens, 3);
    }

    #[test]
    fn skips_all_zero_and_malformed_lines() {
        let root = unique_temp_dir("skip");
        let content = r#"not json at all
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":0,"output":0,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329652715}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":5,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329652716}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 5);
    }

    #[test]
    fn dedup_keys_use_session_and_agent_path_segments() {
        let root = unique_temp_dir("dedup");
        let path = write_wire_file(&session_wire_path(&root), WIRE_SAMPLE);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].dedup_key,
            "kimi:session_abc:agent-5:1784329652715:0"
        );
        assert_eq!(
            records[1].dedup_key,
            "kimi:session_abc:agent-5:1784329693636:1"
        );
        let keys: HashSet<&str> = records.iter().map(|r| r.dedup_key.as_str()).collect();
        assert_eq!(keys.len(), records.len());
    }

    #[test]
    fn unexpected_layout_falls_back_to_file_position_keys() {
        let root = unique_temp_dir("fallback");
        let path = write_wire_file(&root, WIRE_SAMPLE);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .all(|r| r.dedup_key.starts_with("kimi:file:"))
        );
        let keys: HashSet<&str> = records.iter().map(|r| r.dedup_key.as_str()).collect();
        assert_eq!(keys.len(), records.len());
    }

    #[test]
    fn collect_usage_files_only_matches_wire_jsonl() {
        let root = unique_temp_dir("collect");
        let agent_dir = session_wire_path(&root);
        write_wire_file(&agent_dir, WIRE_SAMPLE);
        fs::write(agent_dir.join("other.jsonl"), "{}").expect("write other jsonl");

        let files = collect_usage_files(&root, None);
        fs::remove_dir_all(&root).ok();

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("wire.jsonl"));
    }
}
