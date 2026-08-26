use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};

use crate::data::{
    SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry, file_fallback_key,
};
use crate::model_id::normalize_reasoning_effort;

/// Get the Kimi Code configuration directory.
pub fn get_kimi_dir() -> PathBuf {
    crate::data::config_dir("KIMI_CONFIG_DIR", ".kimi-code")
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

fn non_empty_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(|field| field.as_str())
        .filter(|field| !field.is_empty())
}

fn accepts_usage_scope(data: &serde_json::Value) -> bool {
    match data.get("usageScope") {
        None => true,
        Some(serde_json::Value::String(scope)) => matches!(scope.as_str(), "turn" | "session"),
        Some(_) => false,
    }
}

fn request_step_id(data: &serde_json::Value) -> Option<String> {
    non_empty_str(data, "turnStep").map(str::to_string)
}

fn completion_step_id(event: &serde_json::Value) -> Option<String> {
    let turn_id = non_empty_str(event, "turnId")?;
    let step = event.get("step")?.as_i64().filter(|step| *step >= 0)?;
    Some(format!("{turn_id}.{step}"))
}

fn fallback_dedup_key(path: &Path, time_ms: i64, line_index: usize) -> String {
    file_fallback_key("kimi", path, format!("{time_ms}:{line_index}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KimiUsageCounts {
    input: i64,
    output: i64,
    cache_read: i64,
    cache_creation: i64,
}

impl KimiUsageCounts {
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_object()?;
        Some(Self {
            input: non_negative_token_field(value, "inputOther")?,
            output: non_negative_token_field(value, "output")?,
            cache_read: non_negative_token_field(value, "inputCacheRead")?,
            cache_creation: non_negative_token_field(value, "inputCacheCreation")?,
        })
    }

    fn is_zero(self) -> bool {
        self.input == 0 && self.output == 0 && self.cache_read == 0 && self.cache_creation == 0
    }

    fn into_token_usage(self) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input,
            output_tokens: self.output,
            cache_read_input_tokens: self.cache_read,
            cache_creation_input_tokens: self.cache_creation,
            cache_creation_5m_input_tokens: 0,
            cache_creation_1h_input_tokens: 0,
            reasoning_output_tokens: 0,
        }
    }
}

fn non_negative_token_field(value: &serde_json::Value, name: &str) -> Option<i64> {
    value
        .get(name)?
        .as_i64()
        .filter(|token_count| *token_count >= 0)
}

enum RequestUsage {
    Pending,
    Rejected,
    Accepted {
        record_index: usize,
        counts: KimiUsageCounts,
    },
}

struct PendingRequest {
    request_kind: Option<String>,
    step_id: Option<String>,
    request_model: Option<String>,
    effort: Option<String>,
    completion_id: Option<String>,
    completion_usage: Option<KimiUsageCounts>,
    usage: RequestUsage,
}

impl PendingRequest {
    fn from_json(data: &serde_json::Value) -> Self {
        Self {
            request_kind: non_empty_str(data, "kind").map(str::to_string),
            step_id: request_step_id(data),
            request_model: non_empty_str(data, "modelAlias")
                .or_else(|| non_empty_str(data, "model"))
                .map(normalize_model)
                .map(str::to_string),
            effort: data
                .get("thinkingEffort")
                .and_then(|value| value.as_str())
                .and_then(normalize_reasoning_effort),
            completion_id: None,
            completion_usage: None,
            usage: RequestUsage::Pending,
        }
    }

    fn is_loop(&self) -> bool {
        self.request_kind.as_deref() == Some("loop")
    }

    fn is_complete(&self) -> bool {
        !matches!(self.usage, RequestUsage::Pending)
            && (!self.is_loop() || self.completion_id.is_some())
    }
}

#[derive(Default)]
struct ResponseAssociation {
    requests: VecDeque<PendingRequest>,
    seen_completion_ids: HashSet<String>,
}

impl ResponseAssociation {
    fn begin_request(&mut self, data: &serde_json::Value) {
        self.prune_completed();
        self.requests.retain(|request| {
            !matches!(request.usage, RequestUsage::Pending) || request.completion_id.is_some()
        });
        self.requests.push_back(PendingRequest::from_json(data));
    }

    fn apply_completion(
        &mut self,
        response_id: String,
        completion_step: Option<String>,
        completion_usage: Option<KimiUsageCounts>,
        records: &mut [SourceUsageRecord],
    ) {
        if !self.seen_completion_ids.insert(response_id.clone()) {
            return;
        }
        self.prune_completed();
        let semantic = completion_step.as_ref().and_then(|step_id| {
            self.requests.iter().rposition(|request| {
                request.is_loop()
                    && request.completion_id.is_none()
                    && request.step_id.as_ref() == Some(step_id)
            })
        });
        let may_use_legacy_fallback = |request: &&PendingRequest| {
            request.is_loop()
                && request.completion_id.is_none()
                && (completion_step.is_none() || request.step_id.is_none())
        };
        let exact = completion_usage.and_then(|counts| {
            self.requests.iter().rposition(|request| {
                may_use_legacy_fallback(&request)
                    && matches!(
                        request.usage,
                        RequestUsage::Accepted {
                            counts: usage_counts,
                            ..
                        } if usage_counts == counts
                    )
            })
        });
        let awaiting_usage = self.requests.iter().rposition(|request| {
            may_use_legacy_fallback(&request) && matches!(request.usage, RequestUsage::Pending)
        });
        let position = semantic.or(exact).or(awaiting_usage).or_else(|| {
            self.requests
                .iter()
                .rposition(|request| may_use_legacy_fallback(&request))
        });
        let Some(position) = position else {
            return;
        };
        let request = self.requests.get_mut(position).expect("request position");
        if let RequestUsage::Accepted {
            record_index,
            counts,
        } = request.usage
            && completion_usage.is_none_or(|completion_counts| completion_counts == counts)
            && let Some(record) = records.get_mut(record_index)
        {
            record.dedup_key = format!("kimi:response:{response_id}");
        }
        request.completion_id = Some(response_id);
        request.completion_usage = completion_usage;
        self.prune_completed();
    }

    fn reject_usage(&mut self, usage: Option<KimiUsageCounts>) {
        self.prune_completed();
        let exact = usage.and_then(|counts| {
            self.requests.iter().rposition(|request| {
                matches!(request.usage, RequestUsage::Pending)
                    && request.completion_usage == Some(counts)
            })
        });
        let position = exact.or_else(|| {
            self.requests
                .iter()
                .rposition(|request| matches!(request.usage, RequestUsage::Pending))
        });
        if let Some(position) = position {
            self.requests
                .get_mut(position)
                .expect("request position")
                .usage = RequestUsage::Rejected;
        }
        self.prune_completed();
    }

    fn key_for_usage(
        &mut self,
        counts: KimiUsageCounts,
        fallback: String,
        record_index: usize,
    ) -> (String, Option<String>, Option<String>) {
        self.prune_completed();
        let exact = self.requests.iter().rposition(|request| {
            matches!(request.usage, RequestUsage::Pending)
                && request.completion_usage == Some(counts)
        });
        let without_completion = self.requests.iter().rposition(|request| {
            matches!(request.usage, RequestUsage::Pending) && request.completion_id.is_none()
        });
        let position = exact.or(without_completion).or_else(|| {
            self.requests
                .iter()
                .rposition(|request| matches!(request.usage, RequestUsage::Pending))
        });
        let Some(position) = position else {
            return (fallback, None, None);
        };
        let request = self.requests.get_mut(position).expect("request position");
        let key = match (&request.completion_id, request.completion_usage) {
            (Some(response_id), None) => format!("kimi:response:{response_id}"),
            (Some(response_id), Some(completion_counts)) if completion_counts == counts => {
                format!("kimi:response:{response_id}")
            }
            _ => fallback,
        };
        let model = request.request_model.clone();
        let effort = request.effort.clone();
        request.usage = RequestUsage::Accepted {
            record_index,
            counts,
        };
        self.prune_completed();
        (key, model, effort)
    }

    fn prune_completed(&mut self) {
        self.requests.retain(|request| !request.is_complete());
    }
}

fn usage_time(data: &serde_json::Value) -> Option<(i64, DateTime<Local>)> {
    let time_ms = data
        .get("time")?
        .as_i64()
        .filter(|milliseconds| *milliseconds > 0)?;
    let parsed = DateTime::from_timestamp_millis(time_ms)?.with_timezone(&Local);
    Some((time_ms, parsed))
}

fn response_id(event: &serde_json::Value) -> Option<String> {
    non_empty_str(event, "messageId")
        .or_else(|| non_empty_str(event, "uuid"))
        .map(str::to_string)
}

fn is_usage_line(line: &str) -> bool {
    line.contains("usage.record")
}

fn is_relevant_line(line: &str) -> bool {
    is_usage_line(line)
        || line.contains("llm.request")
        || line.contains("context.append_loop_event") && line.contains("step.end")
}

fn append_usage_record(
    records: &mut Vec<SourceUsageRecord>,
    association: &mut ResponseAssociation,
    path: &Path,
    session_agent: &Option<(String, String)>,
    line_index: usize,
    data: &serde_json::Value,
    counts: KimiUsageCounts,
) {
    let Some((time_ms, parsed_timestamp)) = usage_time(data) else {
        association.reject_usage(Some(counts));
        return;
    };
    let fallback = fallback_dedup_key(path, time_ms, line_index);
    let (dedup_key, request_model, effort) =
        association.key_for_usage(counts, fallback, records.len());
    let timestamp = parsed_timestamp.to_rfc3339();
    records.push(SourceUsageRecord {
        dedup_key,
        entry: UsageEntry {
            host_id: None,
            session_id: session_agent.as_ref().map(|(session, _)| session.clone()),
            timestamp: timestamp.clone(),
            parsed_timestamp: Some(parsed_timestamp),
            session_start_time: timestamp.clone(),
            session_end_time: timestamp,
            model: request_model.unwrap_or_else(|| {
                non_empty_str(data, "model")
                    .map(normalize_model)
                    .unwrap_or("unknown")
                    .to_string()
            }),
            effort,
            fast_tier: UNKNOWN_FAST_TIER,
            usage: counts.into_token_usage(),
            costs: None,
        },
    });
}

fn read_single_kimi_file(path: &Path) -> Vec<SourceUsageRecord> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let session_agent = session_agent_ids(path);
    let mut records = Vec::new();
    let mut association = ResponseAssociation::default();
    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Wire files are dominated by multi-kilobyte context/content lines;
        // only completion events can contribute response identity.
        if !is_relevant_line(line) {
            continue;
        }

        let data: serde_json::Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                if is_usage_line(line) {
                    association.reject_usage(None);
                }
                continue;
            }
        };
        match data.get("type").and_then(|v| v.as_str()) {
            Some("llm.request") => {
                association.begin_request(&data);
                continue;
            }
            Some("context.append_loop_event") => {
                let Some(event) = data.get("event").filter(|event| {
                    event.get("type").and_then(|value| value.as_str()) == Some("step.end")
                }) else {
                    continue;
                };
                if let Some(response_id) = response_id(event) {
                    association.apply_completion(
                        response_id,
                        completion_step_id(event),
                        event.get("usage").and_then(KimiUsageCounts::from_json),
                        &mut records,
                    );
                }
                continue;
            }
            Some("usage.record") => {}
            _ => continue,
        }
        let counts = data.get("usage").and_then(KimiUsageCounts::from_json);
        if !accepts_usage_scope(&data) {
            association.reject_usage(counts);
            continue;
        }
        let Some(counts) = counts else {
            association.reject_usage(None);
            continue;
        };
        if counts.is_zero() {
            association.reject_usage(Some(counts));
            continue;
        }
        append_usage_record(
            &mut records,
            &mut association,
            path,
            &session_agent,
            line_index,
            &data,
            counts,
        );
    }

    records
}

pub fn collect_usage_files(sessions_dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    crate::data::collect_recent_files(sessions_dir, max_age_days, |path| {
        path.file_name().is_some_and(|name| name == "wire.jsonl")
    })
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
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("temp")
            .join(format!("ai-usage-kimi-test-{name}-{stamp}"))
    }

    fn write_wire_file(dir: &Path, content: &str) -> PathBuf {
        fs::create_dir_all(dir).expect("create wire dir");
        let path = dir.join("wire.jsonl");
        fs::write(&path, content).expect("write wire fixture");
        path
    }

    const WIRE_SAMPLE: &str = r#"{"type":"metadata","protocol_version":"1.4","created_at":1784329235246}
{"type":"llm.request","kind":"legacy","provider":"kimi","model":"k3","modelAlias":"kimi-code/k3","thinkingEffort":"max","maxTokens":1048576,"turnStep":"0.1","time":1784329646154}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":2715,"output":117,"inputCacheRead":18944,"inputCacheCreation":0},"usageScope":"turn","time":1784329652715}
{"type":"llm.request","kind":"legacy","provider":"kimi","model":"k3","modelAlias":"kimi-code/k3","thinkingEffort":"high","maxTokens":1048576,"turnStep":"0.2","time":1784329693000}
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
        assert_eq!(records[0].entry.session_id.as_deref(), Some("session_abc"));

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
    fn effort_resets_when_a_later_request_has_no_thinking_effort() {
        let root = unique_temp_dir("effort-reset");
        let content = r#"{"type":"llm.request","thinkingEffort":"max","time":1784329646154}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"llm.request","time":1784329646300}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":20,"output":2,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
{"type":"llm.request","thinkingEffort":"off","time":1784329646500}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":30,"output":3,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646600}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].entry.effort.as_deref(), Some("max"));
        assert_eq!(records[1].entry.effort.as_deref(), None);
        assert_eq!(records[2].entry.effort.as_deref(), None);
    }

    #[test]
    fn response_usage_scopes_accept_turn_session_and_legacy() {
        let root = unique_temp_dir("scope");
        let content = r#"{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":20,"output":2,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"session","time":1784329646300}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":30,"output":3,"inputCacheRead":0,"inputCacheCreation":0},"time":1784329646400}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":999,"output":999,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"account","time":1784329646500}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":999,"output":999,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"","time":1784329646550}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":999,"output":999,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":7,"time":1784329646600}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .map(|record| record.entry.usage.input_tokens)
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn step_end_before_usage_supplies_the_response_key() {
        let root = unique_temp_dir("event-before-usage");
        let content = r#"{"type":"llm.request","kind":"loop","model":"k3","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646100}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-before","messageId":"response-before","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646200}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dedup_key, "kimi:response:response-before");
    }

    #[test]
    fn usage_before_step_end_uses_the_event_uuid_when_message_id_is_absent() {
        let root = unique_temp_dir("usage-before-event");
        let content = r#"{"type":"llm.request","kind":"loop","model":"k3","modelAlias":"kimi-code/k3","thinkingEffort":"high","time":1784329646100}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"context.append_loop_event","event":{"type":"content.part","uuid":"part-a"},"time":1784329646200}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"response-after","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646200}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dedup_key, "kimi:response:response-after");
    }

    #[test]
    fn idless_compaction_uses_request_metadata_and_raw_line_fallback() {
        let root = unique_temp_dir("compaction-fallback");
        let content = r#"{"type":"metadata","protocol_version":"1.4"}
{"type":"llm.request","kind":"compaction","model":"request-model","modelAlias":"kimi-code/request-model","thinkingEffort":"high","time":1784329646100}
{"type":"usage.record","model":"kimi-code/usage-model","usage":{"inputOther":10,"output":1,"inputCacheRead":200,"inputCacheCreation":0},"usageScope":"session","time":1784329646200}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.model, "request-model");
        assert_eq!(records[0].entry.effort.as_deref(), Some("high"));
        assert_eq!(
            records[0].dedup_key,
            crate::data::file_fallback_key("kimi", &path, "1784329646200:2")
        );
    }

    #[test]
    fn inserting_compaction_does_not_rename_loop_responses() {
        let root = unique_temp_dir("compaction-insertion");
        let baseline = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646100}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-a","messageId":"response-a"},"time":1784329646200}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646300}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-b","messageId":"response-b"},"time":1784329646400}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":20,"output":2,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
"#;
        let with_compaction = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646100}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-a","messageId":"response-a"},"time":1784329646200}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"llm.request","kind":"compaction","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646250}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":30,"output":3,"inputCacheRead":100,"inputCacheCreation":0},"usageScope":"session","time":1784329646251}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646300}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-b","messageId":"response-b"},"time":1784329646400}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":20,"output":2,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
"#;
        let baseline_path = write_wire_file(&session_wire_path(&root).join("baseline"), baseline);
        let expanded_path =
            write_wire_file(&session_wire_path(&root).join("expanded"), with_compaction);

        let baseline_records = read_kimi_file_records(&baseline_path);
        let expanded_records = read_kimi_file_records(&expanded_path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(
            baseline_records
                .iter()
                .map(|record| record.dedup_key.as_str())
                .collect::<Vec<_>>(),
            vec!["kimi:response:response-a", "kimi:response:response-b"]
        );
        assert_eq!(expanded_records.len(), 3);
        assert_eq!(expanded_records[0].dedup_key, baseline_records[0].dedup_key);
        assert_eq!(expanded_records[2].dedup_key, baseline_records[1].dedup_key);
    }

    #[test]
    fn retries_only_emit_completed_usage_with_distinct_response_keys() {
        let root = unique_temp_dir("retries");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"low","time":1784329646000}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"high","time":1784329646100}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-a","messageId":"retry-response-a"},"time":1784329646200}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"max","time":1784329646300}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":20,"output":2,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
{"type":"context.append_loop_event","event":{"type":"step.end","uuid":"event-b","messageId":"retry-response-b"},"time":1784329646400}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/k3","thinkingEffort":"low","time":1784329646500}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].dedup_key, "kimi:response:retry-response-a");
        assert_eq!(records[0].entry.effort.as_deref(), Some("high"));
        assert_eq!(records[1].dedup_key, "kimi:response:retry-response-b");
        assert_eq!(records[1].entry.effort.as_deref(), Some("max"));
    }

    #[test]
    fn delayed_completion_across_a_request_boundary_keeps_both_response_identities() {
        let root = unique_temp_dir("delayed-completion");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-b","thinkingEffort":"max","time":1784329646200}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646300}
{"type":"usage.record","model":"kimi-code/model-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0}},"time":1784329646500}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].dedup_key, "kimi:response:response-a");
        assert_eq!(records[0].entry.model, "model-a");
        assert_eq!(records[1].dedup_key, "kimi:response:response-b");
        assert_eq!(records[1].entry.model, "model-b");
    }

    #[test]
    fn semantic_step_identity_disambiguates_equal_token_counts() {
        let root = unique_temp_dir("semantic-step-identity");
        let content = r#"{"type":"llm.request","kind":"loop","turnStep":"0.1","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"llm.request","kind":"loop","turnStep":"0.2","modelAlias":"kimi-code/model-b","thinkingEffort":"max","time":1784329646200}
{"type":"usage.record","model":"kimi-code/model-b","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646300}
{"type":"context.append_loop_event","event":{"type":"step.end","turnId":"0","step":2,"messageId":"response-b","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646400}
{"type":"context.append_loop_event","event":{"type":"step.end","turnId":"0","step":1,"messageId":"response-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646500}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].dedup_key, "kimi:response:response-a");
        assert_eq!(records[0].entry.model, "model-a");
        assert_eq!(records[1].dedup_key, "kimi:response:response-b");
        assert_eq!(records[1].entry.model, "model-b");
    }

    #[test]
    fn same_step_retry_associates_completion_with_the_newest_request() {
        let root = unique_temp_dir("same-step-retry");
        let content = r#"{"type":"llm.request","kind":"loop","turnStep":"0.1","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"llm.request","kind":"loop","turnStep":"0.1","modelAlias":"kimi-code/model-b","thinkingEffort":"max","time":1784329646200}
{"type":"context.append_loop_event","event":{"type":"step.end","turnId":"0","step":1,"messageId":"response-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0}},"time":1784329646300}
{"type":"usage.record","model":"kimi-code/model-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert!(records[0].dedup_key.starts_with("kimi:file:sha256:"));
        assert_eq!(records[0].entry.model, "model-a");
        assert_eq!(records[1].dedup_key, "kimi:response:response-b");
        assert_eq!(records[1].entry.model, "model-b");
    }

    #[test]
    fn legacy_equal_count_retry_associates_completion_with_the_newest_request() {
        let root = unique_temp_dir("legacy-equal-count-retry");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-b","thinkingEffort":"max","time":1784329646200}
{"type":"usage.record","model":"kimi-code/model-b","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646300}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-b","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646400}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_ne!(records[0].dedup_key, "kimi:response:response-b");
        assert_eq!(records[0].entry.model, "model-a");
        assert_eq!(records[1].dedup_key, "kimi:response:response-b");
        assert_eq!(records[1].entry.model, "model-b");
    }

    #[test]
    fn usage_without_completion_keeps_its_fallback_identity() {
        let root = unique_temp_dir("incomplete-usage-retry");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"low","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-b","thinkingEffort":"high","time":1784329646200}
{"type":"usage.record","model":"kimi-code/model-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646300}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0}},"time":1784329646400}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert!(records[0].dedup_key.starts_with("kimi:file:sha256:"));
        assert_eq!(records[0].entry.model, "model-a");
        assert_eq!(records[1].dedup_key, "kimi:response:response-b");
        assert_eq!(records[1].entry.model, "model-b");
    }

    #[test]
    fn rejected_usage_consumes_a_pending_completion() {
        let root = unique_temp_dir("rejected-usage-state");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646100}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"account","time":1784329646200}
{"type":"usage.record","model":"kimi-code/legacy","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646300}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert!(records[0].dedup_key.starts_with("kimi:file:sha256:"));
        assert_eq!(records[0].entry.model, "legacy");
    }

    #[test]
    fn duplicate_completion_after_pairing_does_not_rekey_the_next_usage() {
        let root = unique_temp_dir("duplicate-completion");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646200}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646201}
{"type":"usage.record","model":"kimi-code/legacy","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646300}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].dedup_key, "kimi:response:response-a");
        assert_ne!(records[1].dedup_key, "kimi:response:response-a");
        assert!(records[1].dedup_key.starts_with("kimi:file:sha256:"));
    }

    #[test]
    fn completion_before_usage_prefers_the_new_waiting_request() {
        let root = unique_temp_dir("new-request-completion-first");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"low","time":1784329646000}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646100}
{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-b","thinkingEffort":"high","time":1784329646200}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0}},"time":1784329646300}
{"type":"usage.record","model":"kimi-code/model-b","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert!(records[0].dedup_key.starts_with("kimi:file:sha256:"));
        assert_eq!(records[0].entry.model, "model-a");
        assert_eq!(records[1].dedup_key, "kimi:response:response-b");
        assert_eq!(records[1].entry.model, "model-b");
    }

    #[test]
    fn malformed_numeric_usage_consumes_a_pending_completion() {
        let root = unique_temp_dir("malformed-numeric-state");
        let content = r#"{"type":"llm.request","kind":"loop","modelAlias":"kimi-code/model-a","thinkingEffort":"high","time":1784329646000}
{"type":"context.append_loop_event","event":{"type":"step.end","messageId":"response-a","usage":{"inputOther":10,"output":1,"inputCacheRead":2,"inputCacheCreation":0}},"time":1784329646100}
{"type":"usage.record","model":"kimi-code/model-a","usage":{"inputOther":"invalid","output":1,"inputCacheRead":2,"inputCacheCreation":0},"usageScope":"turn","time":1784329646200}
{"type":"usage.record","model":"kimi-code/legacy","usage":{"inputOther":20,"output":2,"inputCacheRead":3,"inputCacheCreation":0},"usageScope":"turn","time":1784329646300}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_ne!(records[0].dedup_key, "kimi:response:response-a");
        assert_eq!(records[0].entry.model, "legacy");
    }

    #[test]
    fn records_without_a_usable_time_are_skipped() {
        let root = unique_temp_dir("no-time");
        let content = r#"{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":10,"output":1,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn"}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":20,"output":2,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":"not-a-number"}
{"type":"usage.record","model":"kimi-code/k3","usage":{"inputOther":30,"output":3,"inputCacheRead":0,"inputCacheCreation":0},"usageScope":"turn","time":1784329646400}
"#;
        let path = write_wire_file(&session_wire_path(&root), content);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 30);
        assert!(records[0].entry.parsed_timestamp.is_some());
    }

    #[test]
    fn idless_records_use_bounded_file_position_keys() {
        let root = unique_temp_dir("dedup");
        let path = write_wire_file(&session_wire_path(&root), WIRE_SAMPLE);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].dedup_key,
            crate::data::file_fallback_key("kimi", &path, "1784329652715:2")
        );
        assert_eq!(
            records[1].dedup_key,
            crate::data::file_fallback_key("kimi", &path, "1784329693636:4")
        );
        let keys: HashSet<&str> = records.iter().map(|r| r.dedup_key.as_str()).collect();
        assert_eq!(keys.len(), records.len());
        assert!(records.iter().all(|record| record.dedup_key.len() <= 512));
    }

    #[test]
    fn long_session_layout_keeps_fallback_keys_bounded_and_path_scoped() {
        let root = unique_temp_dir("long-session-fallback");
        let session = "s".repeat(240);
        let agent = "a".repeat(240);
        let dir = root
            .join("workspace")
            .join(session)
            .join("agents")
            .join(agent);
        let path = write_wire_file(&dir, WIRE_SAMPLE);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].dedup_key,
            crate::data::file_fallback_key("kimi", &path, "1784329652715:2")
        );
        assert!(records.iter().all(|record| record.dedup_key.len() <= 512));
    }

    #[test]
    fn unexpected_layout_falls_back_to_file_position_keys() {
        let root = unique_temp_dir("fallback");
        let path = write_wire_file(&root, WIRE_SAMPLE);

        let records = read_kimi_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].dedup_key,
            crate::data::file_fallback_key("kimi", &path, "1784329652715:2")
        );
        assert_eq!(
            records[1].dedup_key,
            crate::data::file_fallback_key("kimi", &path, "1784329693636:4")
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
