use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::Value;
use walkdir::WalkDir;

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry, as_i64};
use crate::model_id::normalize_reasoning_effort;
use crate::time_utils::parse_timestamp;

/// Get the DeepSeek Harness home directory: `$DSH_HOME`, else `~/.dsh`.
pub fn get_dsh_dir() -> PathBuf {
    crate::data::config_dir("DSH_HOME", ".dsh")
}

/// Event types whose lines can carry token usage or the model route that
/// attributes it. Everything else in the session log (assistant chunks,
/// reasoning chunks, tool output, messages, ...) is skipped before parsing so
/// large payload lines never pay for a JSON parse.
const PARSED_EVENT_MARKERS: [&str; 8] = [
    "\"type\":\"session\"",
    "\"type\":\"session/end-seed\"",
    "\"type\":\"request/header\"",
    "\"type\":\"request/context\"",
    "\"type\":\"assistant/message\"",
    "\"type\":\"assistant/attempt\"",
    "\"type\":\"llm/retry-started\"",
    "\"type\":\"compaction/summary\"",
];

/// One session log generation. The persistence backend keeps a single log per
/// session directory: `session.jsonl[.zstd]` for the original format and
/// `session.vN.jsonl[.zstd]` once the log was migrated to version `N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SessionLogGeneration {
    version: u32,
    compressed: bool,
}

fn session_log_generation(file_name: &str) -> Option<SessionLogGeneration> {
    let rest = file_name.strip_prefix("session")?;
    let (version, tail) = match rest.strip_prefix(".v") {
        Some(versioned) => {
            let (digits, tail) = versioned.split_once('.')?;
            if digits.is_empty()
                || !digits.bytes().all(|byte| byte.is_ascii_digit())
                || (digits.len() > 1 && digits.starts_with('0'))
            {
                return None;
            }
            (digits.parse().ok()?, format!(".{tail}"))
        }
        None => (0, rest.to_string()),
    };
    let compressed = match tail.as_str() {
        ".jsonl" => false,
        ".jsonl.zstd" => true,
        _ => return None,
    };
    Some(SessionLogGeneration {
        version,
        compressed,
    })
}

/// Collect session logs under `sessions_dir`, one per session directory.
///
/// A session directory can briefly hold both the original and a migrated log
/// (or an older and a newer generation while upgrading), so only the newest
/// generation is read. Records parsed from a previous generation stay in the
/// persistent cache and are deduplicated by their content-stable keys.
pub fn collect_usage_files(sessions_dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    if !sessions_dir.exists() {
        return Vec::new();
    }

    let cutoff = max_age_days
        .map(|days| SystemTime::now() - std::time::Duration::from_secs((days as u64 + 1) * 86_400));

    let mut selected: HashMap<PathBuf, (SessionLogGeneration, PathBuf)> = HashMap::new();
    for entry in WalkDir::new(sessions_dir)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let Some(file_name) = entry.file_name().to_str() else {
            continue;
        };
        let Some(generation) = session_log_generation(file_name) else {
            continue;
        };
        let Some(session_dir) = entry.path().parent() else {
            continue;
        };
        let newer = selected
            .get(session_dir)
            .is_none_or(|(current, _)| generation > *current);
        if newer {
            selected.insert(
                session_dir.to_path_buf(),
                (generation, entry.path().to_path_buf()),
            );
        }
    }

    let mut files: Vec<PathBuf> = selected
        .into_values()
        .filter(|(_, path)| {
            cutoff.is_none_or(|cutoff_time| {
                fs::metadata(path)
                    .and_then(|metadata| metadata.modified())
                    .is_ok_and(|mtime| mtime >= cutoff_time)
            })
        })
        .map(|(_, path)| path)
        .collect();
    files.sort();
    files
}

/// Parse all usage records from one DSH session log.
pub fn read_dsh_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    let Some(content) = read_session_log(path) else {
        return Vec::new();
    };

    let mut parser = DshSessionParser::new(session_id_from_path(path));
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty()
            || !PARSED_EVENT_MARKERS
                .iter()
                .any(|marker| line.contains(marker))
        {
            continue;
        }
        parser.observe_line(line);
    }
    parser.records
}

fn read_session_log(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    if path.extension().is_none_or(|extension| extension != "zstd") {
        return Some(String::from_utf8_lossy(&bytes).into_owned());
    }

    // Every append writes an independent zstd frame, including the session
    // header frame. A crash can leave a torn final frame behind, so keep all
    // frames decoded before the first error instead of dropping the log.
    let mut decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(bytes)).ok()?;
    let mut decoded = Vec::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        match decoder.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => decoded.extend_from_slice(&buffer[..read]),
            Err(_) => break,
        }
    }
    Some(String::from_utf8_lossy(&decoded).into_owned())
}

/// Session ids name the session directory; the log header repeats the id.
fn session_id_from_path(path: &Path) -> String {
    path.parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

fn non_empty_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn token_usage(value: &Value) -> TokenUsage {
    TokenUsage {
        input_tokens: as_i64(value, "inputTokens"),
        output_tokens: as_i64(value, "outputTokens"),
        cache_read_input_tokens: as_i64(value, "cacheReadTokens"),
        cache_creation_input_tokens: as_i64(value, "cacheWriteTokens"),
        cache_creation_5m_input_tokens: 0,
        cache_creation_1h_input_tokens: 0,
        reasoning_output_tokens: as_i64(value, "reasoningTokens"),
    }
}

fn usage_has_tokens(usage: &TokenUsage) -> bool {
    usage.input_tokens > 0
        || usage.output_tokens > 0
        || usage.cache_read_input_tokens > 0
        || usage.cache_creation_input_tokens > 0
}

/// Token buckets that decide whether two settlements of one step report the
/// same provider usage. Reasoning tokens are a subset of the output tokens and
/// cache-write duration buckets do not exist in the DSH wire format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UsageBuckets {
    input: i64,
    output: i64,
    cache_read: i64,
    cache_write: i64,
}

impl From<&TokenUsage> for UsageBuckets {
    fn from(usage: &TokenUsage) -> Self {
        Self {
            input: usage.input_tokens,
            output: usage.output_tokens,
            cache_read: usage.cache_read_input_tokens,
            cache_write: usage.cache_creation_input_tokens,
        }
    }
}

/// The usage sample of one assistant settlement: `assistant/message` carries
/// the final usage in `data.usage`, while `assistant/attempt` only reports the
/// last usage chunk of its stream. Both forms are paid provider calls.
fn settlement_usage(event_type: &str, data: &Value) -> Option<TokenUsage> {
    if event_type == "assistant/message"
        && let Some(usage) = data.get("usage")
    {
        return Some(token_usage(usage));
    }
    last_stream_usage(data.get("stream"))
}

fn last_stream_usage(stream: Option<&Value>) -> Option<TokenUsage> {
    stream?.as_array()?.iter().rev().find_map(|record| {
        if record.get("type").and_then(Value::as_str) != Some("chunk") {
            return None;
        }
        let chunk = record.get("chunk")?;
        if chunk.get("type").and_then(Value::as_str) != Some("usage") {
            return None;
        }
        Some(token_usage(chunk.get("usage")?))
    })
}

fn event_timestamp(event: &Value) -> Option<String> {
    let millis = event.get("time").and_then(Value::as_i64)?;
    DateTime::<Utc>::from_timestamp_millis(millis)
        .map(|timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Millis, true))
}

struct SettlementSlot {
    record_index: usize,
    buckets: UsageBuckets,
}

struct DshSessionParser {
    session_id: String,
    /// Set while a seeded session replays its inherited prefix. Those events
    /// were already counted in the parent session and must not be billed again.
    seeded_prefix_open: bool,
    route_model: String,
    route_effort: Option<String>,
    records: Vec<SourceUsageRecord>,
    slots: HashMap<(u64, u64), SettlementSlot>,
    next_settlement_index: HashMap<(u64, u64), u32>,
    last_settlement_slot: Option<(u64, u64)>,
}

impl DshSessionParser {
    fn new(session_id: String) -> Self {
        Self {
            session_id,
            seeded_prefix_open: false,
            route_model: "unknown".to_string(),
            route_effort: None,
            records: Vec::new(),
            slots: HashMap::new(),
            next_settlement_index: HashMap::new(),
            last_settlement_slot: None,
        }
    }

    fn observe_line(&mut self, line: &str) {
        let event: Value = match serde_json::from_str(line) {
            Ok(event) => event,
            Err(_) => return,
        };
        match event.get("type").and_then(Value::as_str) {
            Some("session") => self.observe_header(&event),
            Some("session/end-seed") => self.observe_seed_end(&event),
            Some("request/header") => self.observe_route(&event, true),
            Some("request/context") => self.observe_route(&event, false),
            Some("llm/retry-started") => self.observe_retry_started(&event),
            Some("assistant/message") => self.observe_settlement("assistant/message", &event),
            Some("assistant/attempt") => self.observe_settlement("assistant/attempt", &event),
            Some("compaction/summary") => self.observe_compaction(&event),
            _ => {}
        }
    }

    fn observe_header(&mut self, event: &Value) {
        // The session header stores its fields at the top level, unlike the
        // event rows below it.
        if let Some(id) = non_empty_str(event, "id") {
            self.session_id = id.to_string();
        }
        self.seeded_prefix_open =
            matches!(event.get("isSeeded").and_then(Value::as_bool), Some(true));
    }

    /// A seeded child marks the end of the inherited prefix with an
    /// `inherited: true` marker. Untagged end-seed markers are benign and do
    /// not change what counts.
    fn observe_seed_end(&mut self, event: &Value) {
        let inherited = event
            .pointer("/data/inherited")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if self.seeded_prefix_open && inherited {
            self.seeded_prefix_open = false;
        }
    }

    fn observe_route(&mut self, event: &Value, header: bool) {
        let Some(data) = event.get("data") else {
            return;
        };
        let config = if header {
            data.pointer("/header/config")
                .or_else(|| data.get("config"))
        } else {
            Some(data)
        };
        let Some(config) = config else {
            return;
        };
        if let Some(model) = non_empty_str(config, "model") {
            self.route_model = model.to_string();
        }
        // Headers snapshot the full request config, so a missing effort means
        // the request runs without one instead of inheriting the previous.
        if header {
            self.route_effort = non_empty_str(config, "reasoningEffort")
                .or_else(|| non_empty_str(config, "effort"))
                .and_then(normalize_reasoning_effort);
        }
    }

    fn observe_retry_started(&mut self, event: &Value) {
        let Some(data) = event.get("data") else {
            return;
        };
        let (Some(turn), Some(step)) = (
            data.get("turn").and_then(Value::as_u64),
            data.get("step").and_then(Value::as_u64),
        ) else {
            return;
        };
        // The retried attempt in the same step is billed separately, so the
        // replacement slot from the previous settlement must be closed.
        if self.last_settlement_slot == Some((turn, step)) {
            self.slots.remove(&(turn, step));
            self.last_settlement_slot = None;
        }
    }

    fn observe_settlement(&mut self, event_type: &str, event: &Value) {
        if self.seeded_prefix_open {
            return;
        }
        let Some(data) = event.get("data") else {
            return;
        };
        let (Some(turn), Some(step)) = (
            data.get("turn").and_then(Value::as_u64),
            data.get("step").and_then(Value::as_u64),
        ) else {
            return;
        };
        let Some(usage) = settlement_usage(event_type, data) else {
            return;
        };
        if !usage_has_tokens(&usage) {
            return;
        }
        let Some(timestamp) = event_timestamp(event) else {
            return;
        };
        self.record_settlement(turn, step, usage, timestamp);
    }

    /// Mirror the DSH token-usage projection: settlements of the same step
    /// replace each other (`assistant/attempt` followed by `assistant/message`
    /// is one provider call), while a retry re-opens the slot for a new
    /// settlement. Each emitted settlement becomes one usage entry.
    fn record_settlement(&mut self, turn: u64, step: u64, usage: TokenUsage, timestamp: String) {
        let buckets = UsageBuckets::from(&usage);
        if let Some(slot) = self.slots.get_mut(&(turn, step)) {
            if slot.buckets == buckets {
                return;
            }
            let record_index = slot.record_index;
            slot.buckets = buckets;
            self.records[record_index].entry.usage = usage;
            self.last_settlement_slot = Some((turn, step));
            return;
        }

        let settlement_index = {
            let counter = self.next_settlement_index.entry((turn, step)).or_insert(0);
            let index = *counter;
            *counter += 1;
            index
        };
        let dedup_key = format!(
            "dsh:session:{}:message:turn:{turn}:step:{step}:settlement:{settlement_index}",
            self.session_id
        );
        let record_index = self.records.len();
        self.records.push(SourceUsageRecord {
            dedup_key,
            entry: UsageEntry {
                host_id: None,
                session_id: Some(self.session_id.clone()),
                timestamp: timestamp.clone(),
                parsed_timestamp: parse_timestamp(&timestamp),
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model: self.route_model.clone(),
                effort: self.route_effort.clone(),
                fast_tier: UNKNOWN_FAST_TIER,
                usage,
                costs: None,
            },
        });
        self.slots.insert(
            (turn, step),
            SettlementSlot {
                record_index,
                buckets,
            },
        );
        self.last_settlement_slot = Some((turn, step));
    }

    /// Compaction summaries are separate provider calls and carry their own
    /// model, so they are always counted, keyed by the stable compaction id.
    fn observe_compaction(&mut self, event: &Value) {
        if self.seeded_prefix_open {
            return;
        }
        let Some(data) = event.get("data") else {
            return;
        };
        let Some(raw_usage) = data.get("usage") else {
            return;
        };
        let usage = token_usage(raw_usage);
        if !usage_has_tokens(&usage) {
            return;
        }
        let Some(timestamp) = event_timestamp(event) else {
            return;
        };
        let compaction_id = non_empty_str(data, "compactionId")
            .map(str::to_string)
            .or_else(|| {
                event
                    .get("seq")
                    .and_then(Value::as_u64)
                    .map(|seq| format!("event:{seq}"))
            })
            .unwrap_or_else(|| format!("record:{}", self.records.len()));
        let model = non_empty_str(data, "model")
            .map(str::to_string)
            .unwrap_or_else(|| self.route_model.clone());
        self.records.push(SourceUsageRecord {
            dedup_key: format!("dsh:session:{}:compaction:{compaction_id}", self.session_id),
            entry: UsageEntry {
                host_id: None,
                session_id: Some(self.session_id.clone()),
                timestamp: timestamp.clone(),
                parsed_timestamp: parse_timestamp(&timestamp),
                session_start_time: timestamp.clone(),
                session_end_time: timestamp,
                model,
                effort: None,
                fast_tier: UNKNOWN_FAST_TIER,
                usage,
                costs: None,
            },
        });
    }
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
        std::env::temp_dir().join(format!("ai-usage-dsh-test-{name}-{stamp}"))
    }

    fn session_dir(root: &Path, session_id: &str) -> PathBuf {
        let dir = root.join(format!("--home-user-project--/{}", session_id));
        fs::create_dir_all(&dir).expect("create session dir");
        dir
    }

    fn write_plain(path: &Path, lines: &[String]) {
        fs::write(path, lines.join("\n")).expect("write session log");
    }

    fn write_frames(path: &Path, lines: &[String], frame_size: usize) {
        let mut compressed = Vec::new();
        for frame in lines.chunks(frame_size) {
            let mut encoder =
                zstd::stream::write::Encoder::new(Vec::new(), 3).expect("zstd encoder");
            for line in frame {
                std::io::Write::write_all(&mut encoder, line.as_bytes()).expect("encode line");
                std::io::Write::write_all(&mut encoder, b"\n").expect("encode newline");
            }
            compressed.extend_from_slice(&encoder.finish().expect("finish frame"));
        }
        fs::write(path, compressed).expect("write compressed session log");
    }

    fn header(id: &str, seeded: bool) -> String {
        serde_json::json!({
            "type": "session",
            "version": 4,
            "id": id,
            "createdAt": 1_790_000_000_000i64,
            "cwd": "/home/user/project",
            "isSeeded": seeded,
        })
        .to_string()
    }

    fn route(model: &str, effort: Option<&str>) -> String {
        let mut config = serde_json::json!({
            "provider": "ark",
            "model": model,
        });
        if let Some(effort) = effort {
            config["reasoningEffort"] = serde_json::json!(effort);
        }
        serde_json::json!({
            "type": "request/header",
            "seq": 1,
            "time": 1_790_000_000_001i64,
            "data": { "header": { "config": config } },
        })
        .to_string()
    }

    fn message(seq: u64, turn: u64, step: u64, time: i64, usage: Value) -> String {
        serde_json::json!({
            "type": "assistant/message",
            "seq": seq,
            "time": time,
            "data": { "turn": turn, "step": step, "message": { "role": "assistant" }, "usage": usage },
        })
        .to_string()
    }

    fn attempt(seq: u64, turn: u64, step: u64, time: i64, usage: Value) -> String {
        let stream = serde_json::json!([
            { "type": "delta", "text": "partial" },
            { "type": "chunk", "chunk": { "type": "usage", "usage": usage } },
        ]);
        serde_json::json!({
            "type": "assistant/attempt",
            "seq": seq,
            "time": time,
            "data": { "turn": turn, "step": step, "stream": stream },
        })
        .to_string()
    }

    fn compaction(seq: u64, time: i64, id: &str, usage: Value) -> String {
        serde_json::json!({
            "type": "compaction/summary",
            "seq": seq,
            "time": time,
            "data": {
                "compactionId": id,
                "provider": "ark",
                "model": "deepseek-v4-1-flash-260910",
                "usage": usage,
            },
        })
        .to_string()
    }

    fn usage(input: i64, output: i64, cache_read: i64, cache_write: i64) -> Value {
        serde_json::json!({
            "inputTokens": input,
            "outputTokens": output,
            "cacheReadTokens": cache_read,
            "cacheWriteTokens": cache_write,
        })
    }

    #[test]
    fn session_log_generation_accepts_versioned_and_legacy_names() {
        assert_eq!(
            session_log_generation("session.jsonl"),
            Some(SessionLogGeneration {
                version: 0,
                compressed: false
            })
        );
        assert_eq!(
            session_log_generation("session.jsonl.zstd"),
            Some(SessionLogGeneration {
                version: 0,
                compressed: true
            })
        );
        assert_eq!(
            session_log_generation("session.v4.jsonl.zstd"),
            Some(SessionLogGeneration {
                version: 4,
                compressed: true
            })
        );
        assert_eq!(
            session_log_generation("session.v12.jsonl"),
            Some(SessionLogGeneration {
                version: 12,
                compressed: false
            })
        );
        for name in [
            "session.lock",
            "session.v.jsonl",
            "session.v04.jsonl.zstd",
            "session.v4.jsonl.gz",
            "other.jsonl",
        ] {
            assert_eq!(session_log_generation(name), None, "name={name}");
        }
    }

    #[test]
    fn collect_usage_files_keeps_the_newest_generation_per_session() {
        let root = unique_temp_dir("collect");
        let session = session_dir(&root, "session-a");
        fs::write(session.join("session.v3.jsonl.zstd"), "stale").expect("v3 log");
        let newest = session.join("session.v4.jsonl.zstd");
        fs::write(&newest, "newest").expect("v4 log");
        fs::write(session.join("session.lock"), "").expect("lock file");
        let plain = session_dir(&root, "session-b").join("session.jsonl");
        fs::write(&plain, "legacy").expect("legacy log");

        let files = collect_usage_files(&root, None);
        fs::remove_dir_all(&root).ok();

        assert_eq!(files, vec![newest, plain]);
    }

    #[test]
    fn reads_concatenated_zstd_frames_and_plain_logs() {
        let root = unique_temp_dir("frames");
        let dir = session_dir(&root, "session-frames");
        let lines = vec![
            header("session-frames", false),
            route("deepseek-v4-1-flash-260910", Some("high")),
            message(3, 1, 1, 1_790_000_000_100, usage(100, 20, 1_000, 0)),
            message(4, 1, 2, 1_790_000_000_200, usage(50, 10, 2_000, 8)),
        ];
        let compressed = dir.join("session.v4.jsonl.zstd");
        write_frames(&compressed, &lines, 2);

        let records = read_dsh_file_records(&compressed);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].entry.model, "deepseek-v4-1-flash-260910");
        assert_eq!(records[0].entry.effort.as_deref(), Some("high"));
        assert_eq!(records[0].entry.timestamp, "2026-09-21T14:13:20.100Z");
        assert_eq!(records[0].entry.usage.input_tokens, 100);
        assert_eq!(records[0].entry.usage.output_tokens, 20);
        assert_eq!(records[0].entry.usage.cache_read_input_tokens, 1_000);
        assert_eq!(records[1].entry.usage.cache_creation_input_tokens, 8);

        let plain = dir.join("session.v5.jsonl");
        write_plain(&plain, &lines);
        assert_eq!(read_dsh_file_records(&plain).len(), 2);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn torn_tail_frame_keeps_the_frames_decoded_before_it() {
        let root = unique_temp_dir("torn-frame");
        let dir = session_dir(&root, "session-torn");
        let lines = vec![
            header("session-torn", false),
            route("deepseek-v4-flash", None),
            message(3, 1, 1, 1_790_000_000_100, usage(10, 2, 0, 0)),
        ];
        let path = dir.join("session.v4.jsonl.zstd");
        write_frames(&path, &lines, 2);
        let mut bytes = fs::read(&path).expect("read frames");
        bytes.extend_from_slice(&[0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x01, 0x02]);
        fs::write(&path, bytes).expect("write torn log");

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 10);
    }

    #[test]
    fn skipped_inherited_seed_prefix_is_not_billed_twice() {
        let root = unique_temp_dir("seeded");
        let dir = session_dir(&root, "session-child");
        let lines = vec![
            header("session-child", true),
            route("deepseek-v4-flash", None),
            message(2, 1, 1, 1_790_000_000_100, usage(7_000, 300, 0, 0)),
            message(3, 1, 2, 1_790_000_000_200, usage(5_000, 200, 0, 0)),
            serde_json::json!({
                "type": "session/end-seed",
                "seq": 4,
                "time": 1_790_000_000_300i64,
                "data": { "inherited": true },
            })
            .to_string(),
            message(5, 1, 3, 1_790_000_000_400, usage(10, 2, 0, 0)),
        ];
        let path = dir.join("session.v4.jsonl");
        write_plain(&path, &lines);

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 10);
        assert_eq!(records[0].entry.usage.output_tokens, 2);
    }

    #[test]
    fn untagged_end_seed_marker_does_not_drop_live_usage() {
        let root = unique_temp_dir("unseeded");
        let dir = session_dir(&root, "session-parent");
        let lines = vec![
            header("session-parent", false),
            route("deepseek-v4-flash", None),
            serde_json::json!({
                "type": "session/end-seed",
                "seq": 2,
                "time": 1_790_000_000_050i64,
                "data": {},
            })
            .to_string(),
            message(3, 1, 1, 1_790_000_000_100, usage(10, 2, 0, 0)),
        ];
        let path = dir.join("session.v4.jsonl");
        write_plain(&path, &lines);

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 10);
    }

    #[test]
    fn same_step_settlements_replace_while_retries_add() {
        let root = unique_temp_dir("settlements");
        let dir = session_dir(&root, "session-retry");
        let lines = vec![
            header("session-retry", false),
            route("deepseek-v4-flash", None),
            attempt(2, 1, 1, 1_790_000_000_100, usage(5, 1, 0, 0)),
            message(3, 1, 1, 1_790_000_000_200, usage(9, 3, 0, 0)),
            serde_json::json!({
                "type": "llm/retry-started",
                "seq": 4,
                "time": 1_790_000_000_300i64,
                "data": { "turn": 2, "step": 1 },
            })
            .to_string(),
            attempt(5, 2, 1, 1_790_000_000_400, usage(4, 2, 0, 0)),
            serde_json::json!({
                "type": "llm/retry-started",
                "seq": 6,
                "time": 1_790_000_000_500i64,
                "data": { "turn": 2, "step": 1 },
            })
            .to_string(),
            message(7, 2, 1, 1_790_000_000_600, usage(11, 4, 0, 0)),
        ];
        let path = dir.join("session.v4.jsonl");
        write_plain(&path, &lines);

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 3);
        // Step 1/1 was reported twice without a retry: the later sample wins.
        assert_eq!(records[0].entry.usage.input_tokens, 9);
        assert_eq!(records[0].entry.usage.output_tokens, 3);
        // A retried attempt is its own provider call and adds to the total.
        assert_eq!(records[1].entry.usage.input_tokens, 4);
        assert_eq!(records[2].entry.usage.input_tokens, 11);
        let keys: HashSet<&str> = records.iter().map(|r| r.dedup_key.as_str()).collect();
        assert_eq!(keys.len(), records.len());
        assert!(records.iter().all(|record| {
            record
                .dedup_key
                .starts_with("dsh:session:session-retry:message:")
        }));
    }

    #[test]
    fn dedup_keys_and_reported_usage_survive_log_migration() {
        let root = unique_temp_dir("migration");
        let dir = session_dir(&root, "session-migrated");
        let build = |first_seq: u64, second_seq: u64| {
            vec![
                header("session-migrated", false),
                route("deepseek-v4-flash", None),
                message(first_seq, 1, 1, 1_790_000_000_100, usage(10, 2, 0, 0)),
                message(second_seq, 1, 2, 1_790_000_000_200, usage(20, 4, 0, 0)),
            ]
        };
        let legacy = dir.join("session.jsonl.zstd");
        let migrated = dir.join("session.v4.jsonl.zstd");
        write_frames(&legacy, &build(10, 20), 2);
        write_frames(&migrated, &build(1, 2), 2);

        let legacy_records = read_dsh_file_records(&legacy);
        let migrated_records = read_dsh_file_records(&migrated);
        fs::remove_dir_all(&root).ok();

        assert_eq!(legacy_records.len(), migrated_records.len());
        for (legacy, migrated) in legacy_records.iter().zip(&migrated_records) {
            assert_eq!(legacy.dedup_key, migrated.dedup_key);
            assert_eq!(
                legacy.entry.usage.input_tokens,
                migrated.entry.usage.input_tokens
            );
        }
    }

    #[test]
    fn compaction_summaries_count_as_their_own_provider_call() {
        let root = unique_temp_dir("compaction");
        let dir = session_dir(&root, "session-compaction");
        let lines = vec![
            header("session-compaction", false),
            route("deepseek-v4-flash", None),
            message(2, 1, 1, 1_790_000_000_100, usage(10, 2, 0, 0)),
            compaction(
                3,
                1_790_000_000_200,
                "compaction-1",
                usage(1_000, 500, 0, 0),
            ),
        ];
        let path = dir.join("session.v4.jsonl.zstd");
        write_frames(&path, &lines, 3);

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[1].entry.usage.input_tokens, 1_000);
        assert_eq!(records[1].entry.model, "deepseek-v4-1-flash-260910");
        assert_eq!(
            records[1].dedup_key,
            "dsh:session:session-compaction:compaction:compaction-1"
        );
    }

    #[test]
    fn falls_back_to_stream_usage_and_resets_effort_per_request() {
        let root = unique_temp_dir("stream-usage");
        let dir = session_dir(&root, "session-stream");
        let stream_only = serde_json::json!({
            "type": "assistant/message",
            "seq": 3,
            "time": 1_790_000_000_100i64,
            "data": {
                "turn": 1,
                "step": 1,
                "message": { "role": "assistant" },
                "stream": [
                    { "type": "chunk", "chunk": { "type": "usage", "usage": usage(1, 1, 0, 0) } },
                    { "type": "chunk", "chunk": { "type": "usage", "usage": usage(12, 3, 40, 0) } },
                ],
            },
        })
        .to_string();
        let lines = vec![
            header("session-stream", false),
            route("deepseek-v4-1-flash-260910", Some("max")),
            stream_only,
            route("deepseek-v4-flash", None),
            message(5, 2, 1, 1_790_000_000_200, usage(5, 1, 0, 0)),
        ];
        let path = dir.join("session.v4.jsonl");
        write_plain(&path, &lines);

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].entry.usage.input_tokens, 12);
        assert_eq!(records[0].entry.usage.cache_read_input_tokens, 40);
        assert_eq!(records[0].entry.effort.as_deref(), Some("max"));
        assert_eq!(records[1].entry.model, "deepseek-v4-flash");
        assert_eq!(records[1].entry.effort, None);
    }

    #[test]
    fn session_id_falls_back_to_the_session_directory() {
        let root = unique_temp_dir("fallback-id");
        let dir = session_dir(&root, "session-directory-name");
        let lines = vec![
            route("deepseek-v4-flash", None),
            message(2, 1, 1, 1_790_000_000_100, usage(1, 1, 0, 0)),
        ];
        let path = dir.join("session.v4.jsonl");
        write_plain(&path, &lines);

        let records = read_dsh_file_records(&path);
        fs::remove_dir_all(&root).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].entry.session_id.as_deref(),
            Some("session-directory-name")
        );
    }
}
