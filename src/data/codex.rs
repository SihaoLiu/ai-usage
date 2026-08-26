use std::fs;
use std::path::{Path, PathBuf};

use crate::data::{
    SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry, file_fallback_key,
};
use crate::time_utils::parse_timestamp;

/// When a Codex session is resumed via `--fork`, the new rollout file replays every
/// event from the source session at the top. Replayed events are re-stamped at
/// fork-creation time, but the `task_started.started_at` field preserves each turn's
/// original (pre-fork) start time. That makes `started_at` the unambiguous signal:
/// any turn whose `started_at` is before the fork boundary belongs to the source
/// session and must not be recounted.
///
/// When a `token_count` is seen before any `task_started` (should not happen under
/// normal codex output, but we guard against it), fall back to a short timestamp
/// window: replayed `token_count` events cluster within milliseconds of each other
/// right after the fork boundary, while real post-fork events only appear once the
/// user submits something after resume.
const FORK_REPLAY_FALLBACK_WINDOW_MS: i64 = 500;

/// Get the Codex configuration directory.
pub fn get_codex_dir() -> PathBuf {
    crate::data::config_dir("CODEX_CONFIG_DIR", ".codex")
}

fn read_top_level_service_tier_from_config() -> Option<String> {
    let path = get_codex_dir().join("config.toml");
    let content = fs::read_to_string(&path).ok()?;
    parse_top_level_service_tier(&content)
}

pub fn detect_fast_tier_snapshot() -> i8 {
    let Some(raw) = read_top_level_service_tier_from_config() else {
        return 0;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "fast" | "priority" => 1,
        _ => 0,
    }
}

fn parse_top_level_service_tier(content: &str) -> Option<String> {
    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            break;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = match line.split_once('=') {
            Some(pair) => pair,
            None => continue,
        };
        if key.trim() != "service_tier" {
            continue;
        }
        let value = value
            .split('#')
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CodexTokenUsageSnapshot {
    input: i64,
    cached_input: i64,
    cache_write_input: i64,
    output: i64,
    reasoning_output: i64,
    total: i64,
}

impl CodexTokenUsageSnapshot {
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_object()?;
        let snapshot = Self {
            input: required_token_field(value, "input_tokens")?,
            cached_input: required_token_field(value, "cached_input_tokens")?,
            cache_write_input: optional_token_field(value, "cache_write_input_tokens")?,
            output: required_token_field(value, "output_tokens")?,
            reasoning_output: required_token_field(value, "reasoning_output_tokens")?,
            total: required_token_field(value, "total_tokens")?,
        };
        ai_usage_proto::is_valid_codex_usage_snapshot(
            snapshot.input,
            snapshot.cached_input,
            snapshot.cache_write_input,
            snapshot.output,
            snapshot.reasoning_output,
            snapshot.total,
        )
        .then_some(snapshot)
    }

    fn key_component(self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.input,
            self.cached_input,
            self.cache_write_input,
            self.output,
            self.reasoning_output,
            self.total
        )
    }

    fn is_zero(self) -> bool {
        self.input == 0
            && self.cached_input == 0
            && self.cache_write_input == 0
            && self.output == 0
            && self.reasoning_output == 0
    }
}

fn required_token_field(value: &serde_json::Value, name: &str) -> Option<i64> {
    value.get(name)?.as_i64()
}

fn optional_token_field(value: &serde_json::Value, name: &str) -> Option<i64> {
    match value.get(name) {
        Some(value) => value.as_i64(),
        None => Some(0),
    }
}

fn cumulative_transition_key(
    path: &Path,
    line_index: usize,
    turn_id: &str,
    previous: Option<CodexTokenUsageSnapshot>,
    current: CodexTokenUsageSnapshot,
) -> String {
    if turn_id.is_empty() {
        return file_fallback_key("codex", path, line_index);
    }
    let previous = previous.map_or_else(|| "start".to_string(), |usage| usage.key_component());
    format!(
        "codex:turn:{turn_id}:cumulative:{previous}->{}",
        current.key_component()
    )
}

/// An entry produced by a single file, paired with its cross-file dedup key.
struct RawEntry {
    entry: UsageEntry,
    dedup_key: String,
}

/// Parse one Codex rollout file.
///
/// Cumulative usage transitions identify completed responses. Repeated emissions
/// with an unchanged cumulative snapshot are ignored.
fn read_single_codex_file(path: &Path) -> Vec<RawEntry> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let fork_boundary_ms = detect_fork_boundary(&content);
    let session_id = session_id_from_content(&content);

    let mut current_model = "unknown".to_string();
    let mut current_effort = "unknown".to_string();
    let mut current_turn_id: String = String::new();
    let mut current_turn_started_ms: i64 = 0;
    let mut previous_total_usage: Option<CodexTokenUsageSnapshot> = None;
    let mut results: Vec<RawEntry> = Vec::new();

    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let data: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let entry_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match entry_type {
            "turn_context" => {
                if let Some(payload) = data.get("payload") {
                    if let Some(model) = payload.get("model").and_then(|v| v.as_str()) {
                        current_model = model.to_string();
                    }
                    if let Some(effort) = payload.get("effort").and_then(|v| v.as_str()) {
                        current_effort = effort.to_string();
                    }
                }
            }
            "event_msg" => {
                let payload = match data.get("payload") {
                    Some(p) => p,
                    None => continue,
                };
                let payload_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");

                match payload_type {
                    "task_started" => {
                        current_turn_id.clear();
                        if let Some(tid) = payload.get("turn_id").and_then(|v| v.as_str()) {
                            current_turn_id.push_str(tid);
                        }
                        if let Some(sa) = payload.get("started_at").and_then(|v| v.as_i64()) {
                            current_turn_started_ms = sa.saturating_mul(1000);
                        } else {
                            current_turn_started_ms = 0;
                        }
                    }
                    "token_count" => {
                        let entry_timestamp =
                            data.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
                        let parsed_ts = parse_timestamp(entry_timestamp);
                        let info = match payload.get("info") {
                            Some(i) => i,
                            None => continue,
                        };
                        let Some(current_total) = info
                            .get("total_token_usage")
                            .and_then(CodexTokenUsageSnapshot::from_json)
                        else {
                            continue;
                        };
                        let previous_total = previous_total_usage.replace(current_total);
                        if previous_total == Some(current_total) {
                            continue;
                        }

                        let Some(last_usage) = info
                            .get("last_token_usage")
                            .and_then(CodexTokenUsageSnapshot::from_json)
                        else {
                            continue;
                        };
                        if last_usage.is_zero()
                            || is_replayed_event(
                                fork_boundary_ms,
                                parsed_ts.map(|t| t.timestamp_millis()),
                                current_turn_started_ms,
                            )
                        {
                            continue;
                        }

                        let non_cached_input = last_usage
                            .input
                            .saturating_sub(last_usage.cached_input)
                            .saturating_sub(last_usage.cache_write_input)
                            .max(0);
                        let dedup_key = cumulative_transition_key(
                            path,
                            line_index,
                            &current_turn_id,
                            previous_total,
                            current_total,
                        );

                        let ts_owned = entry_timestamp.to_string();
                        results.push(RawEntry {
                            entry: UsageEntry {
                                host_id: None,
                                session_id: session_id.clone(),
                                timestamp: ts_owned.clone(),
                                parsed_timestamp: parsed_ts,
                                session_start_time: ts_owned.clone(),
                                session_end_time: ts_owned,
                                model: current_model.clone(),
                                effort: Some(current_effort.clone()),
                                fast_tier: UNKNOWN_FAST_TIER,
                                usage: TokenUsage {
                                    input_tokens: non_cached_input,
                                    output_tokens: last_usage.output,
                                    cache_read_input_tokens: last_usage.cached_input,
                                    cache_creation_input_tokens: last_usage.cache_write_input,
                                    cache_creation_5m_input_tokens: 0,
                                    cache_creation_1h_input_tokens: 0,
                                    reasoning_output_tokens: last_usage.reasoning_output,
                                },
                                costs: None,
                            },
                            dedup_key,
                        });
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    results
}

/// The first metadata event records the stable conversation id for a rollout.
fn session_id_from_content(content: &str) -> Option<String> {
    content.lines().find_map(|line| {
        let data: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
        (data.get("type").and_then(|v| v.as_str()) == Some("session_meta"))
            .then(|| data.get("payload"))
            .flatten()
            .and_then(|payload| payload.get("id"))
            .and_then(|id| id.as_str())
            .filter(|id| !id.is_empty())
            .map(str::to_string)
    })
}

/// If the first session_meta line carries `forked_from_id`, return the fork-creation
/// time in milliseconds. Otherwise return `None` (the file is a fresh session and no
/// replay filtering is needed).
fn detect_fork_boundary(content: &str) -> Option<i64> {
    let first_line = content.lines().find(|l| !l.trim().is_empty())?;
    let data: serde_json::Value = serde_json::from_str(first_line.trim()).ok()?;
    if data.get("type").and_then(|v| v.as_str()) != Some("session_meta") {
        return None;
    }
    let payload = data.get("payload")?;
    payload.get("forked_from_id")?;
    let ts_str = data.get("timestamp").and_then(|v| v.as_str())?;
    parse_timestamp(ts_str).map(|t| t.timestamp_millis())
}

/// Decide whether a `token_count` event is a replay of events from the source of a
/// forked session. Returns `false` for any file that is not a forked session.
///
/// Primary signal: the current turn's `started_at` (if known). A real post-fork turn
/// is initiated by the user after resume, so its `started_at` is >= fork boundary.
/// A replayed turn carries the original session's `started_at`, which is strictly
/// before the fork boundary.
///
/// Fallback: when no `task_started` has been observed yet, use a narrow
/// timestamp-proximity check to catch replayed events at the very top of the file.
#[inline]
fn is_replayed_event(
    fork_boundary_ms: Option<i64>,
    event_ts_ms: Option<i64>,
    turn_started_ms: i64,
) -> bool {
    let Some(fork_ms) = fork_boundary_ms else {
        return false;
    };
    if turn_started_ms > 0 {
        return turn_started_ms.div_euclid(1000) < fork_ms.div_euclid(1000);
    }
    match event_ts_ms {
        Some(ts_ms) => ts_ms <= fork_ms + FORK_REPLAY_FALLBACK_WINDOW_MS,
        None => false,
    }
}

pub fn collect_usage_files(sessions_dir: &Path, max_age_days: Option<i64>) -> Vec<PathBuf> {
    crate::data::collect_recent_files(sessions_dir, max_age_days, crate::data::has_jsonl_extension)
}

pub fn read_codex_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    read_single_codex_file(path)
        .into_iter()
        .map(|raw| SourceUsageRecord {
            dedup_key: raw.dedup_key,
            entry: raw.entry,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("ai-usage-codex-test-{name}-{stamp}.jsonl"))
    }

    #[test]
    fn non_fork_session_has_no_boundary() {
        let content = r#"{"timestamp":"2026-04-22T01:04:48.994Z","type":"session_meta","payload":{"id":"abc","timestamp":"2026-04-22T01:04:46.000Z"}}"#;
        assert_eq!(detect_fork_boundary(content), None);
        assert_eq!(session_id_from_content(content).as_deref(), Some("abc"));
    }

    #[test]
    fn forked_session_returns_outer_timestamp_ms() {
        let content = r#"{"timestamp":"2026-04-22T14:53:04.267Z","type":"session_meta","payload":{"id":"new","forked_from_id":"orig","timestamp":"2026-04-22T05:38:54.803Z"}}
{"timestamp":"2026-04-22T14:53:04.268Z","type":"session_meta","payload":{"id":"orig","timestamp":"2026-04-22T05:38:54.803Z"}}"#;
        let boundary = detect_fork_boundary(content).expect("should detect fork");
        let expected = parse_timestamp("2026-04-22T14:53:04.267Z")
            .unwrap()
            .timestamp_millis();
        assert_eq!(boundary, expected);
    }

    #[test]
    fn started_at_before_fork_is_replay() {
        let fork = 10_000_000_i64;
        assert!(is_replayed_event(Some(fork), Some(fork + 10_000), fork - 1));
        assert!(is_replayed_event(Some(fork), Some(fork + 10_000), 1));
    }

    #[test]
    fn started_at_at_or_after_fork_is_fresh() {
        let fork = 10_000_000_i64;
        assert!(!is_replayed_event(Some(fork), Some(fork + 10_000), fork));
        assert!(!is_replayed_event(
            Some(fork),
            Some(fork + 10_000),
            fork + 5_000
        ));
    }

    #[test]
    fn started_at_in_same_second_as_fork_is_fresh() {
        let fork = 10_957_i64;

        assert!(!is_replayed_event(Some(fork), Some(fork + 10_000), 10_000));
        assert!(is_replayed_event(Some(fork), Some(fork + 10_000), 9_000));
    }

    #[test]
    fn timestamp_fallback_when_no_task_started() {
        let fork = 10_000_000_i64;
        assert!(is_replayed_event(Some(fork), Some(fork), 0));
        assert!(is_replayed_event(
            Some(fork),
            Some(fork + FORK_REPLAY_FALLBACK_WINDOW_MS),
            0
        ));
        assert!(!is_replayed_event(
            Some(fork),
            Some(fork + FORK_REPLAY_FALLBACK_WINDOW_MS + 1),
            0
        ));
    }

    #[test]
    fn not_replayed_when_not_fork() {
        assert!(!is_replayed_event(None, Some(0), 0));
        assert!(!is_replayed_event(None, Some(100), 50));
    }

    #[test]
    fn parses_top_level_service_tier() {
        let toml = "model = \"gpt-5.5\"\nservice_tier = \"fast\"\n[features]\nfast_mode = true\n";
        assert_eq!(parse_top_level_service_tier(toml), Some("fast".to_string()));
    }

    #[test]
    fn ignores_service_tier_inside_section() {
        // A `service_tier` line nested under `[profiles.foo]` must not be
        // picked up as the active global tier.
        let toml = "model = \"gpt-5.5\"\n[profiles.cost-optimized]\nservice_tier = \"flex\"\n";
        assert_eq!(parse_top_level_service_tier(toml), None);
    }

    #[test]
    fn parses_value_with_inline_comment_and_single_quotes() {
        let toml = "service_tier = 'fast' # default tier\n";
        assert_eq!(parse_top_level_service_tier(toml), Some("fast".to_string()));
    }

    #[test]
    fn missing_service_tier_returns_none() {
        let toml = "model = \"gpt-5.5\"\n";
        assert_eq!(parse_top_level_service_tier(toml), None);
    }

    #[test]
    fn reasoning_larger_than_output_is_rejected() {
        let path = unique_temp_file("reasoning-output");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:00Z","type":"turn_context","payload":{"model":"gpt-5.5","effort":"high"}}
{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":80,"cache_write_input_tokens":0,"output_tokens":25,"reasoning_output_tokens":64,"total_tokens":125},"last_token_usage":{"input_tokens":100,"cached_input_tokens":80,"cache_write_input_tokens":0,"output_tokens":25,"reasoning_output_tokens":64,"total_tokens":125}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert!(records.is_empty());
    }

    #[test]
    fn same_last_usage_with_new_cumulative_total_is_counted() {
        let path = unique_temp_file("advancing-cumulative-total");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:00Z","type":"turn_context","payload":{"model":"gpt-5.5","effort":"high"}}
{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":0,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":0,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":20,"cached_input_tokens":4,"cache_write_input_tokens":0,"output_tokens":6,"reasoning_output_tokens":2,"total_tokens":26},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":0,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].entry.usage.input_tokens, 8);
        assert_eq!(
            records[0].dedup_key,
            "codex:turn:turn-a:cumulative:start->10,2,0,3,1,13"
        );
        assert_eq!(
            records[1].dedup_key,
            "codex:turn:turn-a:cumulative:10,2,0,3,1,13->20,4,0,6,2,26"
        );
    }

    #[test]
    fn unchanged_cumulative_total_suppresses_duplicate_emission() {
        let path = unique_temp_file("unchanged-cumulative-total");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":0,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":0,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":0,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":99,"cached_input_tokens":5,"cache_write_input_tokens":0,"output_tokens":9,"reasoning_output_tokens":2,"total_tokens":108}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 8);
    }

    #[test]
    fn cumulative_transition_continues_across_turns() {
        let path = unique_temp_file("cross-turn-cumulative-total");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:03Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-b","started_at":1779192003}}
{"timestamp":"2026-05-19T12:00:04Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":20,"cached_input_tokens":4,"output_tokens":6,"reasoning_output_tokens":2,"total_tokens":26},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert_eq!(records.len(), 2);
        assert_eq!(
            records[1].dedup_key,
            "codex:turn:turn-b:cumulative:10,2,0,3,1,13->20,4,0,6,2,26"
        );
    }

    #[test]
    fn zero_component_update_advances_transition_without_counting_a_message() {
        let path = unique_temp_file("zero-transition");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":0,"cached_input_tokens":0,"cache_write_input_tokens":0,"output_tokens":0,"reasoning_output_tokens":0,"total_tokens":0},"last_token_usage":{"input_tokens":0,"cached_input_tokens":0,"cache_write_input_tokens":0,"output_tokens":0,"reasoning_output_tokens":0,"total_tokens":0}}}}
{"timestamp":"2026-05-19T12:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":1,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"cache_write_input_tokens":1,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 7);
        assert_eq!(records[0].entry.usage.cache_creation_input_tokens, 1);
        assert_eq!(
            records[0].dedup_key,
            "codex:turn:turn-a:cumulative:0,0,0,0,0,0->10,2,1,3,1,13"
        );
    }

    #[test]
    fn missing_cumulative_total_is_not_a_usage_record() {
        let path = unique_temp_file("missing-cumulative-total");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        assert!(read_codex_file_records(&path).is_empty());
    }

    #[test]
    fn missing_turn_id_uses_a_host_scoped_file_key() {
        let path = unique_temp_file("missing-turn-id");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert_eq!(records.len(), 1);
        assert!(records[0].dedup_key.starts_with("codex:file:"));
        assert!(!ai_usage_proto::is_globally_stable_usage_key(
            "codex",
            &records[0].dedup_key
        ));
    }

    #[test]
    fn malformed_or_inconsistent_token_snapshots_are_skipped() {
        let path = unique_temp_file("invalid-token-snapshots");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":null,"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":9223372036854775808,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:04Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":-1,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:05Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":99},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}
{"timestamp":"2026-05-19T12:00:06Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":11,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        assert!(read_codex_file_records(&path).is_empty());
    }

    #[test]
    fn malformed_last_usage_advances_a_valid_cumulative_snapshot() {
        let path = unique_temp_file("malformed-last-usage");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":null}}}
{"timestamp":"2026-05-19T12:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#,
        )
        .expect("write codex fixture");

        assert!(read_codex_file_records(&path).is_empty());
    }

    #[test]
    fn copied_transitions_have_identical_keys() {
        let first_path = unique_temp_file("copied-transition-a");
        let second_path = unique_temp_file("copied-transition-b");
        let first = r#"{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#;
        let second = r#"{"timestamp":"2026-05-20T13:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779282001}}
{"timestamp":"2026-05-20T13:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13},"last_token_usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1,"total_tokens":13}}}}"#;
        fs::write(&first_path, first).expect("write first copy");
        fs::write(&second_path, second).expect("write second copy");

        let first_keys: Vec<String> = read_codex_file_records(&first_path)
            .into_iter()
            .map(|record| record.dedup_key)
            .collect();
        let second_keys: Vec<String> = read_codex_file_records(&second_path)
            .into_iter()
            .map(|record| record.dedup_key)
            .collect();

        assert_eq!(first_keys, second_keys);
    }
}
