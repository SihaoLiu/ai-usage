use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use walkdir::WalkDir;

use crate::data::{SourceUsageRecord, TokenUsage, UsageEntry};
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
    std::env::var("CODEX_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".codex"))
                .unwrap_or_else(|_| PathBuf::from("~/.codex"))
        })
}

/// Read the top-level `service_tier` setting from `~/.codex/config.toml`.
/// Returns the literal value (e.g. `"fast"`, `"flex"`) if present in the
/// global section, or `None` when the key is absent or the file is missing.
///
/// Codex rollout JSONL files never store this per turn, so the global
/// config setting is the only deterministic signal we have for whether
/// `/fast` was active. We deliberately stop at the first `[section]` header
/// so a per-profile `service_tier` doesn't get mistaken for the active one.
pub fn detect_service_tier_from_config() -> Option<String> {
    let path = get_codex_dir().join("config.toml");
    let content = fs::read_to_string(&path).ok()?;
    parse_top_level_service_tier(&content)
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

/// Key used for cross-file deduplication. A turn_id (when available) together with
/// the usage tuple uniquely identifies a Codex billing event: a single turn may emit
/// many `token_count` events (one per sub-inference call), each with a distinct
/// `last_token_usage`, so the usage tuple is required to avoid collapsing them. When
/// turn_id is unavailable we fall back to the event timestamp which, within a single
/// session, is unique per event.
type DedupKey = (String, i64, i64, i64, i64);

/// An entry produced by a single file, paired with its cross-file dedup key.
struct RawEntry {
    entry: UsageEntry,
    dedup_key: DedupKey,
}

/// Parse one Codex rollout file.
///
/// Events flagged as replayed (from a forked-from session) are dropped before they
/// are summed. Consecutive duplicates are also skipped to guard against a future
/// codex build emitting the same `token_count` twice in a row.
fn read_single_codex_file(path: &Path) -> Vec<RawEntry> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let fork_boundary_ms = detect_fork_boundary(&content);

    let mut current_model = "unknown".to_string();
    let mut current_effort = "unknown".to_string();
    let mut current_turn_id: String = String::new();
    let mut current_turn_started_ms: i64 = 0;
    let mut last_usage_key: Option<(i64, i64, i64, i64)> = None;
    let mut results: Vec<RawEntry> = Vec::new();

    for line in content.lines() {
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
                        if let Some(tid) = payload.get("turn_id").and_then(|v| v.as_str()) {
                            current_turn_id.clear();
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

                        if is_replayed_event(
                            fork_boundary_ms,
                            parsed_ts.map(|t| t.timestamp_millis()),
                            current_turn_started_ms,
                        ) {
                            continue;
                        }

                        let info = match payload.get("info") {
                            Some(i) => i,
                            None => continue,
                        };
                        let token_usage = match info.get("last_token_usage") {
                            Some(t) => t,
                            None => continue,
                        };

                        let input_tokens = token_usage
                            .get("input_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        let cached_input = token_usage
                            .get("cached_input_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        let output_tokens = token_usage
                            .get("output_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        let reasoning_output = token_usage
                            .get("reasoning_output_tokens")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);

                        let usage_key =
                            (input_tokens, cached_input, output_tokens, reasoning_output);
                        if Some(usage_key) == last_usage_key {
                            continue;
                        }
                        last_usage_key = Some(usage_key);

                        let non_cached_input = input_tokens.saturating_sub(cached_input).max(0);

                        let dedup_id = if !current_turn_id.is_empty() {
                            current_turn_id.clone()
                        } else {
                            entry_timestamp.to_string()
                        };
                        let dedup_key = (
                            dedup_id,
                            input_tokens,
                            cached_input,
                            output_tokens,
                            reasoning_output,
                        );

                        let ts_owned = entry_timestamp.to_string();
                        results.push(RawEntry {
                            entry: UsageEntry {
                                host_id: None,
                                timestamp: ts_owned.clone(),
                                parsed_timestamp: parsed_ts,
                                session_start_time: ts_owned.clone(),
                                session_end_time: ts_owned,
                                model: current_model.clone(),
                                effort: Some(current_effort.clone()),
                                usage: TokenUsage {
                                    input_tokens: non_cached_input,
                                    output_tokens,
                                    cache_read_input_tokens: cached_input,
                                    cache_creation_input_tokens: 0,
                                    reasoning_output_tokens: reasoning_output,
                                },
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
        return turn_started_ms < fork_ms;
    }
    match event_ts_ms {
        Some(ts_ms) => ts_ms <= fork_ms + FORK_REPLAY_FALLBACK_WINDOW_MS,
        None => false,
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

pub fn read_codex_file_records(path: &Path) -> Vec<SourceUsageRecord> {
    read_single_codex_file(path)
        .into_iter()
        .map(|raw| SourceUsageRecord {
            dedup_key: serde_json::to_string(&raw.dedup_key).unwrap_or_default(),
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
        std::env::temp_dir().join(format!("vibe-usage-codex-test-{name}-{stamp}.jsonl"))
    }

    #[test]
    fn non_fork_session_has_no_boundary() {
        let content = r#"{"timestamp":"2026-04-22T01:04:48.994Z","type":"session_meta","payload":{"id":"abc","timestamp":"2026-04-22T01:04:46.000Z"}}"#;
        assert_eq!(detect_fork_boundary(content), None);
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
    fn reasoning_output_does_not_make_visible_output_negative() {
        let path = unique_temp_file("reasoning-output");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-19T12:00:00Z","type":"turn_context","payload":{"model":"gpt-5.5","effort":"high"}}
{"timestamp":"2026-05-19T12:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-a","started_at":1779192001}}
{"timestamp":"2026-05-19T12:00:02Z","type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":100,"cached_input_tokens":150,"output_tokens":25,"reasoning_output_tokens":64}}}}"#,
        )
        .expect("write codex fixture");

        let records = read_codex_file_records(&path);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.usage.input_tokens, 0);
        assert_eq!(records[0].entry.usage.cache_read_input_tokens, 150);
        assert_eq!(records[0].entry.usage.output_tokens, 25);
        assert_eq!(records[0].entry.usage.reasoning_output_tokens, 64);
    }
}
