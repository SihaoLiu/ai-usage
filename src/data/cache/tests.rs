use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageEntry};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PersistedVendorRecordsWithFastTier {
    format_version: u32,
    records: Vec<PersistedSourceRecordWithFastTier>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PersistedSourceRecordWithFastTier {
    source_path: String,
    dedup_key: String,
    timestamp: String,
    session_start_time: String,
    session_end_time: String,
    model: String,
    effort: Option<String>,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    reasoning_output_tokens: i64,
    fast_tier: i8,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PersistedRemoteRecordsWithFastTier {
    format_version: u32,
    records: Vec<PersistedRemoteRecordWithFastTier>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PersistedRemoteRecordWithFastTier {
    vendor: String,
    dedup_key: String,
    timestamp: String,
    session_start_time: String,
    session_end_time: String,
    model: String,
    effort: Option<String>,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    reasoning_output_tokens: i64,
    fast_tier: i8,
}

fn unique_temp_dir(name: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ai-usage-cache-test-{}-{}", name, stamp));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_source(path: &Path, content: &str) {
    fs::write(path, content).expect("write source");
}

fn write_payload_file(path: &Path, magic: &[u8], payload: &[u8]) {
    let checksum = super::fnv1a_bytes(0, payload);
    let mut content = Vec::new();
    content.extend_from_slice(magic);
    content.extend_from_slice(&checksum.to_le_bytes());
    content.extend_from_slice(payload);
    fs::write(path, content).expect("write payload file");
}

fn usage_record(key: &str, timestamp: &str, input_tokens: i64) -> SourceUsageRecord {
    SourceUsageRecord {
        dedup_key: key.to_string(),
        entry: UsageEntry {
            host_id: None,
            session_id: None,
            timestamp: timestamp.to_string(),
            parsed_timestamp: crate::time_utils::parse_timestamp(timestamp),
            session_start_time: timestamp.to_string(),
            session_end_time: timestamp.to_string(),
            model: "test-model".to_string(),
            effort: None,
            fast_tier: UNKNOWN_FAST_TIER,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 2,
                cache_read_input_tokens: 3,
                cache_creation_input_tokens: 4,
                reasoning_output_tokens: 5,
            },
            costs: None,
        },
    }
}

fn entry_tokens(entries: &[UsageEntry]) -> Vec<i64> {
    entries
        .iter()
        .map(|entry| entry.usage.input_tokens)
        .collect()
}

fn remote_record(
    vendor: &str,
    dedup_key: &str,
    timestamp: &str,
    input_tokens: i64,
) -> super::RemoteUsageRecord {
    super::RemoteUsageRecord {
        vendor: vendor.to_string(),
        dedup_key: dedup_key.to_string(),
        entry: UsageEntry {
            host_id: None,
            session_id: None,
            timestamp: timestamp.to_string(),
            parsed_timestamp: crate::time_utils::parse_timestamp(timestamp),
            session_start_time: timestamp.to_string(),
            session_end_time: timestamp.to_string(),
            model: "remote-model".to_string(),
            effort: None,
            fast_tier: UNKNOWN_FAST_TIER,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 2,
                cache_read_input_tokens: 3,
                cache_creation_input_tokens: 4,
                reasoning_output_tokens: 5,
            },
            costs: None,
        },
    }
}

fn omp_v220_key(message_id: &str, response_id: &str, model: &str, input_tokens: i64) -> String {
    serde_json::json!({
        "message": message_id,
        "response": response_id,
        "model": model,
        "input": input_tokens,
        "output": 2,
        "cache_read": 3,
        "cache_write": 4,
    })
    .to_string()
}

fn remote_fingerprints(
    records: &[super::RemoteUsageRecord],
) -> Vec<(String, String, Option<String>, i64)> {
    records
        .iter()
        .map(|record| {
            (
                record.vendor.clone(),
                record.dedup_key.clone(),
                record.entry.host_id.clone(),
                record.entry.usage.input_tokens,
            )
        })
        .collect()
}

type EntryFingerprint = (
    String,
    Option<String>,
    String,
    String,
    String,
    Option<String>,
    i64,
    i64,
    i64,
    i64,
    i64,
    i8,
);

fn entry_fingerprints(entries: &[UsageEntry]) -> Vec<EntryFingerprint> {
    entries
        .iter()
        .map(|entry| {
            (
                entry.timestamp.clone(),
                entry
                    .parsed_timestamp
                    .map(|timestamp| timestamp.to_rfc3339()),
                entry.session_start_time.clone(),
                entry.session_end_time.clone(),
                entry.model.clone(),
                entry.effort.clone(),
                entry.usage.input_tokens,
                entry.usage.output_tokens,
                entry.usage.cache_read_input_tokens,
                entry.usage.cache_creation_input_tokens,
                entry.usage.reasoning_output_tokens,
                entry.fast_tier,
            )
        })
        .collect()
}

fn direct_parse(
    source_files: Vec<std::path::PathBuf>,
    parse_file: impl Fn(&Path) -> Vec<SourceUsageRecord>,
) -> Vec<UsageEntry> {
    let mut seen = std::collections::HashSet::new();
    let mut entries = Vec::new();
    for source_file in source_files {
        for record in parse_file(&source_file) {
            if !record.dedup_key.is_empty() && !seen.insert(record.dedup_key.clone()) {
                continue;
            }
            entries.push(record.entry);
        }
    }
    entries
}

#[test]
fn omp_alias_candidates_ignore_reasoning_effort_values() {
    let with_provider = super::omp_model_candidates_for("gpt-5", Some("rust-cat"));
    let with_effort = super::omp_model_candidates_for("gpt-5", Some("xhigh"));

    assert!(with_provider.contains(&"rust-cat/gpt-5".to_string()));
    assert!(with_provider.contains(&"openai-codex/gpt-5".to_string()));
    assert!(!with_effort.contains(&"xhigh/gpt-5".to_string()));
    assert!(with_effort.contains(&"openai-codex/gpt-5".to_string()));
}

#[test]
fn local_cache_entries_have_no_host_id() {
    let cache_root = unique_temp_dir("local-host");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");

    let entries = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
    });

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].host_id, None);
}

#[test]
fn local_cache_round_trips_session_ids() {
    let cache_root = unique_temp_dir("session-id");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");

    let _ = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        let mut record = usage_record("stable-key", "2026-05-01T00:00:00Z", 42);
        record.entry.session_id = Some("conversation-42".to_string());
        vec![record]
    });
    let records = super::load_vendor_cached_records(&cache_root, "test");

    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].entry.session_id.as_deref(),
        Some("conversation-42")
    );
}

#[test]
fn cache_before_session_ids_remains_readable() {
    let cache_root = unique_temp_dir("before-session-id");
    let entries_dir = cache_root.join(super::ENTRIES_DIR);
    fs::create_dir_all(&entries_dir).expect("create entries directory");

    let legacy = super::PersistedVendorRecordsBeforeSession {
        format_version: super::CACHE_VERSION,
        records: vec![super::PersistedSourceRecordBeforeSession {
            source_path: "source.jsonl".to_string(),
            dedup_key: "stable-key".to_string(),
            timestamp: "2026-05-01T00:00:00Z".to_string(),
            session_start_time: "2026-05-01T00:00:00Z".to_string(),
            session_end_time: "2026-05-01T00:00:00Z".to_string(),
            model: "test-model".to_string(),
            effort: None,
            input_tokens: 42,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning_output_tokens: 0,
            fast_tier: UNKNOWN_FAST_TIER,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
        }],
    };
    let payload = bincode::serialize(&legacy).expect("serialize legacy cache");
    let mut content = Vec::new();
    content.extend_from_slice(super::ENTRY_FILE_MAGIC);
    content.extend_from_slice(&super::fnv1a_bytes(0, &payload).to_le_bytes());
    content.extend_from_slice(&payload);
    fs::write(entries_dir.join("claude.bin"), content).expect("write legacy cache");

    let records = super::load_vendor_cached_records(&cache_root, "claude");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].entry.session_id, None);
    assert_eq!(records[0].entry.usage.input_tokens, 42);
}

#[test]
fn manifest_marks_session_metadata_stale_until_current_parser_revision() {
    let cache_root = unique_temp_dir("session-metadata-revision");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");

    let _ = super::load_or_update_vendor_cache(&cache_root, "claude", vec![source], -1, |_| {
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
    });
    assert!(super::vendor_session_metadata_is_current(
        &cache_root,
        "claude"
    ));

    let manifest_path = cache_root.join(super::MANIFEST_FILE);
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
            .expect("parse manifest");
    manifest["vendors"]["claude"]["session_metadata_revision"] = serde_json::json!(0);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write stale manifest");

    assert!(!super::vendor_session_metadata_is_current(
        &cache_root,
        "claude"
    ));

    manifest["vendors"]["claude"]["files"] = serde_json::json!({});
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("serialize empty manifest"),
    )
    .expect("write empty manifest");

    assert!(!super::vendor_session_metadata_is_current(
        &cache_root,
        "claude"
    ));
}

#[test]
fn retained_inactive_sources_do_not_keep_session_metadata_stale() {
    let cache_root = unique_temp_dir("inactive-session-metadata");
    let active = cache_root.join("active.jsonl");
    let retired = cache_root.join("retired.jsonl");
    write_source(&active, "active");
    write_source(&retired, "retired");

    let _ = super::load_or_update_vendor_cache(
        &cache_root,
        "claude",
        vec![active.clone(), retired.clone()],
        -1,
        |_| vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)],
    );

    let manifest_path = cache_root.join(super::MANIFEST_FILE);
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
            .expect("parse manifest");
    let retired_key = fs::canonicalize(&retired)
        .expect("canonical retired path")
        .to_string_lossy()
        .into_owned();
    manifest["vendors"]["claude"]["files"][&retired_key]["parser_revision"] = serde_json::json!(0);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write stale manifest");
    fs::remove_file(&retired).expect("remove retired source");

    super::refresh_retaining_vendor_cache(&cache_root, "claude", vec![active], -1, |_| {
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
    });

    assert!(super::vendor_session_metadata_is_current(
        &cache_root,
        "claude"
    ));
}

#[test]
fn remote_records_are_loaded_with_host_metadata() {
    let cache_root = unique_temp_dir("remote-load");

    super::merge_remote_records(
        &cache_root,
        "laptop",
        vec![
            remote_record("claude", "a", "2026-05-01T00:00:00Z", 10),
            remote_record("codex", "b", "2026-05-01T00:01:00Z", 20),
        ],
    )
    .expect("merge remote records");

    let records = super::load_remote_entries(&cache_root, None);

    assert_eq!(
        remote_fingerprints(&records),
        vec![
            (
                "claude".to_string(),
                "a".to_string(),
                Some("laptop".to_string()),
                10
            ),
            (
                "codex".to_string(),
                "b".to_string(),
                Some("laptop".to_string()),
                20
            ),
        ]
    );
}

#[test]
fn recent_remote_load_streams_current_format_without_full_history_decode() {
    let cache_root = unique_temp_dir("remote-recent-streaming-fast-path");
    super::merge_remote_records(
        &cache_root,
        "laptop",
        vec![
            remote_record("claude", "old", "2026-05-01T00:00:00Z", 1),
            remote_record("claude", "recent", "2026-05-03T00:00:00Z", 2),
        ],
    )
    .expect("merge remote records");
    super::REMOTE_RECORD_READS.set(0);
    let cutoff = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("cutoff");

    let records = super::load_recent_remote_entries(&cache_root, None, cutoff);

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].dedup_key, "recent");
    assert_eq!(records[0].entry.usage.input_tokens, 2);
    assert_eq!(super::REMOTE_RECORD_READS.get(), 0);
}

#[test]
fn full_remote_load_indexes_a_canonical_pre_index_cache() {
    let cache_root = unique_temp_dir("remote-index-migration");
    let remote_dir = cache_root.join(super::REMOTE_DIR);
    fs::create_dir_all(&remote_dir).expect("create remote dir");
    let data_path = remote_dir.join("laptop.bin");
    let payload = bincode::serialize(&super::PersistedRemoteRecords {
        format_version: super::CACHE_VERSION,
        records: vec![super::PersistedRemoteRecord::from_remote_record(
            remote_record("claude", "recent", "2026-05-03T00:00:00Z", 2),
        )],
    })
    .expect("serialize remote records");
    write_payload_file(&data_path, super::REMOTE_FILE_MAGIC, &payload);
    assert!(!data_path.with_extension("idx").exists());

    let records = super::load_remote_entries(&cache_root, None);

    assert_eq!(records.len(), 1);
    assert!(data_path.with_extension("idx").is_file());
}

#[test]
fn recent_remote_load_reads_window_records_through_the_index() {
    let cache_root = unique_temp_dir("remote-recent-indexed");
    let mut records = (0..5_000)
        .map(|index| {
            remote_record(
                "claude",
                &format!("old-{index}"),
                "2025-01-01T00:00:00Z",
                index,
            )
        })
        .collect::<Vec<_>>();
    records.push(remote_record(
        "claude",
        "recent",
        "2026-05-03T00:00:00Z",
        9_999,
    ));
    super::merge_remote_records(&cache_root, "laptop", records).expect("merge remote records");
    let data_path = cache_root.join(super::REMOTE_DIR).join("laptop.bin");
    let index_path = data_path.with_extension("idx");
    let data_bytes = fs::metadata(&data_path).expect("cache metadata").len();
    assert!(index_path.is_file());
    super::INDEXED_CACHE_BYTES_READ.set(0);
    let cutoff = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("cutoff");

    let loaded = super::load_recent_remote_entries(&cache_root, None, cutoff);

    assert_eq!(loaded.len(), 1);
    let indexed_bytes = super::INDEXED_CACHE_BYTES_READ.get();
    assert!(indexed_bytes > 0);
    assert!(indexed_bytes < data_bytes / 100);
}

#[test]
fn recent_remote_load_uses_old_omp_aliases_when_deduplicating() {
    let cache_root = unique_temp_dir("remote-recent-cross-cutoff-aliases");
    let remote_dir = cache_root.join(super::REMOTE_DIR);
    fs::create_dir_all(&remote_dir).expect("create remote dir");
    let payload = bincode::serialize(&super::PersistedRemoteRecords {
        format_version: super::CACHE_VERSION,
        records: vec![
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                "omp:message:old-message:response:old-response",
                "2026-05-01T00:00:00Z",
                10,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                &omp_v220_key("old-message", "old-response", "remote-model", 10),
                "2026-05-03T00:00:00Z",
                10,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                "omp:file:/tmp/old.jsonl:0",
                "2026-05-01T00:01:00Z",
                20,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                &omp_v220_key("", "", "remote-model", 20),
                "2026-05-03T00:01:00Z",
                20,
            )),
        ],
    })
    .expect("serialize remote records");
    write_payload_file(
        &remote_dir.join("laptop.bin"),
        super::REMOTE_FILE_MAGIC,
        &payload,
    );
    let full_records = super::load_remote_entries(&cache_root, None);
    assert_eq!(full_records.len(), 2);
    assert!(remote_dir.join("laptop.idx").is_file());
    super::REMOTE_RECORD_READS.set(0);
    let cutoff = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("cutoff");

    let records = super::load_recent_remote_entries(&cache_root, None, cutoff);

    assert!(records.is_empty());
    assert_eq!(super::REMOTE_RECORD_READS.get(), 0);
}

#[test]
fn remote_load_suppresses_existing_omp_alias_duplicates() {
    let cache_root = unique_temp_dir("remote-omp-alias-dedup");
    let remote_dir = cache_root.join(super::REMOTE_DIR);
    fs::create_dir_all(&remote_dir).expect("create remote dir");
    let payload = bincode::serialize(&super::PersistedRemoteRecords {
        format_version: super::CACHE_VERSION,
        records: vec![
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                &omp_v220_key("msg-a", "resp-a", "remote-model", 10),
                "2026-05-01T00:00:00Z",
                10,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                "omp:message:msg-a:response:resp-a",
                "2026-05-01T00:00:00Z",
                10,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                &omp_v220_key("", "", "remote-model", 20),
                "2026-05-01T00:01:00Z",
                20,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                "omp:file:/tmp/omp.jsonl:0",
                "2026-05-01T00:01:00Z",
                20,
            )),
            super::PersistedRemoteRecord::from_remote_record(remote_record(
                "omp",
                "omp:file:/tmp/omp.jsonl:1",
                "2026-05-01T00:02:00Z",
                20,
            )),
        ],
    })
    .expect("serialize remote records");
    write_payload_file(
        &remote_dir.join("laptop.bin"),
        super::REMOTE_FILE_MAGIC,
        &payload,
    );

    let records = super::load_remote_entries(&cache_root, None);

    assert!(remote_dir.join("laptop.idx").is_file());
    assert_eq!(
        remote_fingerprints(&records),
        vec![
            (
                "omp".to_string(),
                "omp:message:msg-a:response:resp-a".to_string(),
                Some("laptop".to_string()),
                10
            ),
            (
                "omp".to_string(),
                "omp:file:/tmp/omp.jsonl:0".to_string(),
                Some("laptop".to_string()),
                20
            ),
            (
                "omp".to_string(),
                "omp:file:/tmp/omp.jsonl:1".to_string(),
                Some("laptop".to_string()),
                20
            ),
        ]
    );
}

#[test]
fn remote_records_deduplicate_by_vendor_and_key() {
    let cache_root = unique_temp_dir("remote-dedup");

    super::merge_remote_records(
        &cache_root,
        "laptop",
        vec![
            remote_record("claude", "same", "2026-05-01T00:00:00Z", 10),
            remote_record("codex", "same", "2026-05-01T00:01:00Z", 20),
        ],
    )
    .expect("merge first batch");
    super::merge_remote_records(
        &cache_root,
        "laptop",
        vec![remote_record("claude", "same", "2026-05-01T00:02:00Z", 99)],
    )
    .expect("merge duplicate batch");

    let records = super::load_remote_entries(&cache_root, None);

    assert_eq!(
        remote_fingerprints(&records),
        vec![
            (
                "claude".to_string(),
                "same".to_string(),
                Some("laptop".to_string()),
                99
            ),
            (
                "codex".to_string(),
                "same".to_string(),
                Some("laptop".to_string()),
                20
            ),
        ]
    );
}

#[test]
fn remote_host_ids_are_derived_without_decoding_cache_contents() {
    let cache_root = unique_temp_dir("remote-host-ids");
    let remote_dir = cache_root.join(super::REMOTE_DIR);
    fs::create_dir_all(&remote_dir).expect("create remote dir");
    fs::write(remote_dir.join("zeta.bin"), b"not a cache").expect("write host file");
    fs::write(remote_dir.join("alpha.bin"), b"not a cache").expect("write host file");
    fs::write(remote_dir.join("alpha.idx"), b"derived").expect("write index file");
    super::REMOTE_RECORD_READS.set(0);

    let hosts = super::remote_host_ids(&cache_root);

    assert_eq!(hosts, vec!["alpha".to_string(), "zeta".to_string()]);
    assert_eq!(super::REMOTE_RECORD_READS.get(), 0);
}

#[test]
fn remote_omp_stable_records_refresh_existing_metadata() {
    let cache_root = unique_temp_dir("remote-omp-metadata-refresh");
    let stale = remote_record(
        "omp",
        "omp:message:msg-a:response:resp-a",
        "2026-05-01T00:00:00Z",
        10,
    );
    super::merge_remote_records(&cache_root, "laptop", vec![stale])
        .expect("merge stale remote record");

    let mut refreshed = remote_record(
        "omp",
        "omp:message:msg-a:response:resp-a",
        "2026-05-01T00:00:00Z",
        10,
    );
    refreshed.entry.effort = Some("anthropic".to_string());
    refreshed.entry.costs = Some(crate::data::UsageCost {
        input: 0.01,
        output: 0.02,
        cache_read: 0.03,
        cache_creation: 0.04,
    });
    super::merge_remote_records(&cache_root, "laptop", vec![refreshed])
        .expect("merge refreshed remote record");

    let records = super::load_remote_entries(&cache_root, None);

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].dedup_key, "omp:message:msg-a:response:resp-a");
    assert_eq!(records[0].entry.effort.as_deref(), Some("anthropic"));
    let costs = records[0].entry.costs.expect("refreshed costs");
    assert_eq!(costs.input, 0.01);
    assert_eq!(costs.output, 0.02);
    assert_eq!(costs.cache_read, 0.03);
    assert_eq!(costs.cache_creation, 0.04);
}

#[test]
fn clear_remote_cache_drops_pulled_records_and_leaves_local_cache() {
    let cache_root = unique_temp_dir("clear-remote");
    super::merge_remote_records(
        &cache_root,
        "laptop",
        vec![remote_record("claude", "a", "2026-05-01T00:00:00Z", 10)],
    )
    .expect("merge laptop");
    super::merge_remote_records(
        &cache_root,
        "workstation",
        vec![remote_record("claude", "b", "2026-05-01T00:01:00Z", 20)],
    )
    .expect("merge workstation");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let _ = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        vec![usage_record("local-key", "2026-05-01T00:00:00Z", 7)]
    });

    let removed = super::clear_remote_cache(&cache_root).expect("clear");

    assert_eq!(removed, 2);
    assert!(super::load_remote_entries(&cache_root, None).is_empty());
    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].usage.input_tokens, 7);

    let removed_again = super::clear_remote_cache(&cache_root).expect("clear again");
    assert_eq!(removed_again, 0);
}

#[test]
fn clear_remote_cache_on_missing_directory_returns_zero() {
    let cache_root = unique_temp_dir("clear-remote-missing");

    assert_eq!(super::clear_remote_cache(&cache_root).expect("clear"), 0);
}

#[test]
fn remote_load_honors_host_filter() {
    let cache_root = unique_temp_dir("remote-filter");
    super::merge_remote_records(
        &cache_root,
        "laptop",
        vec![remote_record("claude", "a", "2026-05-01T00:00:00Z", 10)],
    )
    .expect("merge laptop");
    super::merge_remote_records(
        &cache_root,
        "workstation",
        vec![remote_record("claude", "b", "2026-05-01T00:01:00Z", 20)],
    )
    .expect("merge workstation");
    let filter = HashSet::from(["workstation".to_string()]);

    let records = super::load_remote_entries(&cache_root, Some(&filter));

    assert_eq!(
        remote_fingerprints(&records),
        vec![(
            "claude".to_string(),
            "b".to_string(),
            Some("workstation".to_string()),
            20,
        )]
    );
}

#[test]
fn unchanged_source_file_reuses_cached_records_without_reparsing() {
    let cache_root = unique_temp_dir("reuse");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let first =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], 1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });
    let second = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], 0, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 99)]
    });

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].usage.input_tokens, 42);
    assert_eq!(second[0].fast_tier, 1);
}

#[test]
fn unchanged_retaining_refresh_does_not_decode_the_entry_cache() {
    let cache_root = unique_temp_dir("retaining-refresh-fast-path");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "claude", vec![source.clone()], -1, |_| {
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });
    super::CACHED_RECORD_READS.set(0);
    super::INDEX_FULL_VALIDATIONS.set(0);

    super::refresh_retaining_vendor_cache(&cache_root, "claude", vec![source], -1, |_| {
        panic!("unchanged source should not be parsed")
    });

    assert_eq!(super::CACHED_RECORD_READS.get(), 0);
    assert_eq!(super::INDEX_FULL_VALIDATIONS.get(), 0);
}

#[test]
fn recent_cache_load_streams_current_format_without_full_history_decode() {
    let cache_root = unique_temp_dir("recent-streaming-fast-path");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "records");
    let _ = super::load_or_update_vendor_cache(&cache_root, "claude", vec![source], -1, |_| {
        vec![
            usage_record("old", "2026-05-01T00:00:00Z", 1),
            usage_record("recent", "2026-05-03T00:00:00Z", 2),
        ]
    });
    super::CACHED_RECORD_READS.set(0);
    let cutoff = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("cutoff");

    let (records, has_cached_records) =
        super::load_recent_vendor_cached_records(&cache_root, "claude", cutoff);

    assert!(has_cached_records);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].0, "recent");
    assert_eq!(records[0].1.usage.input_tokens, 2);
    assert_eq!(super::CACHED_RECORD_READS.get(), 0);
}

#[test]
fn bounded_cache_load_reads_only_window_records_through_the_index() {
    let cache_root = unique_temp_dir("recent-indexed");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "records");
    let _ = super::load_or_update_vendor_cache(&cache_root, "claude", vec![source], -1, |_| {
        let base =
            crate::time_utils::parse_timestamp("2026-04-01T00:00:00Z").expect("history base");
        let mut records = (0..10_000)
            .map(|index| {
                let timestamp = (base - chrono::Duration::days(index + 1)).to_rfc3339();
                usage_record(&format!("old-{index}"), &timestamp, index)
            })
            .collect::<Vec<_>>();
        records.push(usage_record("recent", "2026-05-03T00:00:00Z", 99_999));
        records.push(usage_record("future", "2026-07-03T00:00:00Z", 100_000));
        records
    });
    let data_path = cache_root.join(super::ENTRIES_DIR).join("claude.bin");
    let index_path = data_path.with_extension("idx");
    let data_bytes = fs::metadata(&data_path).expect("cache metadata").len();
    assert!(index_path.is_file());
    super::INDEXED_CACHE_BYTES_READ.set(0);
    let cutoff = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("cutoff");
    let end = crate::time_utils::parse_timestamp("2026-05-04T00:00:00Z").expect("end");

    let (loaded, has_records) =
        super::load_vendor_cached_records_in_range(&cache_root, "claude", cutoff, end);

    assert!(has_records);
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, "recent");
    let indexed_bytes = super::INDEXED_CACHE_BYTES_READ.get();
    assert!(indexed_bytes > 0);
    assert!(indexed_bytes < data_bytes / 100);
}

#[test]
fn bounded_cache_preserves_global_first_key_ownership() {
    let cache_root = unique_temp_dir("bounded-global-dedup");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "records");
    let _ = super::load_or_update_vendor_cache(&cache_root, "claude", vec![source], -1, |_| {
        vec![
            usage_record("shared", "2026-05-01T00:00:00Z", 1),
            usage_record("shared", "2026-05-03T00:00:00Z", 2),
        ]
    });
    let start = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("window start");
    let end = crate::time_utils::parse_timestamp("2026-05-04T00:00:00Z").expect("window end");
    let full_then_filtered = super::load_vendor_cached_records(&cache_root, "claude")
        .into_iter()
        .filter(|record| {
            crate::time_utils::parse_timestamp(&record.entry.timestamp)
                .is_some_and(|timestamp| timestamp >= start && timestamp <= end)
        })
        .collect::<Vec<_>>();

    let (bounded, _) =
        super::load_vendor_cached_records_in_range(&cache_root, "claude", start, end);

    assert!(full_then_filtered.is_empty());
    assert!(bounded.is_empty());
}

#[test]
fn damaged_record_index_falls_back_once_without_deleting_a_new_generation() {
    let cache_root = unique_temp_dir("damaged-record-index");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "records");
    let _ =
        super::load_or_update_vendor_cache(&cache_root, "claude", vec![source.clone()], -1, |_| {
            vec![
                usage_record("old", "2026-05-01T00:00:00Z", 1),
                usage_record("recent", "2026-05-03T00:00:00Z", 2),
            ]
        });
    let index_path = cache_root.join(super::ENTRIES_DIR).join("claude.idx");
    let mut index_bytes = fs::read(&index_path).expect("read record index");
    *index_bytes.last_mut().expect("nonempty record index") ^= 0xff;
    fs::write(&index_path, &index_bytes).expect("damage record index");
    let cutoff = crate::time_utils::parse_timestamp("2026-05-02T00:00:00Z").expect("cutoff");
    super::CACHED_RECORD_READS.set(0);

    let (loaded, has_records) =
        super::load_recent_vendor_cached_records(&cache_root, "claude", cutoff);

    assert!(has_records);
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, "recent");
    assert_eq!(super::CACHED_RECORD_READS.get(), 1);
    assert!(index_path.is_file());
    assert_ne!(fs::read(&index_path).expect("repaired index"), index_bytes);

    super::CACHED_RECORD_READS.set(0);
    let (loaded_again, _) = super::load_recent_vendor_cached_records(&cache_root, "claude", cutoff);
    assert_eq!(loaded_again.len(), 1);
    assert_eq!(super::CACHED_RECORD_READS.get(), 0);
}

#[test]
fn retaining_refresh_rebuilds_a_tampered_entry_cache() {
    let cache_root = unique_temp_dir("retaining-refresh-tampered-cache");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    super::refresh_retaining_vendor_cache(&cache_root, "claude", vec![source.clone()], -1, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
    });
    let entries_path = cache_root.join("entries").join("claude.bin");
    let mut content = fs::read(&entries_path).expect("read cache");
    let last = content.last_mut().expect("nonempty cache");
    *last = last.wrapping_add(1);
    fs::write(&entries_path, content).expect("tamper cache");

    super::refresh_retaining_vendor_cache(&cache_root, "claude", vec![source], -1, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 77)]
    });

    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "claude");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(entry_tokens(&snapshot), vec![77]);
}

#[test]
fn old_omp_manifest_reparses_unchanged_sources_after_parser_change() {
    let cache_root = unique_temp_dir("omp-provider-refresh");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "omp", vec![source.clone()], -1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            let mut record = usage_record("omp:message:msg-a", "2026-05-01T00:00:00Z", 42);
            record.entry.model = "claude-sonnet-4-5-20250929".to_string();
            record.entry.effort = None;
            vec![record]
        });
    let manifest_path = cache_root.join(super::MANIFEST_FILE);
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
            .expect("manifest json");
    manifest["vendors"]["omp"]["files"]
        .as_object_mut()
        .expect("files object")
        .values_mut()
        .for_each(|meta| meta["parser_revision"] = serde_json::json!(0));
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest json"),
    )
    .expect("write old manifest");

    let refreshed =
        super::load_or_update_vendor_cache(&cache_root, "omp", vec![source], -1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            let mut record = usage_record("omp:message:msg-a", "2026-05-01T00:00:00Z", 42);
            record.entry.model = "claude-sonnet-4-5-20250929".to_string();
            record.entry.effort = None;
            vec![record]
        });

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].effort.as_deref(), None);
}

#[test]
fn changed_source_file_replaces_cached_records() {
    let cache_root = unique_temp_dir("changed");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], 1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });
    std::thread::sleep(Duration::from_millis(2));
    write_source(&source, "second-content");

    let refreshed =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source], 0, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 99)]
        });

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].usage.input_tokens, 99);
    assert_eq!(refreshed[0].fast_tier, 0);
}

#[test]
fn changed_source_file_preserves_matching_record_fast_tier() {
    let cache_root = unique_temp_dir("changed-preserve-fast-tier");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], 1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });
    std::thread::sleep(Duration::from_millis(2));
    write_source(&source, "second-content");

    let refreshed =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source], 0, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].fast_tier, 1);
}

#[test]
fn damaged_entry_cache_triggers_vendor_rebuild() {
    let cache_root = unique_temp_dir("damaged");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], -1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });
    fs::write(cache_root.join("entries").join("test.bin"), b"not binary").expect("damage cache");

    let rebuilt = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 77)]
    });

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(rebuilt.len(), 1);
    assert_eq!(rebuilt[0].usage.input_tokens, 77);
}

#[test]
fn large_record_rebuild_publishes_entries_and_manifest() {
    let cache_root = unique_temp_dir("large-record-rebuild");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let model_len = 16 * 1024 * 1024 + 1;

    let _ = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        let mut record = usage_record("large", "2026-05-01T00:00:00Z", 42);
        record.entry.model = "x".repeat(model_len);
        vec![record]
    });

    let manifest = super::read_manifest(&cache_root.join(super::MANIFEST_FILE));
    let start = crate::time_utils::parse_timestamp("2026-05-01T00:00:00Z").expect("window start");
    let end = start + chrono::Duration::days(1);
    super::CACHED_RECORD_READS.set(0);
    let (loaded, has_records) =
        super::load_vendor_cached_records_in_range(&cache_root, "test", start, end);
    assert!(manifest.vendors.contains_key("test"));
    assert!(has_records);
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].1.model.len(), model_len);
    assert_eq!(super::CACHED_RECORD_READS.get(), 0);
}

#[test]
fn large_incremental_record_publishes_the_new_generation() {
    let cache_root = unique_temp_dir("large-record-incremental");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], -1, |_| {
            vec![usage_record("small", "2026-05-01T00:00:00Z", 42)]
        });
    write_source(&source, "second-content");
    let active_sources = super::current_sources(vec![source.clone()]);
    let model_len = 16 * 1024 * 1024 + 1;

    let _ = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        let mut record = usage_record("large", "2026-05-01T00:00:00Z", 77);
        record.entry.model = "x".repeat(model_len);
        vec![record]
    });

    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");
    assert!(super::retaining_vendor_cache_is_current(
        &cache_root,
        "test",
        &active_sources,
    ));
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].usage.input_tokens, 77);
    assert_eq!(snapshot[0].model.len(), model_len);
}

#[test]
fn failed_rebuild_does_not_publish_a_vendor_manifest() {
    let cache_root = unique_temp_dir("failed-rebuild-publication");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    fs::create_dir_all(cache_root.join("entries").join("test.bin"))
        .expect("create conflicting cache path");

    let _ = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        vec![usage_record("stable", "2026-05-01T00:00:00Z", 42)]
    });

    let manifest = super::read_manifest(&cache_root.join(super::MANIFEST_FILE));
    assert!(!manifest.vendors.contains_key("test"));
}

#[test]
fn failed_incremental_write_keeps_the_previous_manifest_generation() {
    let cache_root = unique_temp_dir("failed-incremental-publication");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], -1, |_| {
            vec![usage_record("stable", "2026-05-01T00:00:00Z", 42)]
        });
    let manifest_path = cache_root.join(super::MANIFEST_FILE);
    let previous_manifest = fs::read_to_string(&manifest_path).expect("read manifest");
    let entries_path = cache_root.join("entries").join("test.bin");
    fs::remove_file(&entries_path).expect("remove cache file");
    fs::create_dir(&entries_path).expect("create conflicting cache path");
    write_source(&source, "second-content");

    let _ = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        vec![usage_record("stable", "2026-05-01T00:00:00Z", 77)]
    });

    assert_eq!(
        fs::read_to_string(manifest_path).expect("read unchanged manifest"),
        previous_manifest
    );
}

#[test]
fn valid_but_tampered_entry_cache_triggers_vendor_rebuild() {
    let cache_root = unique_temp_dir("tampered");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], -1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });

    let entries_path = cache_root.join("entries").join("test.bin");
    let mut original = fs::read(&entries_path).expect("read cache");
    let last = original.last_mut().expect("nonempty cache");
    *last = last.wrapping_add(1);
    fs::write(&entries_path, original).expect("tamper cache");

    let rebuilt = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 77)]
    });

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(entry_tokens(&rebuilt), vec![77]);
}

#[test]
fn cached_records_with_negative_tokens_trigger_vendor_rebuild() {
    let cache_root = unique_temp_dir("negative-tokens");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let calls = AtomicUsize::new(0);

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], -1, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            let mut record = usage_record("stable-key", "2026-05-01T00:00:00Z", 42);
            record.entry.usage.output_tokens = -1;
            vec![record]
        });

    let rebuilt = super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 77)]
    });

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(entry_tokens(&rebuilt), vec![77]);
    assert_eq!(rebuilt[0].usage.output_tokens, 2);
}

#[test]
fn nonempty_dedup_keys_are_counted_once_across_files() {
    let cache_root = unique_temp_dir("dedup");
    let first_source = cache_root.join("a.jsonl");
    let second_source = cache_root.join("b.jsonl");
    write_source(&first_source, "first");
    write_source(&second_source, "second");

    let entries = super::load_or_update_vendor_cache(
        &cache_root,
        "test",
        vec![first_source.clone(), second_source.clone()],
        -1,
        |path| {
            let input = if path == first_source { 10 } else { 20 };
            vec![usage_record("same-key", "2026-05-01T00:00:00Z", input)]
        },
    );

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].usage.input_tokens, 10);
}

#[test]
fn warm_cache_matches_direct_uncached_parse_order_and_dedup() {
    let cache_root = unique_temp_dir("equivalence");
    let first_source = cache_root.join("a.jsonl");
    let second_source = cache_root.join("b.jsonl");
    write_source(&first_source, "first");
    write_source(&second_source, "second");
    let source_files = vec![first_source.clone(), second_source.clone()];

    let parse = |path: &Path| {
        if path == first_source {
            vec![
                usage_record("first", "2026-05-01T00:00:00Z", 1),
                usage_record("shared", "2026-05-01T00:01:00Z", 10),
            ]
        } else {
            vec![
                usage_record("shared", "2026-05-01T00:02:00Z", 20),
                usage_record("", "2026-05-01T00:03:00Z", 30),
                usage_record("", "2026-05-01T00:04:00Z", 40),
            ]
        }
    };

    let direct = direct_parse(source_files.clone(), parse);
    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", source_files.clone(), -1, parse);
    let cached = super::load_or_update_vendor_cache(&cache_root, "test", source_files, -1, |_| {
        panic!("unchanged files should not be reparsed")
    });

    assert_eq!(entry_tokens(&cached), entry_tokens(&direct));
    assert_eq!(entry_tokens(&cached), vec![1, 10, 30, 40]);
}

#[test]
fn full_refresh_and_binary_snapshot_match_direct_uncached_parse_fields() {
    let cache_root = unique_temp_dir("full-equivalence");
    let first_source = cache_root.join("a.jsonl");
    let second_source = cache_root.join("b.jsonl");
    write_source(&first_source, "first");
    write_source(&second_source, "second");
    let source_files = vec![first_source.clone(), second_source.clone()];

    let parse = |path: &Path| {
        if path == first_source {
            let mut first = usage_record("first", "2026-05-01T00:00:00Z", 1);
            first.entry.session_start_time = "2026-05-01T00:00:00Z".to_string();
            first.entry.session_end_time = "2026-05-01T00:05:00Z".to_string();
            first.entry.model = "model-a".to_string();
            first.entry.effort = Some("high".to_string());
            first.entry.usage.output_tokens = 12;
            first.entry.usage.cache_read_input_tokens = 13;
            first.entry.usage.cache_creation_input_tokens = 14;
            first.entry.usage.reasoning_output_tokens = 15;
            vec![first, usage_record("shared", "2026-05-01T00:10:00Z", 2)]
        } else {
            let mut second = usage_record("second", "2026-05-01T00:20:00Z", 3);
            second.entry.model = "model-b".to_string();
            second.entry.effort = Some("low".to_string());
            vec![usage_record("shared", "2026-05-01T00:15:00Z", 99), second]
        }
    };

    let direct = direct_parse(source_files.clone(), parse);
    let refreshed = super::refresh_full_vendor_cache(&cache_root, "test", source_files, -1, parse);
    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert_eq!(entry_fingerprints(&refreshed), entry_fingerprints(&direct));
    assert_eq!(entry_fingerprints(&snapshot), entry_fingerprints(&direct));
}

#[test]
fn repeated_source_paths_match_direct_uncached_parse() {
    let cache_root = unique_temp_dir("repeated");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");
    let source_files = vec![source.clone(), source.clone()];

    let parse = |_path: &Path| vec![usage_record("", "2026-05-01T00:00:00Z", 11)];
    let direct = direct_parse(source_files.clone(), parse);
    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", source_files.clone(), -1, parse);
    let cached = super::load_or_update_vendor_cache(&cache_root, "test", source_files, -1, |_| {
        panic!("unchanged files should not be reparsed")
    });

    assert_eq!(entry_tokens(&cached), entry_tokens(&direct));
    assert_eq!(entry_tokens(&cached), vec![11, 11]);
}

#[test]
fn cached_snapshot_reads_entries_without_source_validation_or_parsing() {
    let cache_root = unique_temp_dir("snapshot");
    let source = cache_root.join("source.jsonl");
    write_source(&source, "first");

    let _ =
        super::load_or_update_vendor_cache(&cache_root, "test", vec![source.clone()], -1, |_| {
            vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
        });
    fs::remove_file(source).expect("remove source");

    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert_eq!(entry_tokens(&snapshot), vec![42]);
}

#[test]
fn old_cached_records_default_to_unknown_fast_tier() {
    let cache_root = unique_temp_dir("snapshot-old-fast-tier");
    fs::create_dir_all(cache_root.join("entries")).expect("create entries dir");
    let payload = bincode::serialize(&super::PersistedVendorRecordsV1 {
        format_version: super::CACHE_VERSION,
        records: vec![super::PersistedSourceRecordV1 {
            source_path: "source.jsonl".to_string(),
            dedup_key: "stable-key".to_string(),
            timestamp: "2026-05-01T00:00:00Z".to_string(),
            session_start_time: "2026-05-01T00:00:00Z".to_string(),
            session_end_time: "2026-05-01T00:00:00Z".to_string(),
            model: "test-model".to_string(),
            effort: None,
            input_tokens: 42,
            output_tokens: 2,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 4,
            reasoning_output_tokens: 5,
        }],
    })
    .expect("serialize old cache");
    let checksum = super::fnv1a_bytes(0, &payload);
    let mut content = Vec::new();
    content.extend_from_slice(super::ENTRY_FILE_MAGIC);
    content.extend_from_slice(&checksum.to_le_bytes());
    content.extend_from_slice(&payload);
    fs::write(cache_root.join("entries").join("test.bin"), content).expect("write old cache");

    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].usage.input_tokens, 42);
    assert_eq!(snapshot[0].fast_tier, UNKNOWN_FAST_TIER);
}

#[test]
fn cache_records_from_fast_tier_format_keep_fast_tier_after_cost_fields() {
    let cache_root = unique_temp_dir("snapshot-fast-tier-compat");
    let entries_dir = cache_root.join("entries");
    fs::create_dir_all(&entries_dir).expect("create entries dir");
    let payload = bincode::serialize(&PersistedVendorRecordsWithFastTier {
        format_version: super::CACHE_VERSION,
        records: vec![PersistedSourceRecordWithFastTier {
            source_path: "source.jsonl".to_string(),
            dedup_key: "stable-key".to_string(),
            timestamp: "2026-05-01T00:00:00Z".to_string(),
            session_start_time: "2026-05-01T00:00:00Z".to_string(),
            session_end_time: "2026-05-01T00:00:00Z".to_string(),
            model: "test-model".to_string(),
            effort: None,
            input_tokens: 42,
            output_tokens: 2,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 4,
            reasoning_output_tokens: 5,
            fast_tier: 1,
        }],
    })
    .expect("serialize fast-tier cache");
    write_payload_file(
        &entries_dir.join("test.bin"),
        super::ENTRY_FILE_MAGIC,
        &payload,
    );

    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].fast_tier, 1);
    assert!(snapshot[0].costs.is_none());
}

#[test]
fn remote_records_from_fast_tier_format_keep_fast_tier_after_cost_fields() {
    let cache_root = unique_temp_dir("remote-fast-tier-compat");
    let remote_dir = cache_root.join("remote");
    fs::create_dir_all(&remote_dir).expect("create remote dir");
    let payload = bincode::serialize(&PersistedRemoteRecordsWithFastTier {
        format_version: super::CACHE_VERSION,
        records: vec![PersistedRemoteRecordWithFastTier {
            vendor: "codex".to_string(),
            dedup_key: "remote-key".to_string(),
            timestamp: "2026-05-01T00:00:00Z".to_string(),
            session_start_time: "2026-05-01T00:00:00Z".to_string(),
            session_end_time: "2026-05-01T00:00:00Z".to_string(),
            model: "test-model".to_string(),
            effort: None,
            input_tokens: 42,
            output_tokens: 2,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 4,
            reasoning_output_tokens: 5,
            fast_tier: 1,
        }],
    })
    .expect("serialize fast-tier remote cache");
    write_payload_file(
        &remote_dir.join("laptop.bin"),
        super::REMOTE_FILE_MAGIC,
        &payload,
    );

    let records = super::load_remote_entries(&cache_root, None);

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].entry.host_id.as_deref(), Some("laptop"));
    assert_eq!(records[0].entry.fast_tier, 1);
    assert!(records[0].entry.costs.is_none());
}

#[test]
fn damaged_cached_snapshot_returns_empty_without_panicking() {
    let cache_root = unique_temp_dir("snapshot-damaged");
    fs::create_dir_all(cache_root.join("entries")).expect("create entries dir");
    fs::write(cache_root.join("entries").join("test.bin"), b"not binary").expect("damage cache");

    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert!(snapshot.is_empty());
}

#[test]
fn full_refresh_purges_records_for_deleted_sources() {
    let cache_root = unique_temp_dir("purge-deleted");
    let first_source = cache_root.join("a.jsonl");
    let second_source = cache_root.join("b.jsonl");
    write_source(&first_source, "first");
    write_source(&second_source, "second");

    let _ = super::refresh_full_vendor_cache(
        &cache_root,
        "test",
        vec![first_source.clone(), second_source.clone()],
        -1,
        |path| {
            if path == first_source {
                vec![usage_record("first", "2026-05-01T00:00:00Z", 1)]
            } else {
                vec![usage_record("second", "2026-05-01T00:01:00Z", 2)]
            }
        },
    );
    fs::remove_file(&second_source).expect("remove source");

    let refreshed =
        super::refresh_full_vendor_cache(&cache_root, "test", vec![first_source], -1, |_| {
            vec![usage_record("first", "2026-05-01T00:00:00Z", 1)]
        });
    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert_eq!(entry_tokens(&refreshed), vec![1]);
    assert_eq!(entry_tokens(&snapshot), vec![1]);
}

#[test]
fn retaining_refresh_keeps_records_for_deleted_sources_in_cache() {
    let cache_root = unique_temp_dir("retain-deleted");
    let first_source = cache_root.join("a.jsonl");
    let second_source = cache_root.join("b.jsonl");
    write_source(&first_source, "first");
    write_source(&second_source, "second");

    super::refresh_retaining_vendor_cache(
        &cache_root,
        "test",
        vec![first_source.clone(), second_source.clone()],
        -1,
        |path| {
            if path == first_source {
                vec![usage_record("first", "2026-05-01T00:00:00Z", 1)]
            } else {
                vec![usage_record("second", "2026-05-01T00:01:00Z", 2)]
            }
        },
    );
    fs::remove_file(&second_source).expect("remove source");

    super::refresh_retaining_vendor_cache(&cache_root, "test", vec![first_source], -1, |_| {
        vec![usage_record("first", "2026-05-01T00:00:00Z", 1)]
    });
    let snapshot = super::load_vendor_cached_snapshot(&cache_root, "test");

    assert_eq!(entry_tokens(&snapshot), vec![1, 2]);
}
