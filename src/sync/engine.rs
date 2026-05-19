use crate::data::cache::{self, CachedUsageRecord, RemoteUsageRecord};
use crate::data::{TokenUsage, UsageEntry};
use crate::sync::config::EnabledSyncConfig;
use crate::sync::state;
use crate::time_utils::parse_timestamp;
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use vibe_usage_proto::{PullResponse, UploadResponse, WireRecord};

const VENDORS: [&str; 3] = ["claude", "codex", "gemini"];
const BATCH_SIZE: usize = 1000;
const PULL_LIMIT: usize = 5000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncError {
    message: String,
}

impl SyncError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SyncError {}

impl From<std::io::Error> for SyncError {
    fn from(err: std::io::Error) -> Self {
        Self::new(err.to_string())
    }
}

pub trait SyncTransport {
    fn upload(&self, records: &[WireRecord]) -> Result<UploadResponse, SyncError>;
    fn pull(
        &self,
        after_seq: u64,
        exclude_host: &str,
        limit: usize,
    ) -> Result<PullResponse, SyncError>;
}

pub fn run_sync_cycle(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
) -> Result<(), SyncError> {
    let mut sync_state = state::load_sync_state(cache_root);
    let mut upload_log = state::load_upload_log(cache_root);
    let mut upload_records = Vec::new();

    for vendor in VENDORS {
        for record in cache::load_vendor_cached_records(cache_root, vendor) {
            if record.dedup_key.is_empty() {
                continue;
            }
            let key = (record.vendor.clone(), record.dedup_key.clone());
            if upload_log.contains(&key) {
                continue;
            }
            let wire = cached_record_to_wire(config, &record)?;
            upload_records.push((key, wire));
        }
    }

    for batch in upload_records.chunks(BATCH_SIZE) {
        let wire_records: Vec<WireRecord> =
            batch.iter().map(|(_, record)| record.clone()).collect();
        transport.upload(&wire_records)?;
        for (key, _) in batch {
            upload_log.insert(key.clone());
        }
        state::save_upload_log(cache_root, &upload_log)?;
    }

    loop {
        let response = transport.pull(sync_state.last_seen_seq, &config.machine_id, PULL_LIMIT)?;
        merge_pulled_records(cache_root, &response)?;
        sync_state.last_seen_seq = response.max_seq;
        state::save_sync_state(cache_root, &sync_state)?;
        if !response.truncated {
            break;
        }
    }

    sync_state.last_successful_sync = Some(Utc::now().to_rfc3339());
    sync_state.last_error = None;
    state::save_sync_state(cache_root, &sync_state)?;
    Ok(())
}

fn cached_record_to_wire(
    config: &EnabledSyncConfig,
    record: &CachedUsageRecord,
) -> Result<WireRecord, SyncError> {
    let wire = WireRecord {
        schema_version: vibe_usage_proto::SCHEMA_VERSION,
        host_id: config.machine_id.clone(),
        vendor: record.vendor.clone(),
        dedup_key: record.dedup_key.clone(),
        timestamp: record.entry.timestamp.clone(),
        session_start_time: record.entry.session_start_time.clone(),
        session_end_time: record.entry.session_end_time.clone(),
        model: record.entry.model.clone(),
        effort: record.entry.effort.clone(),
        input_tokens: record.entry.usage.input_tokens,
        output_tokens: record.entry.usage.output_tokens,
        cache_read_input_tokens: record.entry.usage.cache_read_input_tokens,
        cache_creation_input_tokens: record.entry.usage.cache_creation_input_tokens,
        reasoning_output_tokens: record.entry.usage.reasoning_output_tokens,
        project_path_sha256: config
            .upload_project_hash
            .then(|| sha256_hex(record.source_path.as_bytes())),
    };
    wire.validate()
        .map_err(|err| SyncError::new(format!("invalid cached record: {err}")))?;
    Ok(wire)
}

fn merge_pulled_records(cache_root: &Path, response: &PullResponse) -> Result<(), SyncError> {
    let mut by_host: BTreeMap<String, Vec<RemoteUsageRecord>> = BTreeMap::new();
    for record in &response.records {
        record
            .validate()
            .map_err(|err| SyncError::new(format!("invalid pulled record: {err}")))?;
        by_host
            .entry(record.record.host_id.clone())
            .or_default()
            .push(wire_to_remote_record(record.record.clone()));
    }
    for (host_id, records) in by_host {
        cache::merge_remote_records(cache_root, &host_id, records)?;
    }
    Ok(())
}

fn wire_to_remote_record(record: WireRecord) -> RemoteUsageRecord {
    RemoteUsageRecord {
        vendor: record.vendor,
        dedup_key: record.dedup_key,
        entry: UsageEntry {
            host_id: Some(record.host_id),
            parsed_timestamp: parse_timestamp(&record.timestamp),
            timestamp: record.timestamp,
            session_start_time: record.session_start_time,
            session_end_time: record.session_end_time,
            model: record.model,
            effort: record.effort,
            usage: TokenUsage {
                input_tokens: record.input_tokens,
                output_tokens: record.output_tokens,
                cache_read_input_tokens: record.cache_read_input_tokens,
                cache_creation_input_tokens: record.cache_creation_input_tokens,
                reasoning_output_tokens: record.reasoning_output_tokens,
            },
        },
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{SourceUsageRecord, TokenUsage, UsageEntry};
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};
    use vibe_usage_proto::{
        PullResponse, SCHEMA_VERSION, SequencedWireRecord, UploadResponse, WireRecord,
    };

    struct FakeTransport {
        uploads: RefCell<Vec<Vec<WireRecord>>>,
        pulls: RefCell<Vec<PullResponse>>,
    }

    impl FakeTransport {
        fn new(pulls: Vec<PullResponse>) -> Self {
            Self {
                uploads: RefCell::new(Vec::new()),
                pulls: RefCell::new(pulls),
            }
        }
    }

    impl SyncTransport for FakeTransport {
        fn upload(&self, records: &[WireRecord]) -> Result<UploadResponse, SyncError> {
            self.uploads.borrow_mut().push(records.to_vec());
            Ok(UploadResponse {
                accepted: records.len(),
                ignored: 0,
                max_seq: 0,
            })
        }

        fn pull(
            &self,
            _after_seq: u64,
            _exclude_host: &str,
            _limit: usize,
        ) -> Result<PullResponse, SyncError> {
            Ok(self.pulls.borrow_mut().remove(0))
        }
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vibe-usage-engine-test-{name}-{stamp}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn usage_record(key: &str, timestamp: &str, input_tokens: i64) -> SourceUsageRecord {
        SourceUsageRecord {
            dedup_key: key.to_string(),
            entry: UsageEntry {
                host_id: None,
                timestamp: timestamp.to_string(),
                parsed_timestamp: crate::time_utils::parse_timestamp(timestamp),
                session_start_time: timestamp.to_string(),
                session_end_time: timestamp.to_string(),
                model: "test-model".to_string(),
                effort: None,
                usage: TokenUsage {
                    input_tokens,
                    output_tokens: 2,
                    cache_read_input_tokens: 3,
                    cache_creation_input_tokens: 4,
                    reasoning_output_tokens: 5,
                },
            },
        }
    }

    fn populate_vendor_cache(cache_root: &Path, vendor: &str, key: &str) {
        let source = cache_root.join(format!("{vendor}.jsonl"));
        std::fs::write(&source, "source").expect("write source");
        crate::data::cache::load_or_update_vendor_cache(cache_root, vendor, vec![source], |_| {
            vec![usage_record(key, "2026-05-18T12:00:00Z", 10)]
        });
    }

    fn enabled_config(machine_id: &str) -> crate::sync::config::EnabledSyncConfig {
        crate::sync::config::EnabledSyncConfig {
            server_url: "https://usage.example.com".to_string(),
            token: "0123456789abcdef0123456789abcdef".to_string(),
            machine_id: machine_id.to_string(),
            upload_project_hash: true,
            request_timeout_seconds: 15,
        }
    }

    #[test]
    fn sync_cycle_uploads_unseen_cached_records_and_updates_upload_log() {
        let cache_root = unique_temp_dir("upload");
        populate_vendor_cache(&cache_root, "claude", "dedup-a");
        let transport = FakeTransport::new(vec![PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        }]);

        run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
            .expect("sync cycle");

        let uploads = transport.uploads.borrow();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].len(), 1);
        assert_eq!(uploads[0][0].host_id, "workstation");
        assert_eq!(uploads[0][0].vendor, "claude");
        assert_eq!(uploads[0][0].dedup_key, "dedup-a");
        assert!(uploads[0][0].project_path_sha256.is_some());
        assert!(
            crate::sync::state::load_upload_log(&cache_root)
                .contains(&("claude".to_string(), "dedup-a".to_string()))
        );
    }

    #[test]
    fn sync_cycle_does_not_reupload_logged_records() {
        let cache_root = unique_temp_dir("skip-upload");
        populate_vendor_cache(&cache_root, "claude", "dedup-a");
        let first = FakeTransport::new(vec![PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        }]);
        run_sync_cycle(&cache_root, &enabled_config("workstation"), &first)
            .expect("first sync cycle");
        let second = FakeTransport::new(vec![PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        }]);

        run_sync_cycle(&cache_root, &enabled_config("workstation"), &second)
            .expect("second sync cycle");

        assert!(second.uploads.borrow().is_empty());
    }

    #[test]
    fn sync_cycle_merges_pulled_records_and_advances_state() {
        let cache_root = unique_temp_dir("pull");
        let pulled = SequencedWireRecord {
            seq: 7,
            uploaded_at: "2026-05-18T12:10:00Z".to_string(),
            record: WireRecord {
                schema_version: SCHEMA_VERSION,
                host_id: "laptop".to_string(),
                vendor: "codex".to_string(),
                dedup_key: "remote-a".to_string(),
                timestamp: "2026-05-18T12:00:00Z".to_string(),
                session_start_time: "2026-05-18T12:00:00Z".to_string(),
                session_end_time: "2026-05-18T12:05:00Z".to_string(),
                model: "remote-model".to_string(),
                effort: Some("high".to_string()),
                input_tokens: 11,
                output_tokens: 12,
                cache_read_input_tokens: 13,
                cache_creation_input_tokens: 14,
                reasoning_output_tokens: 15,
                project_path_sha256: None,
            },
        };
        let transport = FakeTransport::new(vec![PullResponse {
            records: vec![pulled],
            max_seq: 7,
            truncated: false,
        }]);

        run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
            .expect("sync cycle");

        let remote = crate::data::cache::load_remote_entries(&cache_root, None);
        assert_eq!(remote.len(), 1);
        assert_eq!(remote[0].vendor, "codex");
        assert_eq!(remote[0].dedup_key, "remote-a");
        assert_eq!(remote[0].entry.host_id, Some("laptop".to_string()));
        assert_eq!(
            crate::sync::state::load_sync_state(&cache_root).last_seen_seq,
            7
        );
    }
}
