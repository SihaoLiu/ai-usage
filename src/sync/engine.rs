use crate::data::cache::{self, CachedUsageRecord, RemoteUsageRecord};
use crate::data::{TokenUsage, UsageCost, UsageEntry};
use crate::sync::config::EnabledSyncConfig;
use crate::sync::state;
use crate::time_utils::parse_timestamp;
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use vibe_usage_proto::{PullResponse, UploadResponse, WireRecord};

const VENDORS: [&str; 4] = ["claude", "codex", "gemini", "omp"];
const BATCH_SIZE: usize = 1000;
const PULL_LIMIT: usize = 5000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncProgress {
    UploadPlanned {
        total_records: usize,
        total_batches: usize,
        skipped_records: usize,
    },
    UploadBatchFinished {
        batch_index: usize,
        total_batches: usize,
        uploaded_records: usize,
        total_records: usize,
        accepted: usize,
        ignored: usize,
    },
    UploadFinished {
        uploaded_records: usize,
        total_records: usize,
        accepted: usize,
        ignored: usize,
    },
    PullPageFinished {
        page_index: usize,
        page_records: usize,
        pulled_records: usize,
        max_seq: u64,
        truncated: bool,
    },
    PullFinished {
        pages: usize,
        pulled_records: usize,
        max_seq: u64,
    },
}

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

#[cfg(test)]
pub fn run_sync_cycle(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
) -> Result<(), SyncError> {
    run_sync_cycle_with_progress(cache_root, config, transport, |_| {})
}

pub fn run_sync_cycle_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<(), SyncError>
where
    F: FnMut(&SyncProgress),
{
    run_upload_once_with_progress(cache_root, config, transport, &mut on_progress)?;
    run_pull_once_with_progress(cache_root, config, transport, &mut on_progress)
}

pub fn run_upload_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<(), SyncError>
where
    F: FnMut(&SyncProgress),
{
    let mut upload_log = state::load_upload_log(cache_root);
    let mut upload_groups = Vec::new();
    let mut skipped_records = 0;
    let mut consumed_omp_v220_keys = BTreeSet::new();

    for vendor in VENDORS {
        let mut vendor_records = Vec::new();
        for record in cache::load_vendor_cached_records(cache_root, vendor) {
            if record.dedup_key.is_empty() {
                skipped_records += 1;
                continue;
            }
            let key = (record.vendor.clone(), record.dedup_key.clone());
            if upload_log.contains(&key)
                || uploaded_with_omp_v220_key(&record, &upload_log, &mut consumed_omp_v220_keys)
            {
                skipped_records += 1;
                continue;
            }
            let wire = cached_record_to_wire(config, &record)?;
            vendor_records.push((key, wire));
        }
        if !vendor_records.is_empty() {
            upload_groups.push(vendor_records);
        }
    }

    let total_records = upload_groups.iter().map(Vec::len).sum::<usize>();
    let total_batches = upload_groups
        .iter()
        .map(|records| records.len().div_ceil(BATCH_SIZE))
        .sum::<usize>();
    on_progress(&SyncProgress::UploadPlanned {
        total_records,
        total_batches,
        skipped_records,
    });

    let mut uploaded_records = 0;
    let mut accepted = 0;
    let mut ignored = 0;
    let mut batch_index = 0;
    for group in upload_groups {
        for batch in group.chunks(BATCH_SIZE) {
            let wire_records: Vec<WireRecord> =
                batch.iter().map(|(_, record)| record.clone()).collect();
            let response = match transport.upload(&wire_records) {
                Ok(response) => response,
                Err(err)
                    if wire_records.iter().all(|record| record.vendor == "omp")
                        && is_unsupported_vendor_error(&err) =>
                {
                    continue;
                }
                Err(err) => return Err(err),
            };
            uploaded_records += batch.len();
            accepted += response.accepted;
            ignored += response.ignored;
            for (key, _) in batch {
                upload_log.insert(key.clone());
            }
            state::save_upload_log(cache_root, &upload_log)?;
            batch_index += 1;
            on_progress(&SyncProgress::UploadBatchFinished {
                batch_index,
                total_batches,
                uploaded_records,
                total_records,
                accepted,
                ignored,
            });
        }
    }

    on_progress(&SyncProgress::UploadFinished {
        uploaded_records,
        total_records,
        accepted,
        ignored,
    });

    Ok(())
}

fn is_unsupported_vendor_error(err: &SyncError) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("invalid vendor") || message.contains("unsupported vendor")
}

fn uploaded_with_omp_v220_key(
    record: &CachedUsageRecord,
    upload_log: &BTreeSet<(String, String)>,
    consumed_keys: &mut BTreeSet<String>,
) -> bool {
    if record.vendor != "omp" {
        return false;
    }
    let consume_once = record.dedup_key.starts_with("omp:file:");
    for legacy_key in omp_v220_key_candidates(record) {
        if upload_log.contains(&("omp".to_string(), legacy_key.clone()))
            && (!consume_once || consumed_keys.insert(legacy_key))
        {
            return true;
        }
    }
    false
}

fn omp_v220_key_candidates(record: &CachedUsageRecord) -> Vec<String> {
    let (message_id, response_id) = omp_ids_from_dedup_key(&record.dedup_key);
    let mut models = vec![record.entry.model.clone()];
    if let Some(provider) = record
        .entry
        .effort
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        models.push(format!("{provider}/{}", record.entry.model));
    }
    models.sort();
    models.dedup();
    models
        .into_iter()
        .map(|model| omp_v220_key(&message_id, &response_id, &model, &record.entry.usage))
        .collect()
}

fn omp_ids_from_dedup_key(dedup_key: &str) -> (String, String) {
    if let Some(rest) = dedup_key.strip_prefix("omp:message:") {
        if let Some((message_id, response_id)) = rest.split_once(":response:") {
            return (message_id.to_string(), response_id.to_string());
        }
        return (rest.to_string(), String::new());
    }
    if let Some(response_id) = dedup_key.strip_prefix("omp:response:") {
        return (String::new(), response_id.to_string());
    }
    (String::new(), String::new())
}

fn omp_v220_key(message_id: &str, response_id: &str, model: &str, usage: &TokenUsage) -> String {
    serde_json::json!({
        "message": message_id,
        "response": response_id,
        "model": model,
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens,
        "cache_write": usage.cache_creation_input_tokens,
    })
    .to_string()
}

pub fn run_pull_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<(), SyncError>
where
    F: FnMut(&SyncProgress),
{
    let mut sync_state = state::load_sync_state(cache_root);
    let mut page_index = 0;
    let mut pulled_records = 0;

    loop {
        let response = transport.pull(sync_state.last_seen_seq, &config.machine_id, PULL_LIMIT)?;
        page_index += 1;
        pulled_records += response.records.len();
        merge_pulled_records(cache_root, &response)?;
        sync_state.last_seen_seq = response.max_seq;
        state::save_sync_state(cache_root, &sync_state)?;
        on_progress(&SyncProgress::PullPageFinished {
            page_index,
            page_records: response.records.len(),
            pulled_records,
            max_seq: response.max_seq,
            truncated: response.truncated,
        });
        if !response.truncated {
            break;
        }
    }

    sync_state.last_successful_sync = Some(Utc::now().to_rfc3339());
    sync_state.last_error = None;
    state::save_sync_state(cache_root, &sync_state)?;
    on_progress(&SyncProgress::PullFinished {
        pages: page_index,
        pulled_records,
        max_seq: sync_state.last_seen_seq,
    });
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
        fast_tier: record.entry.fast_tier,
        input_tokens: record.entry.usage.input_tokens,
        output_tokens: record.entry.usage.output_tokens,
        cache_read_input_tokens: record.entry.usage.cache_read_input_tokens,
        cache_creation_input_tokens: record.entry.usage.cache_creation_input_tokens,
        reasoning_output_tokens: record.entry.usage.reasoning_output_tokens,
        cost_input: record.entry.costs.map(|costs| costs.input),
        cost_output: record.entry.costs.map(|costs| costs.output),
        cost_cache_read: record.entry.costs.map(|costs| costs.cache_read),
        cost_cache_creation: record.entry.costs.map(|costs| costs.cache_creation),
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
            fast_tier: record.fast_tier,
            usage: TokenUsage {
                input_tokens: record.input_tokens,
                output_tokens: record.output_tokens,
                cache_read_input_tokens: record.cache_read_input_tokens,
                cache_creation_input_tokens: record.cache_creation_input_tokens,
                reasoning_output_tokens: record.reasoning_output_tokens,
            },
            costs: persisted_wire_costs(
                record.cost_input,
                record.cost_output,
                record.cost_cache_read,
                record.cost_cache_creation,
            ),
        },
    }
}

fn persisted_wire_costs(
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_creation: Option<f64>,
) -> Option<UsageCost> {
    match (input, output, cache_read, cache_creation) {
        (None, None, None, None) => None,
        _ => Some(UsageCost {
            input: input.unwrap_or(0.0),
            output: output.unwrap_or(0.0),
            cache_read: cache_read.unwrap_or(0.0),
            cache_creation: cache_creation.unwrap_or(0.0),
        }),
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
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};
    use vibe_usage_proto::{
        PullResponse, SCHEMA_VERSION, SequencedWireRecord, UploadResponse, WireRecord,
    };

    struct FakeTransport {
        uploads: RefCell<Vec<Vec<WireRecord>>>,
        pulls: RefCell<Vec<PullResponse>>,
    }

    struct RejectOmpTransport {
        uploads: RefCell<Vec<Vec<WireRecord>>>,
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

    impl SyncTransport for RejectOmpTransport {
        fn upload(&self, records: &[WireRecord]) -> Result<UploadResponse, SyncError> {
            self.uploads.borrow_mut().push(records.to_vec());
            if records.iter().any(|record| record.vendor == "omp") {
                return Err(SyncError::new("http status: 400: invalid vendor"));
            }
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
            Ok(PullResponse {
                records: Vec::new(),
                max_seq: 0,
                truncated: false,
            })
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
                fast_tier: -1,
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

    fn usage_record_with_costs(key: &str, timestamp: &str) -> SourceUsageRecord {
        let mut record = usage_record(key, timestamp, 10);
        record.entry.costs = Some(UsageCost {
            input: 0.01,
            output: 0.02,
            cache_read: 0.03,
            cache_creation: 0.04,
        });
        record
    }

    fn populate_vendor_cache(cache_root: &Path, vendor: &str, key: &str) {
        let source = cache_root.join(format!("{vendor}.jsonl"));
        std::fs::write(&source, "source").expect("write source");
        crate::data::cache::load_or_update_vendor_cache(
            cache_root,
            vendor,
            vec![source],
            1,
            |_| vec![usage_record(key, "2026-05-18T12:00:00Z", 10)],
        );
    }

    fn populate_vendor_cache_with_record(
        cache_root: &Path,
        vendor: &str,
        record: SourceUsageRecord,
    ) {
        populate_vendor_cache_with_records(cache_root, vendor, vec![record]);
    }

    fn populate_vendor_cache_with_records(
        cache_root: &Path,
        vendor: &str,
        records: Vec<SourceUsageRecord>,
    ) {
        let source = cache_root.join(format!("{vendor}.jsonl"));
        std::fs::write(&source, "source").expect("write source");
        crate::data::cache::load_or_update_vendor_cache(
            cache_root,
            vendor,
            vec![source],
            -1,
            |_| records.clone(),
        );
    }

    fn populate_vendor_cache_with_count(cache_root: &Path, vendor: &str, count: usize) {
        let source = cache_root.join(format!("{vendor}.jsonl"));
        std::fs::write(&source, "source").expect("write source");
        crate::data::cache::load_or_update_vendor_cache(
            cache_root,
            vendor,
            vec![source],
            0,
            |_| {
                (0..count)
                    .map(|idx| usage_record(&format!("dedup-{idx}"), "2026-05-18T12:00:00Z", 10))
                    .collect()
            },
        );
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

    fn omp_v220_key(
        message_id: &str,
        response_id: &str,
        model: &str,
        usage: &TokenUsage,
    ) -> String {
        serde_json::json!({
            "message": message_id,
            "response": response_id,
            "model": model,
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "cache_read": usage.cache_read_input_tokens,
            "cache_write": usage.cache_creation_input_tokens,
        })
        .to_string()
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
        assert_eq!(uploads[0][0].fast_tier, 1);
        assert!(uploads[0][0].project_path_sha256.is_some());
        assert!(
            crate::sync::state::load_upload_log(&cache_root)
                .contains(&("claude".to_string(), "dedup-a".to_string()))
        );
    }

    #[test]
    fn sync_upload_carries_embedded_costs() {
        let cache_root = unique_temp_dir("upload-costs");
        populate_vendor_cache_with_record(
            &cache_root,
            "omp",
            usage_record_with_costs("omp-a", "2026-05-18T12:00:00Z"),
        );
        let transport = FakeTransport::new(Vec::new());

        run_upload_once_with_progress(
            &cache_root,
            &enabled_config("workstation"),
            &transport,
            |_| {},
        )
        .expect("upload");

        let uploads = transport.uploads.borrow();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].len(), 1);
        assert_eq!(uploads[0][0].cost_input, Some(0.01));
        assert_eq!(uploads[0][0].cost_output, Some(0.02));
        assert_eq!(uploads[0][0].cost_cache_read, Some(0.03));
        assert_eq!(uploads[0][0].cost_cache_creation, Some(0.04));
    }

    #[test]
    fn sync_upload_treats_logged_omp_v220_message_key_as_seen() {
        let cache_root = unique_temp_dir("upload-omp-legacy-message");
        let mut record = usage_record(
            "omp:message:msg-a:response:resp-a",
            "2026-05-18T12:00:00Z",
            10,
        );
        record.entry.model = "gpt-5.5".to_string();
        record.entry.effort = Some("openai-codex".to_string());
        populate_vendor_cache_with_record(&cache_root, "omp", record.clone());
        crate::sync::state::save_upload_log(
            &cache_root,
            &BTreeSet::from([(
                "omp".to_string(),
                omp_v220_key(
                    "msg-a",
                    "resp-a",
                    "openai-codex/gpt-5.5",
                    &record.entry.usage,
                ),
            )]),
        )
        .expect("save upload log");
        let transport = FakeTransport::new(Vec::new());

        run_upload_once_with_progress(
            &cache_root,
            &enabled_config("workstation"),
            &transport,
            |_| {},
        )
        .expect("upload");

        assert!(transport.uploads.borrow().is_empty());
    }

    #[test]
    fn sync_upload_consumes_one_logged_omp_v220_file_key() {
        let cache_root = unique_temp_dir("upload-omp-legacy-file");
        let first = usage_record("omp:file:/tmp/omp.jsonl:0", "2026-05-18T12:00:00Z", 10);
        let second = usage_record("omp:file:/tmp/omp.jsonl:1", "2026-05-18T12:01:00Z", 10);
        populate_vendor_cache_with_records(&cache_root, "omp", vec![first.clone(), second]);
        crate::sync::state::save_upload_log(
            &cache_root,
            &BTreeSet::from([(
                "omp".to_string(),
                omp_v220_key("", "", "test-model", &first.entry.usage),
            )]),
        )
        .expect("save upload log");
        let transport = FakeTransport::new(Vec::new());

        run_upload_once_with_progress(
            &cache_root,
            &enabled_config("workstation"),
            &transport,
            |_| {},
        )
        .expect("upload");

        let uploads = transport.uploads.borrow();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].len(), 1);
        assert_eq!(uploads[0][0].dedup_key, "omp:file:/tmp/omp.jsonl:1");
        let upload_log = crate::sync::state::load_upload_log(&cache_root);
        assert!(upload_log.contains(&("omp".to_string(), "omp:file:/tmp/omp.jsonl:1".to_string())));
        assert!(
            !upload_log.contains(&("omp".to_string(), "omp:file:/tmp/omp.jsonl:0".to_string()))
        );
    }

    #[test]
    fn upload_keeps_supported_vendors_when_omp_is_rejected_by_older_server() {
        let cache_root = unique_temp_dir("upload-older-server");
        populate_vendor_cache(&cache_root, "claude", "claude-a");
        populate_vendor_cache_with_record(
            &cache_root,
            "omp",
            usage_record_with_costs("omp-a", "2026-05-18T12:00:00Z"),
        );
        let transport = RejectOmpTransport {
            uploads: RefCell::new(Vec::new()),
        };

        run_upload_once_with_progress(
            &cache_root,
            &enabled_config("workstation"),
            &transport,
            |_| {},
        )
        .expect("upload should keep supported vendors");

        let uploads = transport.uploads.borrow();
        assert_eq!(uploads.len(), 2);
        assert_eq!(uploads[0][0].vendor, "claude");
        assert_eq!(uploads[1][0].vendor, "omp");
        let upload_log = crate::sync::state::load_upload_log(&cache_root);
        assert!(upload_log.contains(&("claude".to_string(), "claude-a".to_string())));
        assert!(!upload_log.contains(&("omp".to_string(), "omp-a".to_string())));
    }

    #[test]
    fn upload_progress_reports_planned_and_finished_batches() {
        let cache_root = unique_temp_dir("upload-progress");
        populate_vendor_cache_with_count(&cache_root, "claude", BATCH_SIZE + 1);
        let transport = FakeTransport::new(Vec::new());
        let mut events = Vec::new();

        run_upload_once_with_progress(
            &cache_root,
            &enabled_config("workstation"),
            &transport,
            |event| events.push(event.clone()),
        )
        .expect("upload");

        assert_eq!(
            events,
            vec![
                SyncProgress::UploadPlanned {
                    total_records: BATCH_SIZE + 1,
                    total_batches: 2,
                    skipped_records: 0,
                },
                SyncProgress::UploadBatchFinished {
                    batch_index: 1,
                    total_batches: 2,
                    uploaded_records: BATCH_SIZE,
                    total_records: BATCH_SIZE + 1,
                    accepted: BATCH_SIZE,
                    ignored: 0,
                },
                SyncProgress::UploadBatchFinished {
                    batch_index: 2,
                    total_batches: 2,
                    uploaded_records: BATCH_SIZE + 1,
                    total_records: BATCH_SIZE + 1,
                    accepted: BATCH_SIZE + 1,
                    ignored: 0,
                },
                SyncProgress::UploadFinished {
                    uploaded_records: BATCH_SIZE + 1,
                    total_records: BATCH_SIZE + 1,
                    accepted: BATCH_SIZE + 1,
                    ignored: 0,
                },
            ]
        );
    }

    #[test]
    fn pull_progress_reports_pages_and_record_totals() {
        let cache_root = unique_temp_dir("pull-progress");
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
                fast_tier: 1,
                input_tokens: 11,
                output_tokens: 12,
                cache_read_input_tokens: 13,
                cache_creation_input_tokens: 14,
                reasoning_output_tokens: 15,
                cost_input: None,
                cost_output: None,
                cost_cache_read: None,
                cost_cache_creation: None,
                project_path_sha256: None,
            },
        };
        let transport = FakeTransport::new(vec![
            PullResponse {
                records: vec![pulled],
                max_seq: 7,
                truncated: true,
            },
            PullResponse {
                records: Vec::new(),
                max_seq: 7,
                truncated: false,
            },
        ]);
        let mut events = Vec::new();

        run_pull_once_with_progress(
            &cache_root,
            &enabled_config("workstation"),
            &transport,
            |event| events.push(event.clone()),
        )
        .expect("pull");

        assert_eq!(
            events,
            vec![
                SyncProgress::PullPageFinished {
                    page_index: 1,
                    page_records: 1,
                    pulled_records: 1,
                    max_seq: 7,
                    truncated: true,
                },
                SyncProgress::PullPageFinished {
                    page_index: 2,
                    page_records: 0,
                    pulled_records: 1,
                    max_seq: 7,
                    truncated: false,
                },
                SyncProgress::PullFinished {
                    pages: 2,
                    pulled_records: 1,
                    max_seq: 7,
                },
            ]
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
                fast_tier: 1,
                input_tokens: 11,
                output_tokens: 12,
                cache_read_input_tokens: 13,
                cache_creation_input_tokens: 14,
                reasoning_output_tokens: 15,
                cost_input: None,
                cost_output: None,
                cost_cache_read: None,
                cost_cache_creation: None,
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
        assert_eq!(remote[0].entry.fast_tier, 1);
        assert_eq!(
            crate::sync::state::load_sync_state(&cache_root).last_seen_seq,
            7
        );
    }

    #[test]
    fn sync_pull_preserves_remote_embedded_costs() {
        let cache_root = unique_temp_dir("pull-costs");
        let pulled = SequencedWireRecord {
            seq: 7,
            uploaded_at: "2026-05-18T12:10:00Z".to_string(),
            record: WireRecord {
                schema_version: SCHEMA_VERSION,
                host_id: "laptop".to_string(),
                vendor: "omp".to_string(),
                dedup_key: "remote-omp-a".to_string(),
                timestamp: "2026-05-18T12:00:00Z".to_string(),
                session_start_time: "2026-05-18T12:00:00Z".to_string(),
                session_end_time: "2026-05-18T12:05:00Z".to_string(),
                model: "remote-model".to_string(),
                effort: Some("openai-codex".to_string()),
                fast_tier: -1,
                input_tokens: 11,
                output_tokens: 12,
                cache_read_input_tokens: 13,
                cache_creation_input_tokens: 14,
                reasoning_output_tokens: 15,
                cost_input: Some(0.11),
                cost_output: Some(0.12),
                cost_cache_read: Some(0.13),
                cost_cache_creation: Some(0.14),
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
        let costs = remote[0].entry.costs.expect("remote costs");
        assert_eq!(costs.input, 0.11);
        assert_eq!(costs.output, 0.12);
        assert_eq!(costs.cache_read, 0.13);
        assert_eq!(costs.cache_creation, 0.14);
    }
}
