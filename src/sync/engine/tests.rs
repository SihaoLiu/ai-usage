use super::*;
use crate::data::{SourceUsageRecord, TokenUsage, UsageEntry};
use ai_usage_proto::{
    PullResponse, RecordKey, SCHEMA_VERSION, SequencedWireRecord, UploadResponse, WireRecord,
};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// (after_seq, exclude_host, limit, supported_vendors) seen by pull().
type PullRequestLog = (u64, String, usize, Vec<String>);

struct FakeTransport {
    uploads: RefCell<Vec<Vec<WireRecord>>>,
    pulls: RefCell<Vec<PullResponse>>,
    pull_requests: RefCell<Vec<PullRequestLog>>,
    integrity_submissions: RefCell<Vec<IntegrityReport>>,
    integrity_reports: RefCell<Vec<IntegrityReport>>,
    /// When set, pull() rejects any request whose vendor list contains
    /// this vendor, imitating an older server's "invalid vendor" 400.
    reject_pull_vendor: Option<&'static str>,
    /// When set, upload() rejects any batch containing this vendor,
    /// imitating an older server that predates it.
    reject_upload_vendor: Option<&'static str>,
}

struct RejectVendorTransport {
    rejected: &'static str,
    uploads: RefCell<Vec<Vec<WireRecord>>>,
}

struct DiffSnapshotTransport {
    uploads: RefCell<Vec<Vec<WireRecord>>>,
    snapshot_diffs: RefCell<Vec<SnapshotDiffRequest>>,
    snapshot_record_batches: RefCell<Vec<SnapshotRecordBatch>>,
    snapshot_finalizations: RefCell<Vec<SnapshotFinalizeRequest>>,
    needed: Vec<RecordKey>,
}

impl FakeTransport {
    fn new(pulls: Vec<PullResponse>) -> Self {
        Self {
            uploads: RefCell::new(Vec::new()),
            pulls: RefCell::new(pulls),
            pull_requests: RefCell::new(Vec::new()),
            integrity_submissions: RefCell::new(Vec::new()),
            integrity_reports: RefCell::new(Vec::new()),
            reject_pull_vendor: None,
            reject_upload_vendor: None,
        }
    }

    fn new_rejecting_pull_vendor(pulls: Vec<PullResponse>, vendor: &'static str) -> Self {
        Self {
            reject_pull_vendor: Some(vendor),
            ..Self::new(pulls)
        }
    }

    fn new_with_integrity(pulls: Vec<PullResponse>, reports: Vec<IntegrityReport>) -> Self {
        Self {
            integrity_reports: RefCell::new(reports),
            ..Self::new(pulls)
        }
    }
}

impl SyncTransport for FakeTransport {
    fn upload(&self, records: &[WireRecord]) -> Result<UploadResponse, SyncError> {
        self.uploads.borrow_mut().push(records.to_vec());
        if let Some(rejected) = self.reject_upload_vendor
            && records.iter().any(|record| record.vendor == rejected)
        {
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
        after_seq: u64,
        exclude_host: &str,
        limit: usize,
        supported_vendors: &[&str],
    ) -> Result<PullResponse, SyncError> {
        self.pull_requests.borrow_mut().push((
            after_seq,
            exclude_host.to_string(),
            limit,
            supported_vendors
                .iter()
                .map(|vendor| (*vendor).to_string())
                .collect(),
        ));
        if let Some(rejected) = self.reject_pull_vendor
            && supported_vendors.contains(&rejected)
        {
            return Err(SyncError::new(
                "http status: 400: supported_vendors contains invalid vendor",
            ));
        }
        Ok(self.pulls.borrow_mut().remove(0))
    }

    fn submit_integrity_report(
        &self,
        report: &IntegrityReport,
    ) -> Result<IntegritySubmitResponse, SyncError> {
        self.integrity_submissions.borrow_mut().push(report.clone());
        Ok(IntegritySubmitResponse { accepted: true })
    }

    fn integrity_reports(&self) -> Result<IntegrityReportList, SyncError> {
        Ok(IntegrityReportList {
            reports: self.integrity_reports.borrow().clone(),
        })
    }
}

impl SyncTransport for RejectVendorTransport {
    fn upload(&self, records: &[WireRecord]) -> Result<UploadResponse, SyncError> {
        self.uploads.borrow_mut().push(records.to_vec());
        if records.iter().any(|record| record.vendor == self.rejected) {
            return Err(SyncError::new("http status: 400: invalid vendor"));
        }
        Ok(UploadResponse {
            accepted: records.len(),
            ignored: 0,
            max_seq: 0,
        })
    }

    fn snapshot_diff(
        &self,
        request: &SnapshotDiffRequest,
    ) -> Result<SnapshotDiffResponse, SyncError> {
        // An older server that has snapshot endpoints but predates the
        // rejected vendor: it 400s the diff instead of 404ing the route.
        if request
            .records
            .iter()
            .any(|record| record.vendor == self.rejected)
        {
            return Err(SyncError::new("http status: 400: invalid vendor"));
        }
        Err(SyncError::new("snapshot unsupported"))
    }

    fn pull(
        &self,
        _after_seq: u64,
        _exclude_host: &str,
        _limit: usize,
        _supported_vendors: &[&str],
    ) -> Result<PullResponse, SyncError> {
        Ok(PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        })
    }
}

impl DiffSnapshotTransport {
    fn new(needed: Vec<RecordKey>) -> Self {
        Self {
            uploads: RefCell::new(Vec::new()),
            snapshot_diffs: RefCell::new(Vec::new()),
            snapshot_record_batches: RefCell::new(Vec::new()),
            snapshot_finalizations: RefCell::new(Vec::new()),
            needed,
        }
    }
}

impl SyncTransport for DiffSnapshotTransport {
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
        _supported_vendors: &[&str],
    ) -> Result<PullResponse, SyncError> {
        Ok(PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        })
    }

    fn snapshot_diff(
        &self,
        request: &SnapshotDiffRequest,
    ) -> Result<SnapshotDiffResponse, SyncError> {
        self.snapshot_diffs.borrow_mut().push(request.clone());
        let needed = self.needed.clone();
        Ok(SnapshotDiffResponse {
            matched: request.records.len().saturating_sub(needed.len()),
            missing_or_changed: needed.len(),
            needed,
            max_seq: 0,
        })
    }

    fn snapshot_records(&self, batch: &SnapshotRecordBatch) -> Result<UploadResponse, SyncError> {
        self.snapshot_record_batches
            .borrow_mut()
            .push(batch.clone());
        Ok(UploadResponse {
            accepted: batch.records.len(),
            ignored: 0,
            max_seq: 0,
        })
    }

    fn snapshot_finalize(
        &self,
        request: &SnapshotFinalizeRequest,
    ) -> Result<SnapshotFinalizeResponse, SyncError> {
        self.snapshot_finalizations
            .borrow_mut()
            .push(request.clone());
        Ok(SnapshotFinalizeResponse {
            deleted: 0,
            max_seq: 0,
        })
    }
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ai-usage-engine-test-{name}-{stamp}"));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
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
    crate::data::cache::load_or_update_vendor_cache(cache_root, vendor, vec![source], 1, |_| {
        vec![usage_record(key, "2026-05-18T12:00:00Z", 10)]
    });
}

fn populate_vendor_cache_with_record(cache_root: &Path, vendor: &str, record: SourceUsageRecord) {
    populate_vendor_cache_with_records(cache_root, vendor, vec![record]);
}

fn populate_vendor_cache_with_records(
    cache_root: &Path,
    vendor: &str,
    records: Vec<SourceUsageRecord>,
) {
    let source = cache_root.join(format!("{vendor}.jsonl"));
    std::fs::write(&source, "source").expect("write source");
    crate::data::cache::load_or_update_vendor_cache(cache_root, vendor, vec![source], -1, |_| {
        records.clone()
    });
}

fn populate_vendor_cache_with_count(cache_root: &Path, vendor: &str, count: usize) {
    let source = cache_root.join(format!("{vendor}.jsonl"));
    std::fs::write(&source, "source").expect("write source");
    crate::data::cache::load_or_update_vendor_cache(cache_root, vendor, vec![source], 0, |_| {
        (0..count)
            .map(|idx| usage_record(&format!("dedup-{idx}"), "2026-05-18T12:00:00Z", 10))
            .collect()
    });
}

fn remote_usage_record(
    host_id: &str,
    vendor: &str,
    dedup_key: &str,
    timestamp: &str,
    input_tokens: i64,
) -> crate::data::cache::RemoteUsageRecord {
    crate::data::cache::RemoteUsageRecord {
        vendor: vendor.to_string(),
        dedup_key: dedup_key.to_string(),
        entry: UsageEntry {
            host_id: Some(host_id.to_string()),
            session_id: None,
            timestamp: timestamp.to_string(),
            parsed_timestamp: crate::time_utils::parse_timestamp(timestamp),
            session_start_time: timestamp.to_string(),
            session_end_time: timestamp.to_string(),
            model: "remote-model".to_string(),
            effort: None,
            fast_tier: 1,
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

fn sequenced_remote_record(
    seq: u64,
    record: crate::data::cache::RemoteUsageRecord,
) -> SequencedWireRecord {
    SequencedWireRecord {
        seq,
        uploaded_at: "2026-05-18T12:10:00Z".to_string(),
        record: WireRecord {
            schema_version: SCHEMA_VERSION,
            host_id: record.entry.host_id.expect("remote host id"),
            vendor: record.vendor,
            dedup_key: record.dedup_key,
            timestamp: record.entry.timestamp,
            session_start_time: record.entry.session_start_time,
            session_end_time: record.entry.session_end_time,
            model: record.entry.model,
            effort: record.entry.effort,
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
            project_path_sha256: None,
        },
    }
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

#[test]
fn sync_cycle_uploads_unseen_cached_records_and_updates_upload_log() {
    let cache_root = unique_temp_dir("upload");
    populate_vendor_cache(&cache_root, "claude", "dedup-a");
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport).expect("sync cycle");

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
fn sync_upload_refreshes_logged_omp_stable_metadata_once() {
    let cache_root = unique_temp_dir("upload-omp-refresh");
    let mut record = usage_record(
        "omp:message:msg-a:response:resp-a",
        "2026-05-18T12:00:00Z",
        10,
    );
    record.entry.model = "claude-sonnet-4-5-20250929".to_string();
    record.entry.effort = Some("anthropic".to_string());
    record.entry.costs = Some(UsageCost {
        input: 0.01,
        output: 0.02,
        cache_read: 0.03,
        cache_creation: 0.04,
    });
    populate_vendor_cache_with_record(&cache_root, "omp", record.clone());
    crate::sync::state::save_upload_log(
        &cache_root,
        &BTreeSet::from([("omp".to_string(), record.dedup_key.clone())]),
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
    assert_eq!(uploads[0][0].dedup_key, "omp:message:msg-a:response:resp-a");
    assert_eq!(uploads[0][0].effort.as_deref(), Some("anthropic"));
    assert_eq!(uploads[0][0].cost_input, Some(0.01));
    drop(uploads);

    let second_transport = FakeTransport::new(Vec::new());
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &second_transport,
        |_| {},
    )
    .expect("second upload");

    assert!(second_transport.uploads.borrow().is_empty());
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
fn sync_upload_keeps_omp_file_records_for_server_side_aliasing() {
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
    assert_eq!(uploads[0].len(), 2);
    assert_eq!(uploads[0][0].dedup_key, "omp:file:/tmp/omp.jsonl:0");
    assert_eq!(uploads[0][1].dedup_key, "omp:file:/tmp/omp.jsonl:1");
    let upload_log = crate::sync::state::load_upload_log(&cache_root);
    assert!(upload_log.contains(&("omp".to_string(), "omp:file:/tmp/omp.jsonl:0".to_string())));
    assert!(upload_log.contains(&("omp".to_string(), "omp:file:/tmp/omp.jsonl:1".to_string())));
}

#[test]
fn snapshot_upload_uses_server_diff_and_only_sends_needed_records() {
    let cache_root = unique_temp_dir("snapshot-diff-upload");
    populate_vendor_cache(&cache_root, "claude", "claude-a");
    populate_vendor_cache(&cache_root, "codex", "codex-a");
    let first = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "codex".to_string(),
        dedup_key: "codex-a".to_string(),
    }]);

    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &first, |_| {})
        .expect("first upload");

    assert!(first.uploads.borrow().is_empty());
    assert_eq!(first.snapshot_diffs.borrow().len(), 1);
    assert_eq!(first.snapshot_diffs.borrow()[0].records.len(), 2);
    let record_batches = first.snapshot_record_batches.borrow();
    assert_eq!(record_batches.len(), 1);
    assert_eq!(record_batches[0].records.len(), 1);
    assert_eq!(record_batches[0].records[0].vendor, "codex");
    assert_eq!(record_batches[0].records[0].dedup_key, "codex-a");
    drop(record_batches);
    assert_eq!(first.snapshot_finalizations.borrow().len(), 1);

    let second = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &second, |_| {})
        .expect("second upload");

    assert!(second.uploads.borrow().is_empty());
    assert!(second.snapshot_diffs.borrow().is_empty());
    assert!(second.snapshot_record_batches.borrow().is_empty());
    assert!(second.snapshot_finalizations.borrow().is_empty());
}

#[test]
fn snapshot_upload_assigns_stable_keys_to_empty_dedup_records() {
    let cache_root = unique_temp_dir("snapshot-empty-dedup");
    populate_vendor_cache_with_records(
        &cache_root,
        "claude",
        vec![
            usage_record("", "2026-05-18T12:00:00Z", 10),
            usage_record("", "2026-05-18T12:01:00Z", 20),
        ],
    );
    let transport = DiffSnapshotTransport::new(Vec::new());

    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("upload");

    let diffs = transport.snapshot_diffs.borrow();
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].records.len(), 2);
    assert_eq!(diffs[0].records[0].vendor, "claude");
    assert_eq!(diffs[0].records[1].vendor, "claude");
    assert!(diffs[0].records[0].dedup_key.starts_with("fallback:v1:"));
    assert!(diffs[0].records[1].dedup_key.starts_with("fallback:v1:"));
    assert_ne!(diffs[0].records[0].dedup_key, diffs[0].records[1].dedup_key);
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
    let transport = RejectVendorTransport {
        rejected: "omp",
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
fn upload_skips_any_vendor_batch_an_older_server_rejects() {
    let cache_root = unique_temp_dir("upload-older-server-kimi");
    populate_vendor_cache(&cache_root, "claude", "claude-a");
    populate_vendor_cache(&cache_root, "kimi", "kimi-a");
    populate_vendor_cache(&cache_root, "omp", "omp-a");
    let transport = RejectVendorTransport {
        rejected: "kimi",
        uploads: RefCell::new(Vec::new()),
    };
    let mut events = Vec::new();

    let outcome = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("upload should skip the rejected vendor and continue");

    // The kimi batch was attempted and rejected, but claude before it and
    // omp after it both uploaded.
    let uploads = transport.uploads.borrow();
    let vendors: Vec<&str> = uploads
        .iter()
        .map(|batch| batch[0].vendor.as_str())
        .collect();
    assert_eq!(vendors, ["claude", "kimi", "omp"]);
    let upload_log = crate::sync::state::load_upload_log(&cache_root);
    assert!(upload_log.contains(&("claude".to_string(), "claude-a".to_string())));
    assert!(upload_log.contains(&("omp".to_string(), "omp-a".to_string())));
    assert!(!upload_log.contains(&("kimi".to_string(), "kimi-a".to_string())));

    // The held-back vendor is reported and the progress totals reconcile
    // once its batches drop out of the plan.
    assert_eq!(outcome.held_back_vendors, ["kimi".to_string()]);
    assert!(events.iter().any(|event| matches!(
        event,
        SyncProgress::UploadVendorHeldBack { vendor, records }
            if vendor == "kimi" && *records == 1
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        SyncProgress::UploadFinished {
            uploaded_records: 2,
            total_records: 2,
            ..
        }
    )));
}

#[test]
fn upload_fails_loudly_when_a_core_vendor_is_rejected() {
    let cache_root = unique_temp_dir("upload-core-rejected");
    populate_vendor_cache(&cache_root, "claude", "claude-a");
    let transport = RejectVendorTransport {
        rejected: "claude",
        uploads: RefCell::new(Vec::new()),
    };

    let result = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    );

    // No legitimate older server rejects a core vendor: this signals a
    // broken server or proxy and must not be silently held back.
    assert!(result.is_err());
    let upload_log = crate::sync::state::load_upload_log(&cache_root);
    assert!(!upload_log.contains(&("claude".to_string(), "claude-a".to_string())));
}

#[test]
fn snapshot_upload_falls_back_to_batch_when_server_rejects_vendor() {
    let cache_root = unique_temp_dir("snapshot-older-server-kimi");
    populate_vendor_cache(&cache_root, "claude", "claude-a");
    populate_vendor_cache(&cache_root, "kimi", "kimi-a");
    let transport = RejectVendorTransport {
        rejected: "kimi",
        uploads: RefCell::new(Vec::new()),
    };

    let outcome = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("snapshot rejection should fall back to batch upload");

    // The snapshot diff 400ed on the kimi fingerprints; the batch path
    // still uploaded claude and held kimi back.
    let uploads = transport.uploads.borrow();
    let vendors: Vec<&str> = uploads
        .iter()
        .map(|batch| batch[0].vendor.as_str())
        .collect();
    assert_eq!(vendors, ["claude", "kimi"]);
    assert_eq!(outcome.held_back_vendors, ["kimi".to_string()]);
    let upload_log = crate::sync::state::load_upload_log(&cache_root);
    assert!(upload_log.contains(&("claude".to_string(), "claude-a".to_string())));
    assert!(!upload_log.contains(&("kimi".to_string(), "kimi-a".to_string())));
}

#[test]
fn sync_cycle_skips_integrity_when_upload_holds_back_a_vendor() {
    let cache_root = unique_temp_dir("cycle-held-back");
    populate_vendor_cache(&cache_root, "claude", "claude-a");
    populate_vendor_cache(&cache_root, "kimi", "kimi-a");
    let transport = FakeTransport {
        reject_upload_vendor: Some("kimi"),
        ..FakeTransport::new(vec![PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        }])
    };

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("degraded cycle should still succeed");

    // Pull ran, but no integrity report was submitted: the digest would
    // have counted the held-back kimi records peers cannot pull.
    assert_eq!(transport.pull_requests.borrow().len(), 1);
    assert!(transport.integrity_submissions.borrow().is_empty());
}

#[test]
fn pull_and_integrity_skips_integrity_on_degraded_pull() {
    let cache_root = unique_temp_dir("degraded-pull-integrity");
    let transport = FakeTransport::new_rejecting_pull_vendor(
        vec![PullResponse {
            records: Vec::new(),
            max_seq: 0,
            truncated: false,
        }],
        "kimi",
    );

    run_pull_and_integrity_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("degraded pull should succeed without integrity");

    assert_eq!(transport.pull_requests.borrow().len(), 2);
    assert!(transport.integrity_submissions.borrow().is_empty());
}

#[test]
fn pull_downgrade_after_rollback_keeps_cached_records_and_cursor() {
    let cache_root = unique_temp_dir("pull-rollback");
    // A kimi record already pulled while the server was new.
    let cached_kimi =
        remote_usage_record("laptop", "kimi", "remote-kimi", "2026-05-18T12:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![cached_kimi])
        .expect("seed remote cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS),
            last_successful_sync: None,
            last_error: None,
        },
    )
    .expect("save migrated state");
    let transport = FakeTransport::new_rejecting_pull_vendor(
        vec![PullResponse {
            records: Vec::new(),
            max_seq: 12,
            truncated: false,
        }],
        "kimi",
    );

    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("rollback pull should downgrade without wiping");

    // The fallback pull stayed incremental (cursor kept) and the cached
    // kimi record survived the downgrade.
    let requests = transport.pull_requests.borrow();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].0, 10);
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "remote-kimi");
    // The reduced fingerprint is adopted so a later server upgrade
    // triggers the full backfill refetch.
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state.pull_vendors,
        pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS)
    );
}

#[test]
fn pull_preserves_cache_and_downgrades_when_older_server_rejects_new_vendor() {
    let cache_root = unique_temp_dir("pull-older-server");
    let existing = remote_usage_record("laptop", "claude", "remote-a", "2026-05-18T12:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![existing])
        .expect("seed remote cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS),
            last_successful_sync: None,
            last_error: None,
        },
    )
    .expect("save pre-upgrade state");
    let new_remote =
        remote_usage_record("laptop", "claude", "remote-b", "2026-05-18T13:00:00Z", 20);
    let transport = FakeTransport::new_rejecting_pull_vendor(
        vec![PullResponse {
            records: vec![sequenced_remote_record(11, new_remote)],
            max_seq: 11,
            truncated: false,
        }],
        "kimi",
    );

    let mut events = Vec::new();
    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("pull should fall back to the previous vendor set");

    // Full set offered first, then the previous set from the stored cursor.
    let requests = transport.pull_requests.borrow();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].3.contains(&"kimi".to_string()));
    assert!(!requests[1].3.contains(&"kimi".to_string()));
    assert_eq!(requests[1].0, 10);
    // The degraded pull names the vendors the server could not serve.
    assert!(events.iter().any(|event| matches!(
        event,
        SyncProgress::PullVendorsUnavailable { vendors }
            if vendors == &vec!["kimi".to_string()]
    )));
    // The pre-existing cached record survived and the new one merged in.
    let mut remote_keys: Vec<String> = crate::data::cache::load_remote_entries(&cache_root, None)
        .iter()
        .map(|record| record.dedup_key.clone())
        .collect();
    remote_keys.sort();
    assert_eq!(remote_keys, ["remote-a", "remote-b"]);
    // The fingerprint stays on the previous set so the migration retries
    // once the server is upgraded.
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state.pull_vendors,
        pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS)
    );
}

#[test]
fn pull_migrates_vendor_fingerprint_once_server_accepts_new_vendor() {
    let cache_root = unique_temp_dir("pull-migrate");
    let stale = remote_usage_record("laptop", "claude", "stale-a", "2026-05-18T12:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![stale])
        .expect("seed remote cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS),
            last_successful_sync: None,
            last_error: None,
        },
    )
    .expect("save pre-upgrade state");
    let refetched =
        remote_usage_record("laptop", "kimi", "remote-kimi", "2026-05-18T13:00:00Z", 20);
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![sequenced_remote_record(12, refetched)],
        max_seq: 12,
        truncated: false,
    }]);

    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("pull should migrate to the full vendor set");

    // Migration refetches from the beginning and replaces the cache.
    let requests = transport.pull_requests.borrow();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0, 0);
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "remote-kimi");
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state.pull_vendors,
        pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS)
    );
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
fn sync_cycle_submits_local_integrity_report_and_verifies_remote_reports() {
    let cache_root = unique_temp_dir("integrity-cycle");
    let owner_cache = unique_temp_dir("integrity-owner");
    populate_vendor_cache(&cache_root, "claude", "local-a");
    crate::data::cache::merge_remote_records(
        &owner_cache,
        "laptop",
        vec![remote_usage_record(
            "laptop",
            "claude",
            "remote-a",
            "2000-01-01T00:00:00Z",
            10,
        )],
    )
    .expect("seed owner cache");
    let range_end = crate::sync::integrity::integrity_range_end_utc(Utc::now());
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        range_end,
        Utc::now(),
    )
    .expect("owner report");
    let pulled = SequencedWireRecord {
        seq: 1,
        uploaded_at: "2000-01-01T00:00:01Z".to_string(),
        record: WireRecord {
            schema_version: SCHEMA_VERSION,
            host_id: "laptop".to_string(),
            vendor: "claude".to_string(),
            dedup_key: "remote-a".to_string(),
            timestamp: "2000-01-01T00:00:00Z".to_string(),
            session_start_time: "2000-01-01T00:00:00Z".to_string(),
            session_end_time: "2000-01-01T00:00:00Z".to_string(),
            model: "remote-model".to_string(),
            effort: None,
            fast_tier: 1,
            input_tokens: 10,
            output_tokens: 2,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 4,
            reasoning_output_tokens: 5,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
            project_path_sha256: None,
        },
    };
    let transport = FakeTransport::new_with_integrity(
        vec![PullResponse {
            records: vec![pulled],
            max_seq: 1,
            truncated: false,
        }],
        vec![owner_report],
    );
    let mut events = Vec::new();

    run_sync_cycle_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("sync cycle");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            SyncProgress::IntegrityCheckFinished {
                verification: crate::sync::integrity::IntegrityVerification::Checked {
                    checked_hosts: 1
                }
            }
        )
    }));
}

#[test]
fn integrity_failure_clears_remote_cache_and_rechecks_after_repull() {
    let cache_root = unique_temp_dir("integrity-repair");
    let owner_cache = unique_temp_dir("integrity-repair-owner");
    let correct = remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
    let stale = remote_usage_record("laptop", "claude", "stale-a", "2000-01-01T00:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![stale])
        .expect("seed stale cache");
    crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![correct.clone()])
        .expect("seed owner cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 0,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS),
            last_successful_sync: None,
            last_error: None,
        },
    )
    .expect("save current cursor");
    let range_end = crate::sync::integrity::integrity_range_end_utc(Utc::now());
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        range_end,
        Utc::now(),
    )
    .expect("owner report");
    let pulled = sequenced_remote_record(1, correct);
    let transport = FakeTransport::new_with_integrity(
        vec![
            PullResponse {
                records: vec![pulled.clone()],
                max_seq: 1,
                truncated: false,
            },
            PullResponse {
                records: vec![pulled],
                max_seq: 1,
                truncated: false,
            },
        ],
        vec![owner_report],
    );
    let mut events = Vec::new();

    run_sync_cycle_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("sync cycle");

    assert_eq!(transport.pull_requests.borrow().len(), 2);
    assert!(matches!(transport.pull_requests.borrow()[1].0, 0));
    let integrity_events = events
        .iter()
        .filter_map(|event| match event {
            SyncProgress::IntegrityCheckFinished { verification } => Some(verification),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        integrity_events[0],
        crate::sync::integrity::IntegrityVerification::Failed { .. }
    ));
    assert!(matches!(
        integrity_events[1],
        crate::sync::integrity::IntegrityVerification::Checked { checked_hosts: 1 }
    ));
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "remote-a");
}

#[test]
fn pull_integrity_repair_clears_stale_cache_after_incremental_pull_failure() {
    let cache_root = unique_temp_dir("pull-integrity-repair");
    let owner_cache = unique_temp_dir("pull-integrity-repair-owner");
    let correct = remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
    let stale = remote_usage_record("laptop", "claude", "stale-a", "2000-01-01T00:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![stale])
        .expect("seed stale cache");
    crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![correct.clone()])
        .expect("seed owner cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS),
            last_successful_sync: None,
            last_error: None,
        },
    )
    .expect("save current cursor");
    let range_end = crate::sync::integrity::integrity_range_end_utc(Utc::now());
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        range_end,
        Utc::now(),
    )
    .expect("owner report");
    let pulled = sequenced_remote_record(7, correct);
    let transport = FakeTransport::new_with_integrity(
        vec![
            PullResponse {
                records: Vec::new(),
                max_seq: 10,
                truncated: false,
            },
            PullResponse {
                records: vec![pulled],
                max_seq: 7,
                truncated: false,
            },
        ],
        vec![owner_report],
    );
    let mut events = Vec::new();

    run_pull_and_integrity_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("pull integrity");

    let pull_seqs: Vec<u64> = transport
        .pull_requests
        .borrow()
        .iter()
        .map(|request| request.0)
        .collect();
    assert_eq!(pull_seqs, vec![10, 0]);
    let integrity_events = events
        .iter()
        .filter_map(|event| match event {
            SyncProgress::IntegrityCheckFinished { verification } => Some(verification),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        integrity_events[0],
        crate::sync::integrity::IntegrityVerification::Failed { .. }
    ));
    assert!(matches!(
        integrity_events[1],
        crate::sync::integrity::IntegrityVerification::Checked { checked_hosts: 1 }
    ));
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "remote-a");
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
    run_sync_cycle(&cache_root, &enabled_config("workstation"), &first).expect("first sync cycle");
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

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport).expect("sync cycle");

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
fn sync_pull_resets_cursor_when_pull_vendor_set_changes() {
    let cache_root = unique_temp_dir("pull-vendor-set");
    crate::data::cache::merge_remote_records(
        &cache_root,
        "laptop",
        vec![remote_usage_record(
            "laptop",
            "codex",
            "stale-a",
            "2026-05-18T12:00:00Z",
            10,
        )],
    )
    .expect("seed stale remote cache");
    std::fs::write(
        cache_root.join("sync_state.json"),
        r#"{
  "schema_version": 1,
  "last_seen_seq": 2,
  "last_successful_sync": null,
  "last_error": null
}"#,
    )
    .expect("write old sync state");
    let omp_record = WireRecord {
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
        cost_input: None,
        cost_output: None,
        cost_cache_read: None,
        cost_cache_creation: None,
        project_path_sha256: None,
    };
    let mut claude_record = omp_record.clone();
    claude_record.vendor = "claude".to_string();
    claude_record.dedup_key = "remote-claude-a".to_string();
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![
            SequencedWireRecord {
                seq: 1,
                record: omp_record,
                uploaded_at: "2026-05-18T12:10:00Z".to_string(),
            },
            SequencedWireRecord {
                seq: 2,
                record: claude_record,
                uploaded_at: "2026-05-18T12:11:00Z".to_string(),
            },
        ],
        max_seq: 2,
        truncated: false,
    }]);

    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("pull");

    assert_eq!(transport.pull_requests.borrow()[0].0, 0);
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 2);
    assert!(remote.iter().all(|record| record.dedup_key != "stale-a"));
    assert_eq!(
        crate::sync::state::load_sync_state(&cache_root).last_seen_seq,
        2
    );
}

#[test]
fn sync_pull_resets_legacy_exclude_self_cursor_and_requests_all_hosts() {
    let cache_root = unique_temp_dir("pull-include-self");
    std::fs::write(
        cache_root.join("sync_state.json"),
        r#"{
  "schema_version": 1,
  "last_seen_seq": 99,
  "pull_vendors": ["claude", "codex", "gemini", "omp"],
  "last_successful_sync": null,
  "last_error": null
}"#,
    )
    .expect("write old sync state");
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 123,
        truncated: false,
    }]);

    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("pull");

    let (after_seq, exclude_host, limit, _) = transport.pull_requests.borrow()[0].clone();
    assert_eq!(
        (after_seq, exclude_host, limit),
        (0, String::new(), PULL_LIMIT)
    );
    assert_eq!(
        crate::sync::state::load_sync_state(&cache_root).last_seen_seq,
        123
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

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport).expect("sync cycle");

    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    let costs = remote[0].entry.costs.expect("remote costs");
    assert_eq!(costs.input, 0.11);
    assert_eq!(costs.output, 0.12);
    assert_eq!(costs.cache_read, 0.13);
    assert_eq!(costs.cache_creation, 0.14);
}
