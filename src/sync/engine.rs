use crate::data::cache::{self, CachedUsageRecord, RemoteUsageRecord};
use crate::data::{TokenUsage, UsageCost, UsageEntry};
use crate::sync::config::EnabledSyncConfig;
use crate::sync::keys::assign_sync_dedup_keys;
use crate::sync::state;
use crate::time_utils::parse_timestamp;
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use ai_usage_proto::{
    IntegrityReport, IntegrityReportList, IntegritySubmitResponse, PullResponse, RecordFingerprint,
    SnapshotDiffRequest, SnapshotDiffResponse, SnapshotFinalizeRequest, SnapshotFinalizeResponse,
    SnapshotRecordBatch, UploadResponse, WireRecord,
};

pub const SUPPORTED_PULL_VENDORS: [&str; 4] = ["claude", "codex", "gemini", "omp"];
const VENDORS: [&str; 4] = SUPPORTED_PULL_VENDORS;
const BATCH_SIZE: usize = 1000;
const PULL_LIMIT: usize = 20_000;
const SNAPSHOT_DIFF_TARGET_BYTES: usize = 900_000;
const SNAPSHOT_UPLOAD_STATE_VERSION: u32 = state::SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION;
const OMP_METADATA_REFRESH_LOG_VENDOR: &str = "omp-metadata-refresh";
const PULL_SCOPE_ALL_HOSTS_MARKER: &str = "scope:all-hosts";

#[derive(Debug, Clone)]
struct UploadCandidate {
    key: (String, String),
    refresh_key: Option<(String, String)>,
    wire: WireRecord,
}

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
    IntegrityUnsupported,
    IntegrityReportSubmitted {
        record_count: u64,
        range_end_utc: String,
    },
    IntegrityCheckFinished {
        verification: crate::sync::integrity::IntegrityVerification,
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
    fn submit_integrity_report(
        &self,
        _report: &IntegrityReport,
    ) -> Result<IntegritySubmitResponse, SyncError> {
        Err(SyncError::new("integrity unsupported"))
    }
    fn integrity_reports(&self) -> Result<IntegrityReportList, SyncError> {
        Err(SyncError::new("integrity unsupported"))
    }
    fn snapshot_diff(
        &self,
        _request: &SnapshotDiffRequest,
    ) -> Result<SnapshotDiffResponse, SyncError> {
        Err(SyncError::new("snapshot unsupported"))
    }
    fn snapshot_records(&self, _batch: &SnapshotRecordBatch) -> Result<UploadResponse, SyncError> {
        Err(SyncError::new("snapshot unsupported"))
    }
    fn snapshot_finalize(
        &self,
        _request: &SnapshotFinalizeRequest,
    ) -> Result<SnapshotFinalizeResponse, SyncError> {
        Err(SyncError::new("snapshot unsupported"))
    }
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
    run_pull_and_integrity_once_with_progress(cache_root, config, transport, on_progress)
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
    if run_snapshot_upload_once_with_progress(cache_root, config, transport, &mut on_progress)? {
        return Ok(());
    }

    let mut upload_log = state::load_upload_log(cache_root);
    let mut upload_groups = Vec::new();
    let mut skipped_records = 0;

    for vendor in VENDORS {
        let mut vendor_records = Vec::new();
        let mut vendor_refresh_records = Vec::new();
        for keyed in assign_sync_dedup_keys(cache::load_vendor_cached_records(cache_root, vendor)) {
            let record = keyed.record;
            let key = (record.vendor.clone(), keyed.dedup_key.clone());
            let logged = upload_log.contains(&key);
            if logged || uploaded_with_omp_v220_key(&record, &upload_log) {
                if logged
                    && let Some(refresh_key) = omp_metadata_refresh_log_key(&record)
                    && !upload_log.contains(&refresh_key)
                {
                    let wire = cached_record_to_wire(config, &record, &key.1)?;
                    vendor_refresh_records.push(UploadCandidate {
                        key,
                        refresh_key: Some(refresh_key),
                        wire,
                    });
                    continue;
                }
                skipped_records += 1;
                continue;
            }
            let wire = cached_record_to_wire(config, &record, &key.1)?;
            vendor_records.push(UploadCandidate {
                key,
                refresh_key: None,
                wire,
            });
        }
        if !vendor_records.is_empty() {
            upload_groups.push(vendor_records);
        }
        if !vendor_refresh_records.is_empty() {
            upload_groups.push(vendor_refresh_records);
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
            let wire_records: Vec<WireRecord> = batch
                .iter()
                .map(|candidate| candidate.wire.clone())
                .collect();
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
            for candidate in batch {
                upload_log.insert(candidate.key.clone());
                if response.accepted > 0
                    && let Some(refresh_key) = &candidate.refresh_key
                {
                    upload_log.insert(refresh_key.clone());
                }
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

fn run_snapshot_upload_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<bool, SyncError>
where
    F: FnMut(&SyncProgress),
{
    let snapshot_id = snapshot_id(&config.machine_id);
    let snapshot = collect_snapshot_records(cache_root, config)?;
    let previous = state::load_snapshot_upload_state(cache_root);
    if previous.schema_version == SNAPSHOT_UPLOAD_STATE_VERSION
        && previous.full_hash == snapshot.full_hash
    {
        on_progress(&SyncProgress::UploadPlanned {
            total_records: 0,
            total_batches: 0,
            skipped_records: snapshot.records.len(),
        });
        on_progress(&SyncProgress::UploadFinished {
            uploaded_records: 0,
            total_records: 0,
            accepted: 0,
            ignored: 0,
        });
        return Ok(true);
    }

    let key_set_changed = previous.schema_version != SNAPSHOT_UPLOAD_STATE_VERSION
        || previous.key_set_hash != snapshot.key_set_hash;
    let manifest_records = if key_set_changed {
        snapshot.records.iter().collect::<Vec<_>>()
    } else {
        snapshot
            .records
            .iter()
            .filter(|record| {
                previous
                    .record_hashes
                    .get(&record.key)
                    .is_none_or(|hash| hash != &record.record_hash)
            })
            .collect::<Vec<_>>()
    };

    let mut needed_keys = BTreeSet::new();
    let mut chunks = snapshot_fingerprint_chunks(&manifest_records);
    if chunks.is_empty() && key_set_changed {
        chunks.push(Vec::new());
    }
    for chunk in chunks {
        let request = SnapshotDiffRequest {
            host_id: config.machine_id.clone(),
            snapshot_id: snapshot_id.clone(),
            records: chunk,
        };
        match transport.snapshot_diff(&request) {
            Ok(response) => {
                for key in response.needed {
                    needed_keys.insert((key.vendor, key.dedup_key));
                }
            }
            Err(err) if is_unsupported_snapshot_error(&err) => return Ok(false),
            Err(err) => return Err(err),
        }
    }

    let needed_records = snapshot
        .records
        .iter()
        .filter(|record| needed_keys.contains(&record.key))
        .collect::<Vec<_>>();
    let skipped_records = snapshot.records.len().saturating_sub(needed_records.len());
    let total_batches = needed_records.len().div_ceil(BATCH_SIZE);
    on_progress(&SyncProgress::UploadPlanned {
        total_records: needed_records.len(),
        total_batches,
        skipped_records,
    });

    let mut uploaded_records = 0usize;
    let mut accepted = 0usize;
    let mut ignored = 0usize;
    for (batch_index, batch) in needed_records.chunks(BATCH_SIZE).enumerate() {
        let records = batch
            .iter()
            .map(|record| record.wire.clone())
            .collect::<Vec<_>>();
        let response = transport.snapshot_records(&SnapshotRecordBatch {
            host_id: config.machine_id.clone(),
            snapshot_id: snapshot_id.clone(),
            records,
        })?;
        uploaded_records += batch.len();
        accepted += response.accepted;
        ignored += response.ignored;
        on_progress(&SyncProgress::UploadBatchFinished {
            batch_index: batch_index + 1,
            total_batches,
            uploaded_records,
            total_records: needed_records.len(),
            accepted,
            ignored,
        });
    }

    if key_set_changed {
        transport.snapshot_finalize(&SnapshotFinalizeRequest {
            host_id: config.machine_id.clone(),
            snapshot_id,
        })?;
    }

    state::save_snapshot_upload_state(
        cache_root,
        &state::SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_VERSION,
            full_hash: snapshot.full_hash,
            key_set_hash: snapshot.key_set_hash,
            record_hashes: snapshot.record_hashes,
        },
    )?;
    on_progress(&SyncProgress::UploadFinished {
        uploaded_records,
        total_records: needed_records.len(),
        accepted,
        ignored,
    });
    Ok(true)
}

#[derive(Debug)]
struct SnapshotRecords {
    records: Vec<SnapshotRecord>,
    key_set_hash: String,
    full_hash: String,
    record_hashes: BTreeMap<(String, String), String>,
}

#[derive(Debug)]
struct SnapshotRecord {
    key: (String, String),
    wire: WireRecord,
    record_hash: String,
}

fn collect_snapshot_records(
    cache_root: &Path,
    config: &EnabledSyncConfig,
) -> Result<SnapshotRecords, SyncError> {
    let mut records = Vec::new();
    let mut record_hashes = BTreeMap::new();
    for vendor in VENDORS {
        for keyed in assign_sync_dedup_keys(cache::load_vendor_cached_records(cache_root, vendor)) {
            let record = keyed.record;
            let key = (record.vendor.clone(), keyed.dedup_key.clone());
            let wire = cached_record_to_wire(config, &record, &key.1)?;
            let record_hash = wire_record_hash(&wire)?;
            record_hashes.insert(key.clone(), record_hash.clone());
            records.push(SnapshotRecord {
                key,
                wire,
                record_hash,
            });
        }
    }
    records.sort_by(|left, right| left.key.cmp(&right.key));
    let key_set_hash = snapshot_key_set_hash(&records);
    let full_hash = snapshot_full_hash(&records);
    Ok(SnapshotRecords {
        records,
        key_set_hash,
        full_hash,
        record_hashes,
    })
}

fn snapshot_fingerprint_chunks(records: &[&SnapshotRecord]) -> Vec<Vec<RecordFingerprint>> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0usize;
    for record in records {
        let estimated_bytes =
            record.key.0.len() + record.key.1.len() + record.record_hash.len() + 96;
        if !current.is_empty() && current_bytes + estimated_bytes > SNAPSHOT_DIFF_TARGET_BYTES {
            chunks.push(current);
            current = Vec::new();
            current_bytes = 0;
        }
        current.push(RecordFingerprint {
            vendor: record.key.0.clone(),
            dedup_key: record.key.1.clone(),
            record_hash: record.record_hash.clone(),
        });
        current_bytes += estimated_bytes;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn snapshot_key_set_hash(records: &[SnapshotRecord]) -> String {
    let mut hasher = Sha256::new();
    for record in records {
        hasher.update((record.key.0.len() as u64).to_be_bytes());
        hasher.update(record.key.0.as_bytes());
        hasher.update((record.key.1.len() as u64).to_be_bytes());
        hasher.update(record.key.1.as_bytes());
    }
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn snapshot_full_hash(records: &[SnapshotRecord]) -> String {
    let mut hasher = Sha256::new();
    for record in records {
        hasher.update((record.key.0.len() as u64).to_be_bytes());
        hasher.update(record.key.0.as_bytes());
        hasher.update((record.key.1.len() as u64).to_be_bytes());
        hasher.update(record.key.1.as_bytes());
        hasher.update(record.record_hash.as_bytes());
    }
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn wire_record_hash(record: &WireRecord) -> Result<String, SyncError> {
    let bytes = serde_json::to_vec(record)
        .map_err(|err| SyncError::new(format!("serialize snapshot record: {err}")))?;
    Ok(sha256_hex(&bytes))
}

fn snapshot_id(machine_id: &str) -> String {
    format!("{}:{}", machine_id, Utc::now().format("%Y%m%dT%H%M%SZ"))
}

fn is_unsupported_snapshot_error(err: &SyncError) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("snapshot unsupported")
        || message.contains("http status: 404")
        || message.contains("http status: 405")
}

fn is_unsupported_vendor_error(err: &SyncError) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("invalid vendor") || message.contains("unsupported vendor")
}

fn current_pull_vendors() -> Vec<String> {
    SUPPORTED_PULL_VENDORS
        .iter()
        .map(|vendor| (*vendor).to_string())
        .collect()
}

fn current_pull_state_fingerprint() -> Vec<String> {
    let mut fingerprint = current_pull_vendors();
    fingerprint.push(PULL_SCOPE_ALL_HOSTS_MARKER.to_string());
    fingerprint
}

fn uploaded_with_omp_v220_key(
    record: &CachedUsageRecord,
    upload_log: &BTreeSet<(String, String)>,
) -> bool {
    if record.vendor != "omp" {
        return false;
    }
    if record.dedup_key.starts_with("omp:file:") {
        return false;
    }
    for legacy_key in omp_v220_key_candidates(record) {
        if upload_log.contains(&("omp".to_string(), legacy_key)) {
            return true;
        }
    }
    false
}

fn omp_metadata_refresh_log_key(record: &CachedUsageRecord) -> Option<(String, String)> {
    if record.vendor == "omp"
        && is_stable_omp_key(&record.dedup_key)
        && (record
            .entry
            .effort
            .as_deref()
            .is_some_and(|value| !value.is_empty())
            || record.entry.costs.is_some())
    {
        Some((
            OMP_METADATA_REFRESH_LOG_VENDOR.to_string(),
            record.dedup_key.clone(),
        ))
    } else {
        None
    }
}

fn is_stable_omp_key(dedup_key: &str) -> bool {
    dedup_key.starts_with("omp:message:")
        || dedup_key.starts_with("omp:response:")
        || dedup_key.starts_with("omp:file:")
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
    _config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<(), SyncError>
where
    F: FnMut(&SyncProgress),
{
    let mut sync_state = state::load_sync_state(cache_root);
    let pull_state_fingerprint = current_pull_state_fingerprint();
    if sync_state.pull_vendors != pull_state_fingerprint {
        cache::clear_remote_cache(cache_root)?;
        sync_state.last_seen_seq = 0;
        sync_state.pull_vendors = pull_state_fingerprint;
    }
    let mut page_index = 0;
    let mut pulled_records = 0;

    loop {
        let response = transport.pull(sync_state.last_seen_seq, "", PULL_LIMIT)?;
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

pub fn run_pull_and_integrity_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<(), SyncError>
where
    F: FnMut(&SyncProgress),
{
    run_pull_once_with_progress(cache_root, config, transport, &mut on_progress)?;
    run_integrity_once_with_repair(cache_root, config, transport, on_progress)
}

pub fn run_integrity_once_with_repair<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<(), SyncError>
where
    F: FnMut(&SyncProgress),
{
    let verification =
        run_integrity_once_with_progress(cache_root, config, transport, &mut on_progress)?;
    if matches!(
        verification,
        Some(crate::sync::integrity::IntegrityVerification::Failed { .. })
    ) {
        cache::clear_remote_cache(cache_root)?;
        state::clear_sync_state(cache_root)?;
        run_pull_once_with_progress(cache_root, config, transport, &mut on_progress)?;
        run_integrity_once_with_progress(cache_root, config, transport, &mut on_progress)?;
    }
    Ok(())
}

pub fn run_integrity_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<Option<crate::sync::integrity::IntegrityVerification>, SyncError>
where
    F: FnMut(&SyncProgress),
{
    let now = Utc::now();
    let local_report = crate::sync::integrity::build_local_report_at(cache_root, config, now, now)?;
    match transport.submit_integrity_report(&local_report) {
        Ok(_) => on_progress(&SyncProgress::IntegrityReportSubmitted {
            record_count: local_report.record_count,
            range_end_utc: local_report.range_end_utc.clone(),
        }),
        Err(err) if is_unsupported_integrity_error(&err) => {
            on_progress(&SyncProgress::IntegrityUnsupported);
            return Ok(None);
        }
        Err(err) => return Err(err),
    }

    let reports = match transport.integrity_reports() {
        Ok(reports) => reports,
        Err(err) if is_unsupported_integrity_error(&err) => {
            on_progress(&SyncProgress::IntegrityUnsupported);
            return Ok(None);
        }
        Err(err) => return Err(err),
    };
    let verification = crate::sync::integrity::verify_remote_reports_at(
        cache_root,
        &config.machine_id,
        &reports.reports,
        now,
    )?;
    on_progress(&SyncProgress::IntegrityCheckFinished {
        verification: verification.clone(),
    });
    Ok(Some(verification))
}

fn is_unsupported_integrity_error(err: &SyncError) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("integrity unsupported")
        || message.contains("http status: 404")
        || message.contains("http status: 405")
}

fn cached_record_to_wire(
    config: &EnabledSyncConfig,
    record: &CachedUsageRecord,
    dedup_key: &str,
) -> Result<WireRecord, SyncError> {
    let wire = WireRecord {
        schema_version: ai_usage_proto::SCHEMA_VERSION,
        host_id: config.machine_id.clone(),
        vendor: record.vendor.clone(),
        dedup_key: dedup_key.to_string(),
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
    use ai_usage_proto::{
        PullResponse, RecordKey, SCHEMA_VERSION, SequencedWireRecord, UploadResponse, WireRecord,
    };

    struct FakeTransport {
        uploads: RefCell<Vec<Vec<WireRecord>>>,
        pulls: RefCell<Vec<PullResponse>>,
        pull_requests: RefCell<Vec<(u64, String, usize)>>,
        integrity_submissions: RefCell<Vec<IntegrityReport>>,
        integrity_reports: RefCell<Vec<IntegrityReport>>,
    }

    struct RejectOmpTransport {
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
        ) -> Result<PullResponse, SyncError> {
            self.pull_requests
                .borrow_mut()
                .push((after_seq, exclude_host.to_string(), limit));
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

        fn snapshot_records(
            &self,
            batch: &SnapshotRecordBatch,
        ) -> Result<UploadResponse, SyncError> {
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
        let correct =
            remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
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
                pull_vendors: current_pull_state_fingerprint(),
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
        let correct =
            remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
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
                pull_vendors: current_pull_state_fingerprint(),
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

        assert_eq!(
            transport.pull_requests.borrow().as_slice(),
            &[
                (10, String::new(), PULL_LIMIT),
                (0, String::new(), PULL_LIMIT)
            ]
        );
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

        assert_eq!(
            transport.pull_requests.borrow()[0],
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
