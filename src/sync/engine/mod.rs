use crate::data::cache::{self, CachedUsageRecord, RemoteUsageRecord};
use crate::data::{TokenUsage, UsageCost, UsageEntry};
use crate::sync::config::EnabledSyncConfig;
use crate::sync::keys::assign_sync_dedup_keys;
use crate::sync::state;
use crate::time_utils::parse_timestamp;
use ai_usage_proto::{
    IntegrityReport, IntegrityReportList, IntegritySubmitResponse, PullResponse, RecordFingerprint,
    SnapshotDiffRequest, SnapshotDiffResponse, SnapshotFinalizeRequest, SnapshotFinalizeResponse,
    SnapshotRecordBatch, UploadResponse, WireRecord,
};
use chrono::{DateTime, Duration, Utc};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::path::Path;

pub const SUPPORTED_PULL_VENDORS: [&str; 5] = ["claude", "codex", "gemini", "kimi", "omp"];
const VENDORS: [&str; 5] = SUPPORTED_PULL_VENDORS;
/// The previous release's vendor set, offered as a fallback when an older
/// server rejects [`SUPPORTED_PULL_VENDORS`]. Keeps sync alive (minus the new
/// vendor) while the client is upgraded before the server.
const PREVIOUS_PULL_VENDORS: [&str; 4] = ["claude", "codex", "gemini", "omp"];
/// Vendors every supported server generation knows. An "invalid vendor"
/// rejection for one of these signals a broken server or proxy rather than
/// version skew, and must fail the upload loudly instead of being held back.
const CORE_VENDORS: [&str; 3] = ["claude", "codex", "gemini"];
const BATCH_SIZE: usize = 1000;
const PULL_LIMIT: usize = 5_000;
const SNAPSHOT_DIFF_TARGET_BYTES: usize = 700_000;
const UPLOAD_TARGET_BYTES: usize = 700_000;
const INTEGRITY_RECHECK_INTERVAL: Duration = Duration::hours(6);
const LEGACY_SNAPSHOT_RECEIPT_MAX_AGE_SECS: u64 = 6 * 60 * 60;
const LEGACY_PULL_BACKFILL_INTERVAL: Duration = Duration::hours(6);
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
    UploadVendorHeldBack {
        vendor: String,
        records: usize,
    },
    PullVendorsUnavailable {
        vendors: Vec<String>,
    },
    IntegrityUnsupported,
    IntegrityReportSubmitted {
        record_count: u64,
        range_end_utc: String,
    },
    IntegrityCheckFinished {
        verification: crate::sync::integrity::IntegrityVerification,
    },
    IntegrityCheckReused {
        checked_hosts: usize,
    },
}

/// What an upload cycle actually managed to send.
#[derive(Debug, Default)]
pub struct UploadOutcome {
    /// Vendors whose records an older server rejected as unsupported; they
    /// stay local (and unlogged) until the server is upgraded.
    pub held_back_vendors: Vec<String>,
    /// Whether the completed upload changed data covered by the integrity
    /// range ending at the current UTC midnight.
    pub stable_data_changed: bool,
}

/// What a pull cycle actually fetched.
#[derive(Debug)]
pub struct PullOutcome {
    /// False when the server rejected the current vendor set and the pull
    /// ran with the previous release's set instead.
    pub used_full_vendor_set: bool,
    /// Whether this pull changed data covered by the current integrity range.
    pub stable_data_changed: bool,
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
        supported_vendors: &[&str],
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
    fn server_instance_id(&self) -> Result<Option<String>, SyncError> {
        Ok(None)
    }
    fn remote_snapshot_state(&self, _host_id: &str) -> Result<RemoteSnapshotState, SyncError> {
        Ok(RemoteSnapshotState::Unavailable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteSnapshotState {
    Unavailable,
    Missing,
    Present {
        record_count: u64,
        content_revision: Option<u64>,
    },
}

#[derive(Clone, Copy)]
enum RemoteSnapshotStatus {
    Current { content_revision: Option<u64> },
    Stale,
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
    let upload = run_upload_once_with_progress(cache_root, config, transport, &mut on_progress)?;
    if upload.held_back_vendors.is_empty() {
        let pull = run_pull_once_with_progress(cache_root, config, transport, &mut on_progress)?;
        if !pull.used_full_vendor_set {
            return Ok(());
        }
        let now = Utc::now();
        if !upload.stable_data_changed
            && !pull.stable_data_changed
            && let Some(checked_hosts) = reusable_integrity_check(
                cache_root,
                config,
                transport.server_instance_id()?.as_deref(),
                now,
            )
        {
            on_progress(&SyncProgress::IntegrityCheckReused { checked_hosts });
            return Ok(());
        }
        run_integrity_once_with_repair(cache_root, config, transport, on_progress)
    } else {
        // Degraded cycle (older server): integrity digests would count the
        // held-back records the server cannot distribute, tripping every
        // peer's verification into destructive repairs. Pull still runs.
        run_pull_once_with_progress(cache_root, config, transport, &mut on_progress).map(|_| ())
    }
}

pub fn run_upload_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<UploadOutcome, SyncError>
where
    F: FnMut(&SyncProgress),
{
    if let Some(outcome) =
        run_snapshot_upload_once_with_progress(cache_root, config, transport, &mut on_progress)?
    {
        return Ok(outcome);
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

    let mut total_records = upload_groups.iter().map(Vec::len).sum::<usize>();
    let mut total_batches = upload_groups
        .iter()
        .map(|records| upload_candidate_chunks(records).len())
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
    let mut held_back_vendors: Vec<String> = Vec::new();
    let integrity_range_end = crate::sync::integrity::integrity_range_end_utc(Utc::now());
    let mut stable_data_changed = false;
    for group in upload_groups {
        let batches = upload_candidate_chunks(&group);
        let group_batch_count = batches.len();
        for (group_batch_index, batch) in batches.iter().copied().enumerate() {
            let wire_records: Vec<WireRecord> = batch
                .iter()
                .map(|candidate| candidate.wire.clone())
                .collect();
            let response = match transport.upload(&wire_records) {
                Ok(response) => response,
                Err(err)
                    if is_unsupported_vendor_error(&err)
                        && !CORE_VENDORS.contains(&wire_records[0].vendor.as_str()) =>
                {
                    // An older server that does not know this group's vendor
                    // yet (groups are single-vendor). Hold the whole vendor
                    // back: drop its remaining batches from the plan so the
                    // progress totals reconcile, and leave its records
                    // unlogged so they upload once the server is upgraded.
                    let vendor = wire_records[0].vendor.clone();
                    let remaining_records = batches[group_batch_index..]
                        .iter()
                        .map(|remaining| remaining.len())
                        .sum::<usize>();
                    total_records -= remaining_records;
                    total_batches -= group_batch_count - group_batch_index;
                    on_progress(&SyncProgress::UploadVendorHeldBack {
                        vendor: vendor.clone(),
                        records: remaining_records,
                    });
                    if !held_back_vendors.contains(&vendor) {
                        held_back_vendors.push(vendor);
                    }
                    break;
                }
                Err(err) => return Err(err),
            };
            uploaded_records += batch.len();
            accepted += response.accepted;
            ignored += response.ignored;
            stable_data_changed |= batch.iter().any(|candidate| {
                timestamp_precedes_integrity_range(&candidate.wire.timestamp, integrity_range_end)
            });
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

    Ok(UploadOutcome {
        held_back_vendors,
        stable_data_changed,
    })
}

fn run_snapshot_upload_once_with_progress<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<Option<UploadOutcome>, SyncError>
where
    F: FnMut(&SyncProgress),
{
    let snapshot_id = snapshot_id(&config.machine_id);
    let cache_generation =
        crate::sync::cache_generation::local_cache_generation(cache_root, &VENDORS);
    let server_instance_id = transport.server_instance_id()?;
    let sync_scope = crate::sync::cache_generation::sync_scope_fingerprint(
        config,
        server_instance_id.as_deref(),
    );
    let receipt = state::snapshot_cache_receipt(cache_root, &sync_scope);
    let remote_status = receipt
        .as_ref()
        .map(|receipt| remote_snapshot_status(transport, &config.machine_id, receipt))
        .transpose()?;
    let mut requires_full_reconciliation =
        receipt.is_none() || matches!(remote_status, Some(RemoteSnapshotStatus::Stale));
    if let Some(receipt) = receipt
        .as_ref()
        .filter(|receipt| receipt.cache_generation == cache_generation)
        && matches!(remote_status, Some(RemoteSnapshotStatus::Current { .. }))
    {
        on_progress(&SyncProgress::UploadPlanned {
            total_records: 0,
            total_batches: 0,
            skipped_records: receipt.record_count,
        });
        on_progress(&SyncProgress::UploadFinished {
            uploaded_records: 0,
            total_records: 0,
            accepted: 0,
            ignored: 0,
        });
        return Ok(Some(UploadOutcome::default()));
    }

    let mut previous = state::load_snapshot_upload_state(cache_root);
    let snapshot = collect_snapshot_records(cache_root, config)?;
    if !requires_full_reconciliation
        && previous.schema_version == SNAPSHOT_UPLOAD_STATE_VERSION
        && previous.full_hash == snapshot.full_hash
        && let Some(receipt) = receipt.as_ref()
        && receipt.record_count == snapshot.records.len()
        && let Some(RemoteSnapshotStatus::Current { content_revision }) = remote_status
    {
        previous.cache_generation = cache_generation;
        state::save_snapshot_upload_state(cache_root, &previous, &sync_scope, content_revision)?;
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
        return Ok(Some(UploadOutcome::default()));
    }

    requires_full_reconciliation |= previous.schema_version != SNAPSHOT_UPLOAD_STATE_VERSION
        || previous
            .record_hashes
            .keys()
            .any(|key| !snapshot.record_hashes.contains_key(key));
    let manifest_records = if requires_full_reconciliation {
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
    let integrity_range_end = crate::sync::integrity::integrity_range_end_utc(Utc::now());
    let stable_data_changed = requires_full_reconciliation
        || manifest_records.iter().any(|record| {
            timestamp_precedes_integrity_range(&record.wire.timestamp, integrity_range_end)
        });

    let mut needed_keys = BTreeSet::new();
    let mut chunks = snapshot_fingerprint_chunks(&manifest_records);
    if chunks.is_empty() && requires_full_reconciliation {
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
            // An unsupported-vendor rejection means the server predates one
            // of our vendors: fall back to batch upload, which can hold the
            // unknown vendor back while still uploading the others.
            Err(err)
                if is_unsupported_snapshot_error(&err) || is_unsupported_vendor_error(&err) =>
            {
                return Ok(None);
            }
            Err(err) => return Err(err),
        }
    }

    let needed_records = snapshot
        .records
        .iter()
        .filter(|record| needed_keys.contains(&record.key))
        .collect::<Vec<_>>();
    let skipped_records = snapshot.records.len().saturating_sub(needed_records.len());
    let record_batches = snapshot_record_chunks(&needed_records);
    let total_batches = record_batches.len();
    on_progress(&SyncProgress::UploadPlanned {
        total_records: needed_records.len(),
        total_batches,
        skipped_records,
    });

    let mut uploaded_records = 0usize;
    let mut accepted = 0usize;
    let mut ignored = 0usize;
    for (batch_index, batch) in record_batches.into_iter().enumerate() {
        let records = batch
            .iter()
            .map(|record| record.wire.clone())
            .collect::<Vec<_>>();
        let response = match transport.snapshot_records(&SnapshotRecordBatch {
            host_id: config.machine_id.clone(),
            snapshot_id: snapshot_id.clone(),
            records,
        }) {
            Ok(response) => response,
            // A server that defers vendor validation to the record upload:
            // fall back to batch upload, which holds unknown vendors back.
            Err(err) if is_unsupported_vendor_error(&err) => return Ok(None),
            Err(err) => return Err(err),
        };
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

    if requires_full_reconciliation {
        transport.snapshot_finalize(&SnapshotFinalizeRequest {
            host_id: config.machine_id.clone(),
            snapshot_id,
        })?;
    }

    let content_revision =
        reconciled_snapshot_revision(transport, &config.machine_id, snapshot.records.len())?;
    state::save_snapshot_upload_state(
        cache_root,
        &state::SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_VERSION,
            full_hash: snapshot.full_hash,
            cache_generation,
            record_hashes: snapshot.record_hashes,
        },
        &sync_scope,
        content_revision,
    )?;
    on_progress(&SyncProgress::UploadFinished {
        uploaded_records,
        total_records: needed_records.len(),
        accepted,
        ignored,
    });
    Ok(Some(UploadOutcome {
        held_back_vendors: Vec::new(),
        stable_data_changed,
    }))
}

fn remote_snapshot_status(
    transport: &impl SyncTransport,
    host_id: &str,
    receipt: &state::SnapshotCacheReceipt,
) -> Result<RemoteSnapshotStatus, SyncError> {
    Ok(match transport.remote_snapshot_state(host_id)? {
        RemoteSnapshotState::Unavailable => RemoteSnapshotStatus::Current {
            content_revision: None,
        },
        RemoteSnapshotState::Missing => RemoteSnapshotStatus::Stale,
        RemoteSnapshotState::Present {
            record_count,
            content_revision,
        } if record_count == receipt.record_count as u64 => {
            let revision_is_current = match (receipt.content_revision, content_revision) {
                (Some(previous), Some(current)) => previous == current,
                (None, None) => {
                    state_receipt_age_secs(receipt.verified_at_secs)
                        <= LEGACY_SNAPSHOT_RECEIPT_MAX_AGE_SECS
                }
                _ => false,
            };
            if revision_is_current {
                RemoteSnapshotStatus::Current { content_revision }
            } else {
                RemoteSnapshotStatus::Stale
            }
        }
        RemoteSnapshotState::Present { .. } => RemoteSnapshotStatus::Stale,
    })
}

fn reconciled_snapshot_revision(
    transport: &impl SyncTransport,
    host_id: &str,
    local_record_count: usize,
) -> Result<Option<u64>, SyncError> {
    Ok(match transport.remote_snapshot_state(host_id)? {
        RemoteSnapshotState::Present {
            record_count,
            content_revision,
        } if record_count == local_record_count as u64 => content_revision,
        RemoteSnapshotState::Unavailable
        | RemoteSnapshotState::Missing
        | RemoteSnapshotState::Present { .. } => None,
    })
}

fn state_receipt_age_secs(verified_at_secs: u64) -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .saturating_sub(verified_at_secs)
}

#[derive(Debug)]
struct SnapshotRecords {
    records: Vec<SnapshotRecord>,
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
    let full_hash = snapshot_full_hash(&records);
    Ok(SnapshotRecords {
        records,
        full_hash,
        record_hashes,
    })
}

fn snapshot_fingerprint_chunks(records: &[&SnapshotRecord]) -> Vec<Vec<RecordFingerprint>> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0usize;
    for record in records {
        let fingerprint = RecordFingerprint {
            vendor: record.key.0.clone(),
            dedup_key: record.key.1.clone(),
            record_hash: record.record_hash.clone(),
        };
        let fingerprint_bytes = serde_json::to_vec(&fingerprint)
            .expect("validated fingerprints always serialize")
            .len()
            + 1;
        if !current.is_empty() && current_bytes + fingerprint_bytes > SNAPSHOT_DIFF_TARGET_BYTES {
            chunks.push(current);
            current = Vec::new();
            current_bytes = 0;
        }
        current.push(fingerprint);
        current_bytes += fingerprint_bytes;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn snapshot_record_chunks<'a>(records: &'a [&'a SnapshotRecord]) -> Vec<&'a [&'a SnapshotRecord]> {
    batch_ranges_by_bytes(records, |record| wire_record_bytes(&record.wire))
        .into_iter()
        .map(|range| &records[range])
        .collect()
}

fn upload_candidate_chunks(records: &[UploadCandidate]) -> Vec<&[UploadCandidate]> {
    batch_ranges_by_bytes(records, |record| wire_record_bytes(&record.wire))
        .into_iter()
        .map(|range| &records[range])
        .collect()
}

fn batch_ranges_by_bytes<T>(
    records: &[T],
    mut record_bytes: impl FnMut(&T) -> usize,
) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut current_bytes = 128usize;
    for (index, record) in records.iter().enumerate() {
        let bytes = record_bytes(record);
        let reached_count_limit = index - start >= BATCH_SIZE;
        let reached_byte_limit = index > start && current_bytes + bytes > UPLOAD_TARGET_BYTES;
        if reached_count_limit || reached_byte_limit {
            ranges.push(start..index);
            start = index;
            current_bytes = 128;
        }
        current_bytes += bytes;
    }
    if start < records.len() {
        ranges.push(start..records.len());
    }
    ranges
}

fn wire_record_bytes(record: &WireRecord) -> usize {
    serde_json::to_vec(record)
        .expect("validated wire records always serialize")
        .len()
        + 1
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

fn pull_state_fingerprint_for(vendors: &[&str]) -> Vec<String> {
    let mut fingerprint: Vec<String> = vendors.iter().map(|vendor| (*vendor).to_string()).collect();
    fingerprint.push(PULL_SCOPE_ALL_HOSTS_MARKER.to_string());
    fingerprint
}

/// True when every element of `subset` (vendor names plus the scope marker)
/// also appears in `superset`. A pull cursor advanced under a superset of
/// vendors never skipped a subset vendor's records, so it stays valid for
/// the subset.
fn is_fingerprint_subset(subset: &[String], superset: &[String]) -> bool {
    subset.iter().all(|item| superset.contains(item))
}

fn try_first_pull_page(
    transport: &impl SyncTransport,
    sync_state: &state::SyncState,
    vendors: &'static [&'static str],
) -> Result<(&'static [&'static str], Vec<String>, PullResponse), SyncError> {
    let fingerprint = pull_state_fingerprint_for(vendors);
    let start_seq = if is_fingerprint_subset(&fingerprint, &sync_state.pull_vendors) {
        sync_state.last_seen_seq
    } else {
        0
    };
    let response = transport.pull(start_seq, "", PULL_LIMIT, vendors)?;
    Ok((vendors, fingerprint, response))
}

/// Fetch the first pull page. An older server rejects a vendor name it does
/// not know ("invalid vendor"); retrying with the previous release's set
/// keeps pull alive until the server is upgraded. Returns the accepted set,
/// its fingerprint, and the page fetched with it, so the caller commits any
/// cache migration only for a set the server actually serves.
fn negotiate_first_pull_page(
    transport: &impl SyncTransport,
    sync_state: &state::SyncState,
) -> Result<(&'static [&'static str], Vec<String>, PullResponse), SyncError> {
    match try_first_pull_page(transport, sync_state, &SUPPORTED_PULL_VENDORS) {
        Err(err) if is_unsupported_vendor_error(&err) => {
            try_first_pull_page(transport, sync_state, &PREVIOUS_PULL_VENDORS)
        }
        result => result,
    }
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
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<PullOutcome, SyncError>
where
    F: FnMut(&SyncProgress),
{
    let mut sync_state = state::load_sync_state(cache_root);
    let pull_started_at = Utc::now();
    let server_instance_id = transport.server_instance_id()?;
    let pull_scope = crate::sync::cache_generation::server_scope_fingerprint(
        config,
        server_instance_id.as_deref(),
    );
    let scope_changed = sync_state.pull_scope != pull_scope;
    let legacy_backfill_due = server_instance_id.is_none()
        && legacy_pull_backfill_due(sync_state.last_full_pull.as_deref(), pull_started_at);
    let requires_full_backfill = scope_changed || legacy_backfill_due;
    let mut request_state = sync_state.clone();
    if requires_full_backfill {
        request_state.last_seen_seq = 0;
    }
    let (vendors, fingerprint, mut response) =
        negotiate_first_pull_page(transport, &request_state)?;
    let mut stable_data_changed = false;
    if vendors != SUPPORTED_PULL_VENDORS {
        let unavailable: Vec<String> = SUPPORTED_PULL_VENDORS
            .iter()
            .filter(|vendor| !vendors.contains(vendor))
            .map(|vendor| (*vendor).to_string())
            .collect();
        on_progress(&SyncProgress::PullVendorsUnavailable {
            vendors: unavailable,
        });
    }

    // Commit the vendor-set migration only after the server has accepted the
    // set: an older server rejecting the new vendor must never wipe the
    // existing remote cache.
    let mut performed_full_backfill = false;
    if requires_full_backfill {
        cache::clear_remote_cache(cache_root)?;
        stable_data_changed = true;
        performed_full_backfill = true;
        sync_state.last_seen_seq = 0;
        sync_state.pull_vendors = fingerprint;
        sync_state.pull_scope = pull_scope;
    } else if sync_state.pull_vendors != fingerprint {
        if is_fingerprint_subset(&fingerprint, &sync_state.pull_vendors) {
            // Downgrade (server rollback): cursor and cache stay valid for a
            // vendor subset, so keep the already-pulled records visible and
            // only adopt the reduced fingerprint. A later re-expansion then
            // mismatches again and triggers the full backfill below.
            sync_state.pull_vendors = fingerprint;
        } else {
            cache::clear_remote_cache(cache_root)?;
            stable_data_changed = true;
            performed_full_backfill = true;
            sync_state.last_seen_seq = 0;
            sync_state.pull_vendors = fingerprint;
        }
    }
    let mut page_index = 0;
    let mut pulled_records = 0;
    let integrity_range_end = crate::sync::integrity::integrity_range_end_utc(Utc::now());

    loop {
        page_index += 1;
        pulled_records += response.records.len();
        stable_data_changed |= response.records.iter().any(|record| {
            timestamp_precedes_integrity_range(&record.record.timestamp, integrity_range_end)
        });
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
        response = transport.pull(sync_state.last_seen_seq, "", PULL_LIMIT, vendors)?;
    }

    if performed_full_backfill {
        sync_state.last_full_pull = Some(pull_started_at.to_rfc3339());
    }
    sync_state.last_successful_sync = Some(Utc::now().to_rfc3339());
    sync_state.last_error = None;
    state::save_sync_state(cache_root, &sync_state)?;
    on_progress(&SyncProgress::PullFinished {
        pages: page_index,
        pulled_records,
        max_seq: sync_state.last_seen_seq,
    });
    Ok(PullOutcome {
        used_full_vendor_set: vendors == SUPPORTED_PULL_VENDORS,
        stable_data_changed,
    })
}

fn legacy_pull_backfill_due(last_full_pull: Option<&str>, now: DateTime<Utc>) -> bool {
    let Some(last_full_pull) = last_full_pull
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
    else {
        return true;
    };
    last_full_pull > now || now - last_full_pull >= LEGACY_PULL_BACKFILL_INTERVAL
}

fn timestamp_precedes_integrity_range(timestamp: &str, range_end: DateTime<Utc>) -> bool {
    parse_timestamp(timestamp).is_some_and(|value| value.with_timezone(&Utc) < range_end)
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
    let pull = run_pull_once_with_progress(cache_root, config, transport, &mut on_progress)?;
    if !pull.used_full_vendor_set {
        // Degraded pull (older server): the local remote cache is missing
        // the held-back vendor, so verifying peers' full-set digests would
        // fail and trigger destructive repairs. Skip integrity this cycle.
        return Ok(());
    }
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
    let checked_at = Utc::now();
    let server_instance_id = transport.server_instance_id()?;
    let verification =
        run_integrity_once_with_repair_result(cache_root, config, transport, &mut on_progress)?;
    persist_successful_integrity_check(
        cache_root,
        config,
        server_instance_id.as_deref(),
        checked_at,
        verification.as_ref(),
    )?;
    Ok(())
}

fn run_integrity_once_with_repair_result<F>(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    transport: &impl SyncTransport,
    mut on_progress: F,
) -> Result<Option<crate::sync::integrity::IntegrityVerification>, SyncError>
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
        return run_integrity_once_with_progress(cache_root, config, transport, &mut on_progress);
    }
    Ok(verification)
}

fn reusable_integrity_check(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    server_instance_id: Option<&str>,
    now: DateTime<Utc>,
) -> Option<usize> {
    let check = state::load_sync_state(cache_root).integrity_check?;
    let sync_scope =
        crate::sync::cache_generation::sync_scope_fingerprint(config, server_instance_id);
    let checked_at = DateTime::parse_from_rfc3339(&check.checked_at)
        .ok()?
        .with_timezone(&Utc);
    let range_end = DateTime::parse_from_rfc3339(&check.range_end_utc)
        .ok()?
        .with_timezone(&Utc);
    let current_range_end = crate::sync::integrity::integrity_range_end_utc(now);
    (check.sync_scope == sync_scope
        && range_end == current_range_end
        && checked_at <= now
        && now - checked_at < INTEGRITY_RECHECK_INTERVAL)
        .then_some(check.checked_hosts)
}

fn persist_successful_integrity_check(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    server_instance_id: Option<&str>,
    checked_at: DateTime<Utc>,
    verification: Option<&crate::sync::integrity::IntegrityVerification>,
) -> Result<(), SyncError> {
    let Some(crate::sync::integrity::IntegrityVerification::Checked { checked_hosts }) =
        verification
    else {
        return Ok(());
    };
    let mut sync_state = state::load_sync_state(cache_root);
    sync_state.integrity_check = Some(state::IntegrityCheckState {
        checked_at: checked_at.to_rfc3339(),
        range_end_utc: crate::sync::integrity::integrity_range_end_utc(checked_at).to_rfc3339(),
        checked_hosts: *checked_hosts,
        sync_scope: crate::sync::cache_generation::sync_scope_fingerprint(
            config,
            server_instance_id,
        ),
    });
    state::save_sync_state(cache_root, &sync_state)?;
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
            session_id: None,
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
mod tests;
