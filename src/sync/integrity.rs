use crate::data::UsageEntry;
use crate::data::cache::{self, CachedUsageRecord, RemoteUsageRecord};
use crate::sync::config::EnabledSyncConfig;
use crate::sync::engine::{SUPPORTED_PULL_VENDORS, SyncError};
use crate::sync::keys::assign_sync_dedup_keys;
use chrono::{DateTime, SecondsFormat, Timelike, Utc};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use vibe_usage_proto::{INTEGRITY_ALGORITHM, IntegrityReport};

const TRANSCRIPT_FORMAT: &str = "integrity-transcript-v1";
const TRANSCRIPT_DIR: &str = "integrity";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityVerification {
    Checked { checked_hosts: usize },
    Failed { failures: Vec<IntegrityFailure> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityFailure {
    pub host_id: String,
    pub range_end_utc: String,
    pub expected_record_count: u64,
    pub actual_record_count: u64,
    pub expected_digest_sha256: String,
    pub actual_digest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct CanonicalIntegrityRecord {
    host_id: String,
    vendor: String,
    dedup_key: String,
    timestamp: String,
    session_start_time: String,
    session_end_time: String,
    model: String,
    effort: Option<String>,
    fast_tier: i8,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    reasoning_output_tokens: i64,
    cost_input: Option<f64>,
    cost_output: Option<f64>,
    cost_cache_read: Option<f64>,
    cost_cache_creation: Option<f64>,
}

impl Eq for CanonicalIntegrityRecord {}

#[derive(Debug, Clone)]
struct DigestedIntegrityRecord {
    canonical: CanonicalIntegrityRecord,
    canonical_json_len: usize,
    record_sha256: String,
}

#[derive(Debug, Clone)]
struct DigestResult {
    record_count: u64,
    digest_sha256: String,
    stable_records: Vec<DigestedIntegrityRecord>,
}

impl PartialOrd for CanonicalIntegrityRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CanonicalIntegrityRecord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.host_id
            .cmp(&other.host_id)
            .then_with(|| self.vendor.cmp(&other.vendor))
            .then_with(|| self.dedup_key.cmp(&other.dedup_key))
            .then_with(|| self.timestamp.cmp(&other.timestamp))
    }
}

pub fn integrity_range_end_utc(now: DateTime<Utc>) -> DateTime<Utc> {
    now.with_hour(0)
        .and_then(|value| value.with_minute(0))
        .and_then(|value| value.with_second(0))
        .and_then(|value| value.with_nanosecond(0))
        .expect("UTC midnight is valid")
}

pub fn build_local_report_at(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    now: DateTime<Utc>,
    computed_at: DateTime<Utc>,
) -> Result<IntegrityReport, SyncError> {
    build_local_report_for_range(
        cache_root,
        config,
        integrity_range_end_utc(now),
        computed_at,
    )
}

pub fn build_local_report_for_range(
    cache_root: &Path,
    config: &EnabledSyncConfig,
    range_end_utc: DateTime<Utc>,
    computed_at: DateTime<Utc>,
) -> Result<IntegrityReport, SyncError> {
    let mut records = Vec::new();
    for vendor in SUPPORTED_PULL_VENDORS {
        for keyed in assign_sync_dedup_keys(cache::load_vendor_cached_records(cache_root, vendor)) {
            records.push(local_canonical_record(
                &config.machine_id,
                keyed.record,
                keyed.dedup_key,
            ));
        }
    }
    let (report, digest) =
        build_report_from_records(&config.machine_id, records, range_end_utc, computed_at)?;
    let _ = write_local_transcript(cache_root, &config.machine_id, &report, &digest);
    Ok(report)
}

#[cfg(test)]
pub fn build_remote_report_at(
    cache_root: &Path,
    host_id: &str,
    range_end_utc: DateTime<Utc>,
    computed_at: DateTime<Utc>,
) -> Result<IntegrityReport, SyncError> {
    build_remote_report_digest_at(cache_root, host_id, range_end_utc, computed_at)
        .map(|(report, _)| report)
}

fn build_remote_report_digest_at(
    cache_root: &Path,
    host_id: &str,
    range_end_utc: DateTime<Utc>,
    computed_at: DateTime<Utc>,
) -> Result<(IntegrityReport, DigestResult), SyncError> {
    let hosts = HashSet::from([host_id.to_string()]);
    let records = cache::load_remote_entries(cache_root, Some(&hosts))
        .into_iter()
        .map(remote_canonical_record)
        .collect::<Vec<_>>();
    build_report_from_records(host_id, records, range_end_utc, computed_at)
}

pub fn verify_remote_reports_at(
    cache_root: &Path,
    local_host_id: &str,
    reports: &[IntegrityReport],
    computed_at: DateTime<Utc>,
) -> Result<IntegrityVerification, SyncError> {
    let mut checked_hosts = 0usize;
    let mut failures = Vec::new();

    for report in reports {
        report
            .validate()
            .map_err(|err| SyncError::new(format!("invalid integrity report: {err}")))?;
        if report.host_id == local_host_id {
            continue;
        }
        let range_end_utc = DateTime::parse_from_rfc3339(&report.range_end_utc)
            .map_err(|err| SyncError::new(format!("invalid integrity range end: {err}")))?
            .with_timezone(&Utc);
        let (actual, digest) =
            build_remote_report_digest_at(cache_root, &report.host_id, range_end_utc, computed_at)?;
        checked_hosts += 1;
        let matches_report = actual.digest_sha256 == report.digest_sha256
            && actual.record_count == report.record_count;
        let _ = write_remote_transcript(
            cache_root,
            local_host_id,
            report,
            &actual,
            &digest,
            matches_report,
        );
        if !matches_report {
            failures.push(IntegrityFailure {
                host_id: report.host_id.clone(),
                range_end_utc: report.range_end_utc.clone(),
                expected_record_count: report.record_count,
                actual_record_count: actual.record_count,
                expected_digest_sha256: report.digest_sha256.clone(),
                actual_digest_sha256: actual.digest_sha256,
            });
        }
    }

    if failures.is_empty() {
        Ok(IntegrityVerification::Checked { checked_hosts })
    } else {
        Ok(IntegrityVerification::Failed { failures })
    }
}

fn build_report_from_records(
    host_id: &str,
    records: Vec<CanonicalIntegrityRecord>,
    range_end_utc: DateTime<Utc>,
    computed_at: DateTime<Utc>,
) -> Result<(IntegrityReport, DigestResult), SyncError> {
    let digest = digest_records(records, range_end_utc)?;
    let report = IntegrityReport {
        host_id: host_id.to_string(),
        algorithm: INTEGRITY_ALGORITHM.to_string(),
        range_end_utc: format_utc_timestamp(range_end_utc),
        record_count: digest.record_count,
        digest_sha256: digest.digest_sha256.clone(),
        computed_at: format_utc_timestamp(computed_at),
    };
    report
        .validate()
        .map_err(|err| SyncError::new(format!("invalid integrity report: {err}")))?;
    Ok((report, digest))
}

fn digest_records(
    records: Vec<CanonicalIntegrityRecord>,
    range_end_utc: DateTime<Utc>,
) -> Result<DigestResult, SyncError> {
    let mut stable_records = records
        .into_iter()
        .filter_map(
            |record| match DateTime::parse_from_rfc3339(&record.timestamp) {
                Ok(timestamp) if timestamp.with_timezone(&Utc) < range_end_utc => Some(Ok(record)),
                Ok(_) => None,
                Err(err) => Some(Err(SyncError::new(format!(
                    "invalid integrity record timestamp: {err}"
                )))),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    stable_records.sort();

    let mut hasher = Sha256::new();
    let mut digested_records = Vec::with_capacity(stable_records.len());
    for record in stable_records {
        let bytes = serde_json::to_vec(&record)
            .map_err(|err| SyncError::new(format!("serialize integrity record: {err}")))?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(&bytes);
        let record_sha256 = sha256_hex(&bytes);
        digested_records.push(DigestedIntegrityRecord {
            canonical: record,
            canonical_json_len: bytes.len(),
            record_sha256,
        });
    }
    let digest = hasher.finalize();
    Ok(DigestResult {
        record_count: digested_records.len() as u64,
        digest_sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        stable_records: digested_records,
    })
}

fn write_local_transcript(
    cache_root: &Path,
    local_host_id: &str,
    report: &IntegrityReport,
    digest: &DigestResult,
) -> std::io::Result<()> {
    let summary = json!({
        "line": "summary",
        "format": TRANSCRIPT_FORMAT,
        "view": "local",
        "observer_host_id": local_host_id,
        "subject_host_id": report.host_id,
        "algorithm": report.algorithm,
        "range_end_utc": report.range_end_utc,
        "computed_at": report.computed_at,
        "record_count": report.record_count,
        "digest_sha256": report.digest_sha256,
    });
    write_transcript(cache_root, "local", &report.host_id, summary, digest)
}

fn write_remote_transcript(
    cache_root: &Path,
    local_host_id: &str,
    expected: &IntegrityReport,
    actual: &IntegrityReport,
    digest: &DigestResult,
    matches_report: bool,
) -> std::io::Result<()> {
    let summary = json!({
        "line": "summary",
        "format": TRANSCRIPT_FORMAT,
        "view": "remote",
        "observer_host_id": local_host_id,
        "subject_host_id": expected.host_id,
        "algorithm": expected.algorithm,
        "range_end_utc": expected.range_end_utc,
        "server_computed_at": expected.computed_at,
        "computed_at": actual.computed_at,
        "status": if matches_report { "checked" } else { "failed" },
        "expected_record_count": expected.record_count,
        "actual_record_count": actual.record_count,
        "expected_digest_sha256": expected.digest_sha256,
        "server_digest_sha256": expected.digest_sha256,
        "actual_digest_sha256": actual.digest_sha256,
    });
    write_transcript(cache_root, "remote", &expected.host_id, summary, digest)
}

fn write_transcript(
    cache_root: &Path,
    view: &str,
    subject_host_id: &str,
    summary: serde_json::Value,
    digest: &DigestResult,
) -> std::io::Result<()> {
    let path = transcript_path(cache_root, view, subject_host_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension(format!("jsonl.{}.tmp", std::process::id()));
    {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &summary)?;
        writer.write_all(b"\n")?;
        for (index, record) in digest.stable_records.iter().enumerate() {
            serde_json::to_writer(&mut writer, &transcript_record_line(index, record))?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }
    fs::rename(tmp_path, path)
}

fn transcript_path(cache_root: &Path, view: &str, subject_host_id: &str) -> PathBuf {
    cache_root.join(TRANSCRIPT_DIR).join(format!(
        "{}-{}.jsonl",
        safe_file_stem(view),
        safe_file_stem(subject_host_id)
    ))
}

fn transcript_record_line(index: usize, record: &DigestedIntegrityRecord) -> serde_json::Value {
    let canonical = &record.canonical;
    json!({
        "line": "record",
        "index": index,
        "host_id": canonical.host_id,
        "vendor": canonical.vendor,
        "dedup_key_sha256": sha256_hex(canonical.dedup_key.as_bytes()),
        "timestamp": canonical.timestamp,
        "session_start_time": canonical.session_start_time,
        "session_end_time": canonical.session_end_time,
        "model": canonical.model,
        "effort": canonical.effort,
        "fast_tier": canonical.fast_tier,
        "input_tokens": canonical.input_tokens,
        "output_tokens": canonical.output_tokens,
        "cache_read_input_tokens": canonical.cache_read_input_tokens,
        "cache_creation_input_tokens": canonical.cache_creation_input_tokens,
        "reasoning_output_tokens": canonical.reasoning_output_tokens,
        "cost_input": canonical.cost_input,
        "cost_output": canonical.cost_output,
        "cost_cache_read": canonical.cost_cache_read,
        "cost_cache_creation": canonical.cost_cache_creation,
        "canonical_json_len": record.canonical_json_len,
        "record_sha256": record.record_sha256,
    })
}

fn safe_file_stem(value: &str) -> String {
    let stem: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if stem.is_empty() {
        "unknown".to_string()
    } else {
        stem
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn local_canonical_record(
    host_id: &str,
    record: CachedUsageRecord,
    dedup_key: String,
) -> CanonicalIntegrityRecord {
    canonical_record(host_id.to_string(), record.vendor, dedup_key, record.entry)
}

fn remote_canonical_record(record: RemoteUsageRecord) -> CanonicalIntegrityRecord {
    let host_id = record.entry.host_id.clone().unwrap_or_default();
    canonical_record(host_id, record.vendor, record.dedup_key, record.entry)
}

fn canonical_record(
    host_id: String,
    vendor: String,
    dedup_key: String,
    entry: UsageEntry,
) -> CanonicalIntegrityRecord {
    CanonicalIntegrityRecord {
        host_id,
        vendor,
        dedup_key,
        timestamp: entry.timestamp,
        session_start_time: entry.session_start_time,
        session_end_time: entry.session_end_time,
        model: entry.model,
        effort: entry.effort,
        fast_tier: entry.fast_tier,
        input_tokens: entry.usage.input_tokens,
        output_tokens: entry.usage.output_tokens,
        cache_read_input_tokens: entry.usage.cache_read_input_tokens,
        cache_creation_input_tokens: entry.usage.cache_creation_input_tokens,
        reasoning_output_tokens: entry.usage.reasoning_output_tokens,
        cost_input: entry.costs.map(|costs| costs.input),
        cost_output: entry.costs.map(|costs| costs.output),
        cost_cache_read: entry.costs.map(|costs| costs.cache_read),
        cost_cache_creation: entry.costs.map(|costs| costs.cache_creation),
    }
}

fn format_utc_timestamp(timestamp: DateTime<Utc>) -> String {
    timestamp.to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use crate::data::cache;
    use crate::data::{SourceUsageRecord, TokenUsage, UsageEntry};
    use crate::sync::config::EnabledSyncConfig;
    use chrono::{SecondsFormat, TimeZone, Utc};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vibe-usage-integrity-test-{name}-{stamp}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn enabled_config(machine_id: &str) -> EnabledSyncConfig {
        EnabledSyncConfig {
            server_url: "https://usage.example.com".to_string(),
            token: "0123456789abcdef0123456789abcdef".to_string(),
            machine_id: machine_id.to_string(),
            upload_project_hash: true,
            request_timeout_seconds: 15,
        }
    }

    fn usage_record(dedup_key: &str, timestamp: &str, input_tokens: i64) -> SourceUsageRecord {
        SourceUsageRecord {
            dedup_key: dedup_key.to_string(),
            entry: UsageEntry {
                host_id: None,
                timestamp: timestamp.to_string(),
                parsed_timestamp: crate::time_utils::parse_timestamp(timestamp),
                session_start_time: timestamp.to_string(),
                session_end_time: timestamp.to_string(),
                model: "test-model".to_string(),
                effort: Some("high".to_string()),
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

    fn populate_local(cache_root: &Path, records: Vec<SourceUsageRecord>) {
        let source = cache_root.join("claude.jsonl");
        std::fs::write(&source, "source").expect("write source");
        cache::load_or_update_vendor_cache(cache_root, "claude", vec![source], 1, |_| {
            records.clone()
        });
    }

    fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .expect("read transcript")
            .lines()
            .map(|line| serde_json::from_str(line).expect("parse transcript line"))
            .collect()
    }

    fn remote_record(
        host_id: &str,
        dedup_key: &str,
        timestamp: &str,
        input_tokens: i64,
    ) -> cache::RemoteUsageRecord {
        let mut record = usage_record(dedup_key, timestamp, input_tokens);
        record.entry.host_id = Some(host_id.to_string());
        record.entry.fast_tier = 1;
        cache::RemoteUsageRecord {
            vendor: "claude".to_string(),
            dedup_key: record.dedup_key,
            entry: record.entry,
        }
    }

    #[test]
    fn integrity_range_end_uses_current_utc_midnight() {
        let now = Utc
            .with_ymd_and_hms(2026, 6, 1, 23, 59, 59)
            .single()
            .expect("valid timestamp");

        let range_end = super::integrity_range_end_utc(now);

        assert_eq!(
            range_end.to_rfc3339_opts(SecondsFormat::Secs, true),
            "2026-06-01T00:00:00Z"
        );
    }

    #[test]
    fn integrity_digest_matches_between_local_and_remote_stable_records() {
        let cache_root = unique_temp_dir("local-remote");
        populate_local(
            &cache_root,
            vec![
                usage_record("stable-a", "2026-05-31T23:59:59Z", 10),
                usage_record("cutoff", "2026-06-01T00:00:00Z", 20),
                usage_record("current", "2026-06-01T12:00:00Z", 30),
            ],
        );
        cache::merge_remote_records(
            &cache_root,
            "host-a",
            vec![
                remote_record("host-a", "stable-a", "2026-05-31T23:59:59Z", 10),
                remote_record("host-a", "cutoff", "2026-06-01T00:00:00Z", 20),
                remote_record("host-a", "current", "2026-06-01T12:00:00Z", 30),
            ],
        )
        .expect("merge remote records");
        let now = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 0)
            .single()
            .expect("valid timestamp");
        let computed_at = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 1)
            .single()
            .expect("valid timestamp");

        let local =
            super::build_local_report_at(&cache_root, &enabled_config("host-a"), now, computed_at)
                .expect("local report");
        let remote = super::build_remote_report_at(
            &cache_root,
            "host-a",
            super::integrity_range_end_utc(now),
            computed_at,
        )
        .expect("remote report");

        assert_eq!(local.record_count, 1);
        assert_eq!(local.digest_sha256, remote.digest_sha256);
        assert_eq!(local.range_end_utc, "2026-06-01T00:00:00Z");
    }

    #[test]
    fn local_integrity_report_writes_compact_transcript() {
        let cache_root = unique_temp_dir("local-transcript");
        populate_local(
            &cache_root,
            vec![
                usage_record("stable-a", "2026-05-31T23:59:59Z", 10),
                usage_record("current", "2026-06-01T12:00:00Z", 30),
            ],
        );
        let now = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 0)
            .single()
            .expect("valid timestamp");
        let computed_at = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 1)
            .single()
            .expect("valid timestamp");

        let report =
            super::build_local_report_at(&cache_root, &enabled_config("host-a"), now, computed_at)
                .expect("local report");

        let path = cache_root.join("integrity").join("local-host-a.jsonl");
        let lines = read_jsonl(&path);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["line"], "summary");
        assert_eq!(lines[0]["view"], "local");
        assert_eq!(lines[0]["observer_host_id"], "host-a");
        assert_eq!(lines[0]["subject_host_id"], "host-a");
        assert_eq!(lines[0]["record_count"], 1);
        assert_eq!(lines[0]["digest_sha256"], report.digest_sha256);
        assert_eq!(lines[0]["range_end_utc"], "2026-06-01T00:00:00Z");
        assert_eq!(lines[1]["line"], "record");
        assert_eq!(lines[1]["index"], 0);
        assert_eq!(lines[1]["vendor"], "claude");
        assert_eq!(lines[1]["timestamp"], "2026-05-31T23:59:59Z");
        assert_eq!(lines[1]["input_tokens"], 10);
        assert_eq!(
            lines[1]["record_sha256"]
                .as_str()
                .expect("record hash string")
                .len(),
            64
        );
        assert_eq!(
            lines[1]["dedup_key_sha256"]
                .as_str()
                .expect("dedup key hash string")
                .len(),
            64
        );
        let text = std::fs::read_to_string(path).expect("read transcript text");
        assert!(!text.contains("stable-a"));
        assert!(!text.contains("current"));
    }

    #[test]
    fn local_integrity_uses_stable_keys_for_empty_dedup_records() {
        let cache_root = unique_temp_dir("local-empty-dedup");
        populate_local(
            &cache_root,
            vec![
                usage_record("", "2026-05-31T23:58:00Z", 10),
                usage_record("", "2026-05-31T23:59:00Z", 20),
            ],
        );
        let now = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 0)
            .single()
            .expect("valid timestamp");
        let computed_at = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 1)
            .single()
            .expect("valid timestamp");

        let report =
            super::build_local_report_at(&cache_root, &enabled_config("host-a"), now, computed_at)
                .expect("local report");

        let path = cache_root.join("integrity").join("local-host-a.jsonl");
        let lines = read_jsonl(&path);
        assert_eq!(report.record_count, 2);
        assert_eq!(lines.len(), 3);
        let first = lines[1]["dedup_key_sha256"]
            .as_str()
            .expect("first dedup hash");
        let second = lines[2]["dedup_key_sha256"]
            .as_str()
            .expect("second dedup hash");
        assert_ne!(
            first,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_ne!(
            second,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_ne!(first, second);
    }

    #[test]
    fn integrity_verification_compares_remote_records_with_owner_report() {
        let cache_root = unique_temp_dir("verify-match");
        cache::merge_remote_records(
            &cache_root,
            "host-a",
            vec![remote_record(
                "host-a",
                "stable-a",
                "2026-05-31T23:59:59Z",
                10,
            )],
        )
        .expect("merge remote records");
        let now = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 0)
            .single()
            .expect("valid timestamp");
        let computed_at = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 1)
            .single()
            .expect("valid timestamp");
        let owner_report = super::build_remote_report_at(
            &cache_root,
            "host-a",
            super::integrity_range_end_utc(now),
            computed_at,
        )
        .expect("owner report");

        let verification =
            super::verify_remote_reports_at(&cache_root, "host-b", &[owner_report], computed_at)
                .expect("verify reports");

        assert_eq!(
            verification,
            super::IntegrityVerification::Checked { checked_hosts: 1 }
        );
    }

    #[test]
    fn integrity_verification_fails_when_remote_records_do_not_match_owner_report() {
        let owner_cache = unique_temp_dir("owner-report");
        let viewer_cache = unique_temp_dir("viewer-mismatch");
        cache::merge_remote_records(
            &owner_cache,
            "host-a",
            vec![remote_record(
                "host-a",
                "stable-a",
                "2026-05-31T23:59:59Z",
                10,
            )],
        )
        .expect("merge owner records");
        cache::merge_remote_records(
            &viewer_cache,
            "host-a",
            vec![remote_record(
                "host-a",
                "stable-a",
                "2026-05-31T23:59:59Z",
                99,
            )],
        )
        .expect("merge viewer records");
        let range_end = super::integrity_range_end_utc(
            Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0)
                .single()
                .expect("valid timestamp"),
        );
        let computed_at = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 1)
            .single()
            .expect("valid timestamp");
        let owner_report =
            super::build_remote_report_at(&owner_cache, "host-a", range_end, computed_at)
                .expect("owner report");

        let verification =
            super::verify_remote_reports_at(&viewer_cache, "host-b", &[owner_report], computed_at)
                .expect("verify reports");

        match verification {
            super::IntegrityVerification::Failed { failures } => {
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].host_id, "host-a");
                assert_eq!(failures[0].expected_record_count, 1);
                assert_eq!(failures[0].actual_record_count, 1);
            }
            other => panic!("expected failed verification, got {other:?}"),
        }
    }

    #[test]
    fn remote_integrity_verification_writes_expected_actual_transcript() {
        let owner_cache = unique_temp_dir("owner-transcript");
        let viewer_cache = unique_temp_dir("viewer-transcript");
        cache::merge_remote_records(
            &owner_cache,
            "host-a",
            vec![remote_record(
                "host-a",
                "stable-a",
                "2026-05-31T23:59:59Z",
                10,
            )],
        )
        .expect("merge owner records");
        cache::merge_remote_records(
            &viewer_cache,
            "host-a",
            vec![remote_record(
                "host-a",
                "stable-a",
                "2026-05-31T23:59:59Z",
                99,
            )],
        )
        .expect("merge viewer records");
        let range_end = super::integrity_range_end_utc(
            Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0)
                .single()
                .expect("valid timestamp"),
        );
        let computed_at = Utc
            .with_ymd_and_hms(2026, 6, 1, 12, 0, 1)
            .single()
            .expect("valid timestamp");
        let owner_report =
            super::build_remote_report_at(&owner_cache, "host-a", range_end, computed_at)
                .expect("owner report");

        let verification =
            super::verify_remote_reports_at(&viewer_cache, "host-b", &[owner_report], computed_at)
                .expect("verify reports");

        assert!(matches!(
            verification,
            super::IntegrityVerification::Failed { .. }
        ));
        let path = viewer_cache.join("integrity").join("remote-host-a.jsonl");
        let lines = read_jsonl(&path);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["line"], "summary");
        assert_eq!(lines[0]["view"], "remote");
        assert_eq!(lines[0]["observer_host_id"], "host-b");
        assert_eq!(lines[0]["subject_host_id"], "host-a");
        assert_eq!(lines[0]["status"], "failed");
        assert_eq!(lines[0]["expected_record_count"], 1);
        assert_eq!(lines[0]["actual_record_count"], 1);
        assert_eq!(
            lines[0]["expected_digest_sha256"],
            lines[0]["server_digest_sha256"]
        );
        assert_ne!(
            lines[0]["expected_digest_sha256"],
            lines[0]["actual_digest_sha256"]
        );
        assert_eq!(lines[1]["line"], "record");
        assert_eq!(lines[1]["input_tokens"], 99);
    }
}
