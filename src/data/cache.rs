use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageCost, UsageEntry};
use crate::time_utils::parse_timestamp;

const CACHE_VERSION: u32 = 1;
const ENTRY_FILE_MAGIC: &[u8; 8] = b"AIUCACH1";
const REMOTE_FILE_MAGIC: &[u8; 8] = b"AIUREMT1";
const MANIFEST_FILE: &str = "manifest.json";
const ENTRIES_DIR: &str = "entries";
const REMOTE_DIR: &str = "remote";
const OMP_PARSER_REVISION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    version: u32,
    vendors: BTreeMap<String, VendorManifest>,
}

impl Default for CacheManifest {
    fn default() -> Self {
        Self {
            version: CACHE_VERSION,
            vendors: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct VendorManifest {
    files: BTreeMap<String, SourceFileMeta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SourceFileMeta {
    size_bytes: u64,
    modified_secs: u64,
    modified_nanos: u32,
    changed_secs: i64,
    changed_nanos: i64,
    device_id: u64,
    inode: u64,
    record_count: usize,
    records_hash: u64,
    #[serde(default)]
    parser_revision: u32,
}

impl SourceFileMeta {
    fn from_stat(stat: &SourceFileStat, record_stats: RecordStats, parser_revision: u32) -> Self {
        Self {
            size_bytes: stat.size_bytes,
            modified_secs: stat.modified_secs,
            modified_nanos: stat.modified_nanos,
            changed_secs: stat.changed_secs,
            changed_nanos: stat.changed_nanos,
            device_id: stat.device_id,
            inode: stat.inode,
            record_count: record_stats.count,
            records_hash: record_stats.hash,
            parser_revision,
        }
    }

    fn matches_stat(&self, stat: &SourceFileStat) -> bool {
        self.size_bytes == stat.size_bytes
            && self.modified_secs == stat.modified_secs
            && self.modified_nanos == stat.modified_nanos
            && self.changed_secs == stat.changed_secs
            && self.changed_nanos == stat.changed_nanos
            && self.device_id == stat.device_id
            && self.inode == stat.inode
    }
}

fn parser_revision(vendor: &str) -> u32 {
    match vendor {
        "omp" => OMP_PARSER_REVISION,
        _ => 0,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceFileStat {
    size_bytes: u64,
    modified_secs: u64,
    modified_nanos: u32,
    changed_secs: i64,
    changed_nanos: i64,
    device_id: u64,
    inode: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RecordStats {
    count: usize,
    hash: u64,
}

#[derive(Debug, Clone)]
struct CurrentSource {
    key: String,
    path: PathBuf,
    stat: SourceFileStat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSourceRecord {
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
    #[serde(default = "default_fast_tier")]
    fast_tier: i8,
    #[serde(default)]
    cost_input: Option<f64>,
    #[serde(default)]
    cost_output: Option<f64>,
    #[serde(default)]
    cost_cache_read: Option<f64>,
    #[serde(default)]
    cost_cache_creation: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedVendorRecords {
    format_version: u32,
    records: Vec<PersistedSourceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRemoteRecord {
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
    #[serde(default = "default_fast_tier")]
    fast_tier: i8,
    #[serde(default)]
    cost_input: Option<f64>,
    #[serde(default)]
    cost_output: Option<f64>,
    #[serde(default)]
    cost_cache_read: Option<f64>,
    #[serde(default)]
    cost_cache_creation: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRemoteRecords {
    format_version: u32,
    records: Vec<PersistedRemoteRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedVendorRecordsV1 {
    format_version: u32,
    records: Vec<PersistedSourceRecordV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedVendorRecordsWithFastTier {
    format_version: u32,
    records: Vec<PersistedSourceRecordWithFastTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSourceRecordV1 {
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRemoteRecordsV1 {
    format_version: u32,
    records: Vec<PersistedRemoteRecordV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRemoteRecordsWithFastTier {
    format_version: u32,
    records: Vec<PersistedRemoteRecordWithFastTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRemoteRecordV1 {
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone)]
pub struct RemoteUsageRecord {
    pub vendor: String,
    pub dedup_key: String,
    pub entry: UsageEntry,
}

#[derive(Debug, Clone)]
pub struct CachedUsageRecord {
    pub vendor: String,
    pub source_path: String,
    pub dedup_key: String,
    pub entry: UsageEntry,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceRecordFingerprint {
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceRecordFingerprintWithoutEffort {
    dedup_key: String,
    timestamp: String,
    session_start_time: String,
    session_end_time: String,
    model: String,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    reasoning_output_tokens: i64,
}

fn default_fast_tier() -> i8 {
    UNKNOWN_FAST_TIER
}

fn persisted_costs(
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

impl From<PersistedSourceRecordV1> for PersistedSourceRecord {
    fn from(record: PersistedSourceRecordV1) -> Self {
        Self {
            source_path: record.source_path,
            dedup_key: record.dedup_key,
            timestamp: record.timestamp,
            session_start_time: record.session_start_time,
            session_end_time: record.session_end_time,
            model: record.model,
            effort: record.effort,
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            cache_read_input_tokens: record.cache_read_input_tokens,
            cache_creation_input_tokens: record.cache_creation_input_tokens,
            reasoning_output_tokens: record.reasoning_output_tokens,
            fast_tier: UNKNOWN_FAST_TIER,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
        }
    }
}

impl From<PersistedSourceRecordWithFastTier> for PersistedSourceRecord {
    fn from(record: PersistedSourceRecordWithFastTier) -> Self {
        Self {
            source_path: record.source_path,
            dedup_key: record.dedup_key,
            timestamp: record.timestamp,
            session_start_time: record.session_start_time,
            session_end_time: record.session_end_time,
            model: record.model,
            effort: record.effort,
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            cache_read_input_tokens: record.cache_read_input_tokens,
            cache_creation_input_tokens: record.cache_creation_input_tokens,
            reasoning_output_tokens: record.reasoning_output_tokens,
            fast_tier: record.fast_tier,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
        }
    }
}

impl From<PersistedRemoteRecordV1> for PersistedRemoteRecord {
    fn from(record: PersistedRemoteRecordV1) -> Self {
        Self {
            vendor: record.vendor,
            dedup_key: record.dedup_key,
            timestamp: record.timestamp,
            session_start_time: record.session_start_time,
            session_end_time: record.session_end_time,
            model: record.model,
            effort: record.effort,
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            cache_read_input_tokens: record.cache_read_input_tokens,
            cache_creation_input_tokens: record.cache_creation_input_tokens,
            reasoning_output_tokens: record.reasoning_output_tokens,
            fast_tier: UNKNOWN_FAST_TIER,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
        }
    }
}

impl From<PersistedRemoteRecordWithFastTier> for PersistedRemoteRecord {
    fn from(record: PersistedRemoteRecordWithFastTier) -> Self {
        Self {
            vendor: record.vendor,
            dedup_key: record.dedup_key,
            timestamp: record.timestamp,
            session_start_time: record.session_start_time,
            session_end_time: record.session_end_time,
            model: record.model,
            effort: record.effort,
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            cache_read_input_tokens: record.cache_read_input_tokens,
            cache_creation_input_tokens: record.cache_creation_input_tokens,
            reasoning_output_tokens: record.reasoning_output_tokens,
            fast_tier: record.fast_tier,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
        }
    }
}

impl PersistedSourceRecord {
    fn from_source_record(source_path: String, record: SourceUsageRecord, fast_tier: i8) -> Self {
        Self {
            source_path,
            dedup_key: record.dedup_key,
            timestamp: record.entry.timestamp,
            session_start_time: record.entry.session_start_time,
            session_end_time: record.entry.session_end_time,
            model: record.entry.model,
            effort: record.entry.effort,
            input_tokens: record.entry.usage.input_tokens,
            output_tokens: record.entry.usage.output_tokens,
            cache_read_input_tokens: record.entry.usage.cache_read_input_tokens,
            cache_creation_input_tokens: record.entry.usage.cache_creation_input_tokens,
            reasoning_output_tokens: record.entry.usage.reasoning_output_tokens,
            fast_tier,
            cost_input: record.entry.costs.map(|costs| costs.input),
            cost_output: record.entry.costs.map(|costs| costs.output),
            cost_cache_read: record.entry.costs.map(|costs| costs.cache_read),
            cost_cache_creation: record.entry.costs.map(|costs| costs.cache_creation),
        }
    }

    fn to_usage_entry(&self) -> UsageEntry {
        UsageEntry {
            host_id: None,
            timestamp: self.timestamp.clone(),
            parsed_timestamp: parse_timestamp(&self.timestamp),
            session_start_time: self.session_start_time.clone(),
            session_end_time: self.session_end_time.clone(),
            model: self.model.clone(),
            effort: self.effort.clone(),
            fast_tier: self.fast_tier,
            usage: TokenUsage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                cache_read_input_tokens: self.cache_read_input_tokens,
                cache_creation_input_tokens: self.cache_creation_input_tokens,
                reasoning_output_tokens: self.reasoning_output_tokens,
            },
            costs: persisted_costs(
                self.cost_input,
                self.cost_output,
                self.cost_cache_read,
                self.cost_cache_creation,
            ),
        }
    }

    fn has_non_negative_token_usage(&self) -> bool {
        token_counts_are_non_negative([
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
            self.reasoning_output_tokens,
        ])
    }
}

impl PersistedRemoteRecord {
    fn from_remote_record(record: RemoteUsageRecord) -> Self {
        Self {
            vendor: record.vendor,
            dedup_key: record.dedup_key,
            timestamp: record.entry.timestamp,
            session_start_time: record.entry.session_start_time,
            session_end_time: record.entry.session_end_time,
            model: record.entry.model,
            effort: record.entry.effort,
            input_tokens: record.entry.usage.input_tokens,
            output_tokens: record.entry.usage.output_tokens,
            cache_read_input_tokens: record.entry.usage.cache_read_input_tokens,
            cache_creation_input_tokens: record.entry.usage.cache_creation_input_tokens,
            reasoning_output_tokens: record.entry.usage.reasoning_output_tokens,
            fast_tier: record.entry.fast_tier,
            cost_input: record.entry.costs.map(|costs| costs.input),
            cost_output: record.entry.costs.map(|costs| costs.output),
            cost_cache_read: record.entry.costs.map(|costs| costs.cache_read),
            cost_cache_creation: record.entry.costs.map(|costs| costs.cache_creation),
        }
    }

    fn to_remote_usage_record(&self, host_id: &str) -> RemoteUsageRecord {
        RemoteUsageRecord {
            vendor: self.vendor.clone(),
            dedup_key: self.dedup_key.clone(),
            entry: UsageEntry {
                host_id: Some(host_id.to_string()),
                timestamp: self.timestamp.clone(),
                parsed_timestamp: parse_timestamp(&self.timestamp),
                session_start_time: self.session_start_time.clone(),
                session_end_time: self.session_end_time.clone(),
                model: self.model.clone(),
                effort: self.effort.clone(),
                fast_tier: self.fast_tier,
                usage: TokenUsage {
                    input_tokens: self.input_tokens,
                    output_tokens: self.output_tokens,
                    cache_read_input_tokens: self.cache_read_input_tokens,
                    cache_creation_input_tokens: self.cache_creation_input_tokens,
                    reasoning_output_tokens: self.reasoning_output_tokens,
                },
                costs: persisted_costs(
                    self.cost_input,
                    self.cost_output,
                    self.cost_cache_read,
                    self.cost_cache_creation,
                ),
            },
        }
    }

    fn has_non_negative_token_usage(&self) -> bool {
        token_counts_are_non_negative([
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
            self.reasoning_output_tokens,
        ])
    }
}

fn token_counts_are_non_negative(values: [i64; 5]) -> bool {
    values.into_iter().all(|value| value >= 0)
}

/// Return the persistent cache directory used by the CLI.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AI_USAGE_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("ai-usage");
    }
    std::env::var("HOME")
        .map(|home| PathBuf::from(home).join(".cache").join("ai-usage"))
        .unwrap_or_else(|_| PathBuf::from(".cache").join("ai-usage"))
}

/// Load cached entries for one vendor without touching source files.
pub fn load_vendor_cached_snapshot(cache_root: &Path, vendor: &str) -> Vec<UsageEntry> {
    read_cached_records(&vendor_entries_path(cache_root, vendor))
        .map(|records| aggregate_persisted_records(records.iter()))
        .unwrap_or_default()
}

pub fn load_vendor_cached_records(cache_root: &Path, vendor: &str) -> Vec<CachedUsageRecord> {
    read_cached_records(&vendor_entries_path(cache_root, vendor))
        .map(|records| {
            let mut seen = HashSet::new();
            records
                .iter()
                .filter_map(|record| {
                    if !record.dedup_key.is_empty() && !seen.insert(record.dedup_key.clone()) {
                        return None;
                    }
                    Some(CachedUsageRecord {
                        vendor: vendor.to_string(),
                        source_path: record.source_path.clone(),
                        dedup_key: record.dedup_key.clone(),
                        entry: record.to_usage_entry(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn load_remote_entries(
    cache_root: &Path,
    hosts_filter: Option<&HashSet<String>>,
) -> Vec<RemoteUsageRecord> {
    let remote_root = cache_root.join(REMOTE_DIR);
    let Ok(entries) = fs::read_dir(remote_root) else {
        return Vec::new();
    };

    let mut host_files: Vec<(String, PathBuf)> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let host_id = path.file_stem()?.to_str()?.to_string();
            if hosts_filter.is_some_and(|hosts| !hosts.contains(&host_id)) {
                return None;
            }
            Some((host_id, path))
        })
        .collect();
    host_files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut records = Vec::new();
    for (host_id, path) in host_files {
        let Ok(host_records) = read_remote_records(&path) else {
            continue;
        };
        records.extend(
            host_records
                .iter()
                .map(|record| record.to_remote_usage_record(&host_id)),
        );
    }
    records
}

/// Remove every per-host file in the remote cache directory and report
/// how many files were deleted. Returns Ok(0) if the directory does
/// not exist.
pub fn clear_remote_cache(cache_root: &Path) -> io::Result<usize> {
    let remote_root = cache_root.join(REMOTE_DIR);
    let entries = match fs::read_dir(&remote_root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(err),
    };
    let mut removed = 0usize;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            fs::remove_file(&path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

pub fn merge_remote_records(
    cache_root: &Path,
    host_id: &str,
    records: Vec<RemoteUsageRecord>,
) -> io::Result<()> {
    let path = remote_entries_path(cache_root, host_id);
    let mut existing = read_remote_records(&path).unwrap_or_default();
    let mut seen: HashSet<(String, String)> = existing
        .iter()
        .map(|record| (record.vendor.clone(), record.dedup_key.clone()))
        .collect();

    for record in records {
        let key = (record.vendor.clone(), record.dedup_key.clone());
        if seen.insert(key) {
            existing.push(PersistedRemoteRecord::from_remote_record(record));
        }
    }

    write_remote_records(&path, &existing)
}

/// Load normalized entries for one vendor, parsing only files whose metadata changed.
#[cfg(test)]
pub fn load_or_update_vendor_cache<F>(
    cache_root: &Path,
    vendor: &str,
    source_files: Vec<PathBuf>,
    current_fast_tier: i8,
    parse_file: F,
) -> Vec<UsageEntry>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    load_or_update_vendor_cache_inner(
        cache_root,
        vendor,
        source_files,
        current_fast_tier,
        parse_file,
        true,
    )
}

pub fn refresh_full_vendor_cache<F>(
    cache_root: &Path,
    vendor: &str,
    source_files: Vec<PathBuf>,
    current_fast_tier: i8,
    parse_file: F,
) -> Vec<UsageEntry>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    load_or_update_vendor_cache_inner(
        cache_root,
        vendor,
        source_files,
        current_fast_tier,
        parse_file,
        false,
    )
}

fn load_or_update_vendor_cache_inner<F>(
    cache_root: &Path,
    vendor: &str,
    source_files: Vec<PathBuf>,
    current_fast_tier: i8,
    parse_file: F,
    retain_inactive_sources: bool,
) -> Vec<UsageEntry>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    let active_sources = current_sources(source_files);
    if fs::create_dir_all(cache_root).is_err()
        || fs::create_dir_all(cache_root.join(ENTRIES_DIR)).is_err()
    {
        let records = parse_active_sources(&active_sources, current_fast_tier, &parse_file);
        return aggregate_persisted_records(records.iter());
    }

    let manifest_path = cache_root.join(MANIFEST_FILE);
    let entries_path = vendor_entries_path(cache_root, vendor);
    let mut manifest = read_manifest(&manifest_path);
    let vendor_manifest = manifest.vendors.remove(vendor).unwrap_or_default();
    let cached_records = match read_cached_records(&entries_path) {
        Ok(records) => records,
        Err(_) => {
            return rebuild_vendor_cache(
                cache_root,
                vendor,
                &manifest_path,
                manifest,
                active_sources,
                current_fast_tier,
                &parse_file,
            );
        }
    };

    let cached_stats = record_stats_by_path(&cached_records);
    let mut records_by_path = records_by_path(cached_records);
    let mut next_vendor_manifest = vendor_manifest;
    let mut active_records: Vec<PersistedSourceRecord> = Vec::new();
    let mut active_keys = HashSet::new();
    let mut cache_changed = false;
    let parser_revision = parser_revision(vendor);

    for source in &active_sources {
        active_keys.insert(source.key.clone());
        let source_record_stats = cached_stats.get(&source.key).copied().unwrap_or_default();
        let reusable = next_vendor_manifest
            .files
            .get(&source.key)
            .map(|meta| {
                meta.matches_stat(&source.stat)
                    && meta.record_count == source_record_stats.count
                    && meta.records_hash == source_record_stats.hash
                    && meta.parser_revision == parser_revision
            })
            .unwrap_or(false);

        let source_records = if reusable {
            records_by_path.remove(&source.key).unwrap_or_default()
        } else {
            cache_changed = true;
            let previous_records = records_by_path.remove(&source.key).unwrap_or_default();
            let mut fast_tiers = fast_tiers_by_fingerprint(&previous_records);
            let mut fast_tiers_without_effort =
                fast_tiers_by_fingerprint_without_effort(&previous_records);
            parse_file(&source.path)
                .into_iter()
                .map(|record| {
                    let fingerprint = source_record_fingerprint(&record);
                    let fingerprint_without_effort =
                        source_record_fingerprint_without_effort(&record);
                    let fast_tier = fast_tiers
                        .get_mut(&fingerprint)
                        .and_then(|tiers| tiers.pop_front())
                        .or_else(|| {
                            fast_tiers_without_effort
                                .get_mut(&fingerprint_without_effort)
                                .and_then(|tiers| tiers.pop_front())
                        })
                        .unwrap_or(current_fast_tier);
                    PersistedSourceRecord::from_source_record(source.key.clone(), record, fast_tier)
                })
                .collect()
        };

        next_vendor_manifest.files.insert(
            source.key.clone(),
            SourceFileMeta::from_stat(
                &source.stat,
                record_stats(source_records.iter()),
                parser_revision,
            ),
        );
        active_records.extend(source_records);
    }

    if !retain_inactive_sources {
        let has_inactive_records = records_by_path
            .keys()
            .any(|source_path| !active_keys.contains(source_path));
        let has_inactive_manifest_entries = next_vendor_manifest
            .files
            .keys()
            .any(|source_path| !active_keys.contains(source_path));
        cache_changed |= has_inactive_records || has_inactive_manifest_entries;
    }

    if cache_changed {
        if !retain_inactive_sources {
            next_vendor_manifest
                .files
                .retain(|source_path, _| active_keys.contains(source_path));
        }
        let mut records_to_write: Vec<PersistedSourceRecord> = Vec::new();
        for (source_path, records) in records_by_path {
            if retain_inactive_sources && !active_keys.contains(&source_path) {
                records_to_write.extend(records);
            }
        }
        records_to_write.extend(active_records.iter().cloned());

        manifest
            .vendors
            .insert(vendor.to_string(), next_vendor_manifest);
        let _ = write_cached_records(&entries_path, &records_to_write);
        let _ = write_manifest(&manifest_path, &manifest);
    }

    aggregate_persisted_records(active_records.iter())
}

fn rebuild_vendor_cache<F>(
    cache_root: &Path,
    vendor: &str,
    manifest_path: &Path,
    mut manifest: CacheManifest,
    active_sources: Vec<CurrentSource>,
    current_fast_tier: i8,
    parse_file: &F,
) -> Vec<UsageEntry>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    let active_records = parse_active_sources(&active_sources, current_fast_tier, parse_file);
    let mut vendor_manifest = VendorManifest::default();
    let stats = record_stats_by_path(&active_records);

    for source in &active_sources {
        vendor_manifest.files.insert(
            source.key.clone(),
            SourceFileMeta::from_stat(
                &source.stat,
                stats.get(&source.key).copied().unwrap_or_default(),
                parser_revision(vendor),
            ),
        );
    }

    manifest.vendors.insert(vendor.to_string(), vendor_manifest);
    let _ = fs::create_dir_all(cache_root.join(ENTRIES_DIR));
    let _ = write_cached_records(&vendor_entries_path(cache_root, vendor), &active_records);
    let _ = write_manifest(manifest_path, &manifest);

    aggregate_persisted_records(active_records.iter())
}

fn parse_active_sources<F>(
    active_sources: &[CurrentSource],
    current_fast_tier: i8,
    parse_file: &F,
) -> Vec<PersistedSourceRecord>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    let per_source: Vec<Vec<PersistedSourceRecord>> = active_sources
        .par_iter()
        .map(|source| {
            parse_file(&source.path)
                .into_iter()
                .map(|record| {
                    PersistedSourceRecord::from_source_record(
                        source.key.clone(),
                        record,
                        current_fast_tier,
                    )
                })
                .collect()
        })
        .collect();
    per_source.into_iter().flatten().collect()
}

fn current_sources(source_files: Vec<PathBuf>) -> Vec<CurrentSource> {
    let mut sources = Vec::new();
    let mut occurrences: HashMap<String, usize> = HashMap::new();
    for path in source_files {
        let Some(stat) = stat_source_file(&path) else {
            continue;
        };
        let base_key = source_path_key(&path);
        let occurrence = occurrences.entry(base_key.clone()).or_insert(0);
        let key = if *occurrence == 0 {
            base_key
        } else {
            format!("{}#{}", base_key, occurrence)
        };
        *occurrence += 1;
        sources.push(CurrentSource { key, path, stat });
    }
    sources
}

fn stat_source_file(path: &Path) -> Option<SourceFileStat> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let duration = modified
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));
    Some(SourceFileStat {
        size_bytes: metadata.len(),
        modified_secs: duration.as_secs(),
        modified_nanos: duration.subsec_nanos(),
        #[cfg(unix)]
        changed_secs: metadata.ctime(),
        #[cfg(not(unix))]
        changed_secs: 0,
        #[cfg(unix)]
        changed_nanos: metadata.ctime_nsec(),
        #[cfg(not(unix))]
        changed_nanos: 0,
        #[cfg(unix)]
        device_id: metadata.dev(),
        #[cfg(not(unix))]
        device_id: 0,
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(not(unix))]
        inode: 0,
    })
}

fn source_path_key(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn vendor_entries_path(cache_root: &Path, vendor: &str) -> PathBuf {
    cache_root
        .join(ENTRIES_DIR)
        .join(format!("{}.bin", safe_file_stem(vendor)))
}

fn remote_entries_path(cache_root: &Path, host_id: &str) -> PathBuf {
    cache_root
        .join(REMOTE_DIR)
        .join(format!("{}.bin", safe_file_stem(host_id)))
}

fn safe_file_stem(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn read_manifest(path: &Path) -> CacheManifest {
    let Ok(content) = fs::read_to_string(path) else {
        return CacheManifest::default();
    };
    let Ok(manifest) = serde_json::from_str::<CacheManifest>(&content) else {
        return CacheManifest::default();
    };
    if manifest.version == CACHE_VERSION {
        manifest
    } else {
        CacheManifest::default()
    }
}

fn write_manifest(path: &Path, manifest: &CacheManifest) -> io::Result<()> {
    let content = serde_json::to_string_pretty(manifest)?;
    atomic_write(path, content.as_bytes())
}

fn read_cached_records(path: &Path) -> io::Result<Vec<PersistedSourceRecord>> {
    let content = fs::read(path)?;
    if content.len() < ENTRY_FILE_MAGIC.len() + 8
        || &content[..ENTRY_FILE_MAGIC.len()] != ENTRY_FILE_MAGIC
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid cache entry header",
        ));
    }
    let checksum_start = ENTRY_FILE_MAGIC.len();
    let payload_start = checksum_start + 8;
    let stored_checksum = u64::from_le_bytes(
        content[checksum_start..payload_start]
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid checksum"))?,
    );
    let payload = &content[payload_start..];
    let actual_checksum = fnv1a_bytes(0, payload);
    if stored_checksum != actual_checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache entry checksum mismatch",
        ));
    }
    if let Ok(decoded) = bincode::deserialize::<PersistedVendorRecords>(payload) {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported cache entry version",
            ));
        }
        if decoded
            .records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            ));
        }
        return Ok(decoded.records);
    }

    if let Ok(decoded) = bincode::deserialize::<PersistedVendorRecordsWithFastTier>(payload) {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported cache entry version",
            ));
        }
        let records: Vec<PersistedSourceRecord> =
            decoded.records.into_iter().map(Into::into).collect();
        if records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            ));
        }
        return Ok(records);
    }

    let decoded: PersistedVendorRecordsV1 = bincode::deserialize(payload)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if decoded.format_version != CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported cache entry version",
        ));
    }
    let records: Vec<PersistedSourceRecord> = decoded.records.into_iter().map(Into::into).collect();
    if records
        .iter()
        .any(|record| !record.has_non_negative_token_usage())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache entry has negative token count",
        ));
    }
    Ok(records)
}

fn write_cached_records(path: &Path, records: &[PersistedSourceRecord]) -> io::Result<()> {
    let payload = bincode::serialize(&PersistedVendorRecords {
        format_version: CACHE_VERSION,
        records: records.to_vec(),
    })
    .map_err(io::Error::other)?;
    let checksum = fnv1a_bytes(0, &payload);
    let mut content = Vec::with_capacity(ENTRY_FILE_MAGIC.len() + 8 + payload.len());
    content.extend_from_slice(ENTRY_FILE_MAGIC);
    content.extend_from_slice(&checksum.to_le_bytes());
    content.extend_from_slice(&payload);
    atomic_write(path, &content)
}

fn read_remote_records(path: &Path) -> io::Result<Vec<PersistedRemoteRecord>> {
    let content = fs::read(path)?;
    if content.len() < REMOTE_FILE_MAGIC.len() + 8
        || &content[..REMOTE_FILE_MAGIC.len()] != REMOTE_FILE_MAGIC
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid remote cache entry header",
        ));
    }
    let checksum_start = REMOTE_FILE_MAGIC.len();
    let payload_start = checksum_start + 8;
    let stored_checksum = u64::from_le_bytes(
        content[checksum_start..payload_start]
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid checksum"))?,
    );
    let payload = &content[payload_start..];
    let actual_checksum = fnv1a_bytes(0, payload);
    if stored_checksum != actual_checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "remote cache entry checksum mismatch",
        ));
    }
    if let Ok(decoded) = bincode::deserialize::<PersistedRemoteRecords>(payload) {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported remote cache entry version",
            ));
        }
        if decoded
            .records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "remote cache entry has negative token count",
            ));
        }
        return Ok(decoded.records);
    }

    if let Ok(decoded) = bincode::deserialize::<PersistedRemoteRecordsWithFastTier>(payload) {
        if decoded.format_version != CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported remote cache entry version",
            ));
        }
        let records: Vec<PersistedRemoteRecord> =
            decoded.records.into_iter().map(Into::into).collect();
        if records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "remote cache entry has negative token count",
            ));
        }
        return Ok(records);
    }

    let decoded: PersistedRemoteRecordsV1 = bincode::deserialize(payload)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if decoded.format_version != CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported remote cache entry version",
        ));
    }
    let records: Vec<PersistedRemoteRecord> = decoded.records.into_iter().map(Into::into).collect();
    if records
        .iter()
        .any(|record| !record.has_non_negative_token_usage())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "remote cache entry has negative token count",
        ));
    }
    Ok(records)
}

fn write_remote_records(path: &Path, records: &[PersistedRemoteRecord]) -> io::Result<()> {
    let payload = bincode::serialize(&PersistedRemoteRecords {
        format_version: CACHE_VERSION,
        records: records.to_vec(),
    })
    .map_err(io::Error::other)?;
    let checksum = fnv1a_bytes(0, &payload);
    let mut content = Vec::with_capacity(REMOTE_FILE_MAGIC.len() + 8 + payload.len());
    content.extend_from_slice(REMOTE_FILE_MAGIC);
    content.extend_from_slice(&checksum.to_le_bytes());
    content.extend_from_slice(&payload);
    atomic_write(path, &content)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_nanos();
    let tmp_path = path.with_extension(format!("tmp-{}", stamp));
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn records_by_path(
    records: Vec<PersistedSourceRecord>,
) -> BTreeMap<String, Vec<PersistedSourceRecord>> {
    let mut result: BTreeMap<String, Vec<PersistedSourceRecord>> = BTreeMap::new();
    for record in records {
        result
            .entry(record.source_path.clone())
            .or_default()
            .push(record);
    }
    result
}

fn fast_tiers_by_fingerprint(
    records: &[PersistedSourceRecord],
) -> HashMap<SourceRecordFingerprint, VecDeque<i8>> {
    let mut tiers = HashMap::new();
    for record in records {
        tiers
            .entry(persisted_record_fingerprint(record))
            .or_insert_with(VecDeque::new)
            .push_back(record.fast_tier);
    }
    tiers
}

fn fast_tiers_by_fingerprint_without_effort(
    records: &[PersistedSourceRecord],
) -> HashMap<SourceRecordFingerprintWithoutEffort, VecDeque<i8>> {
    let mut tiers = HashMap::new();
    for record in records {
        tiers
            .entry(persisted_record_fingerprint_without_effort(record))
            .or_insert_with(VecDeque::new)
            .push_back(record.fast_tier);
    }
    tiers
}

fn source_record_fingerprint(record: &SourceUsageRecord) -> SourceRecordFingerprint {
    SourceRecordFingerprint {
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
    }
}

fn source_record_fingerprint_without_effort(
    record: &SourceUsageRecord,
) -> SourceRecordFingerprintWithoutEffort {
    SourceRecordFingerprintWithoutEffort {
        dedup_key: record.dedup_key.clone(),
        timestamp: record.entry.timestamp.clone(),
        session_start_time: record.entry.session_start_time.clone(),
        session_end_time: record.entry.session_end_time.clone(),
        model: record.entry.model.clone(),
        input_tokens: record.entry.usage.input_tokens,
        output_tokens: record.entry.usage.output_tokens,
        cache_read_input_tokens: record.entry.usage.cache_read_input_tokens,
        cache_creation_input_tokens: record.entry.usage.cache_creation_input_tokens,
        reasoning_output_tokens: record.entry.usage.reasoning_output_tokens,
    }
}

fn persisted_record_fingerprint(record: &PersistedSourceRecord) -> SourceRecordFingerprint {
    SourceRecordFingerprint {
        dedup_key: record.dedup_key.clone(),
        timestamp: record.timestamp.clone(),
        session_start_time: record.session_start_time.clone(),
        session_end_time: record.session_end_time.clone(),
        model: record.model.clone(),
        effort: record.effort.clone(),
        input_tokens: record.input_tokens,
        output_tokens: record.output_tokens,
        cache_read_input_tokens: record.cache_read_input_tokens,
        cache_creation_input_tokens: record.cache_creation_input_tokens,
        reasoning_output_tokens: record.reasoning_output_tokens,
    }
}

fn persisted_record_fingerprint_without_effort(
    record: &PersistedSourceRecord,
) -> SourceRecordFingerprintWithoutEffort {
    SourceRecordFingerprintWithoutEffort {
        dedup_key: record.dedup_key.clone(),
        timestamp: record.timestamp.clone(),
        session_start_time: record.session_start_time.clone(),
        session_end_time: record.session_end_time.clone(),
        model: record.model.clone(),
        input_tokens: record.input_tokens,
        output_tokens: record.output_tokens,
        cache_read_input_tokens: record.cache_read_input_tokens,
        cache_creation_input_tokens: record.cache_creation_input_tokens,
        reasoning_output_tokens: record.reasoning_output_tokens,
    }
}

fn record_stats_by_path(records: &[PersistedSourceRecord]) -> HashMap<String, RecordStats> {
    let mut stats: HashMap<String, RecordStats> = HashMap::new();
    for record in records {
        let entry = stats.entry(record.source_path.clone()).or_default();
        entry.count += 1;
        entry.hash = update_record_hash(entry.hash, record);
    }
    stats
}

fn record_stats<'a>(records: impl IntoIterator<Item = &'a PersistedSourceRecord>) -> RecordStats {
    let mut stats = RecordStats::default();
    for record in records {
        stats.count += 1;
        stats.hash = update_record_hash(stats.hash, record);
    }
    stats
}

fn update_record_hash(hash: u64, record: &PersistedSourceRecord) -> u64 {
    let mut h = hash;
    for field in [
        record.source_path.as_str(),
        record.dedup_key.as_str(),
        record.timestamp.as_str(),
        record.session_start_time.as_str(),
        record.session_end_time.as_str(),
        record.model.as_str(),
        record.effort.as_deref().unwrap_or(""),
    ] {
        h = fnv1a_bytes(h, field.as_bytes());
        h = fnv1a_bytes(h, &[0xff]);
    }
    for value in [
        record.input_tokens,
        record.output_tokens,
        record.cache_read_input_tokens,
        record.cache_creation_input_tokens,
        record.reasoning_output_tokens,
    ] {
        h = fnv1a_bytes(h, &value.to_le_bytes());
    }
    for value in [
        record.cost_input,
        record.cost_output,
        record.cost_cache_read,
        record.cost_cache_creation,
    ] {
        h = fnv1a_bytes(h, &value.unwrap_or(0.0).to_le_bytes());
    }
    h
}

fn fnv1a_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    if hash == 0 {
        hash = 0xcbf29ce484222325;
    }
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn aggregate_persisted_records<'a>(
    records: impl IntoIterator<Item = &'a PersistedSourceRecord>,
) -> Vec<UsageEntry> {
    let mut seen = HashSet::new();
    let mut entries = Vec::new();
    for record in records {
        if !record.dedup_key.is_empty() && !seen.insert(record.dedup_key.clone()) {
            continue;
        }
        entries.push(record.to_usage_entry());
    }
    entries
}

#[cfg(test)]
mod tests {
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
        let dir = std::env::temp_dir().join(format!("vibe-usage-cache-test-{}-{}", name, stamp));
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
    fn local_cache_entries_have_no_host_id() {
        let cache_root = unique_temp_dir("local-host");
        let source = cache_root.join("source.jsonl");
        write_source(&source, "first");

        let entries =
            super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
            });

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].host_id, None);
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
                    10
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

        let first = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
            },
        );
        let second =
            super::load_or_update_vendor_cache(&cache_root, "test", vec![source], 0, |_| {
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
    fn old_omp_manifest_reparses_unchanged_sources_for_provider_fields() {
        let cache_root = unique_temp_dir("omp-provider-refresh");
        let source = cache_root.join("source.jsonl");
        write_source(&source, "first");
        let calls = AtomicUsize::new(0);

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "omp",
            vec![source.clone()],
            -1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                let mut record = usage_record("omp:message:msg-a", "2026-05-01T00:00:00Z", 42);
                record.entry.model = "claude-sonnet-4-5-20250929".to_string();
                record.entry.effort = None;
                vec![record]
            },
        );
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
                record.entry.effort = Some("anthropic".to_string());
                vec![record]
            });

        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(refreshed.len(), 1);
        assert_eq!(refreshed[0].effort.as_deref(), Some("anthropic"));
    }

    #[test]
    fn changed_source_file_replaces_cached_records() {
        let cache_root = unique_temp_dir("changed");
        let source = cache_root.join("source.jsonl");
        write_source(&source, "first");
        let calls = AtomicUsize::new(0);

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
            },
        );
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

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
            },
        );
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

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            -1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
            },
        );
        fs::write(cache_root.join("entries").join("test.bin"), b"not binary")
            .expect("damage cache");

        let rebuilt =
            super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 77)]
            });

        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(rebuilt.len(), 1);
        assert_eq!(rebuilt[0].usage.input_tokens, 77);
    }

    #[test]
    fn valid_but_tampered_entry_cache_triggers_vendor_rebuild() {
        let cache_root = unique_temp_dir("tampered");
        let source = cache_root.join("source.jsonl");
        write_source(&source, "first");
        let calls = AtomicUsize::new(0);

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            -1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)]
            },
        );

        let entries_path = cache_root.join("entries").join("test.bin");
        let mut original = fs::read(&entries_path).expect("read cache");
        let last = original.last_mut().expect("nonempty cache");
        *last = last.wrapping_add(1);
        fs::write(&entries_path, original).expect("tamper cache");

        let rebuilt =
            super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
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

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            -1,
            |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                let mut record = usage_record("stable-key", "2026-05-01T00:00:00Z", 42);
                record.entry.usage.output_tokens = -1;
                vec![record]
            },
        );

        let rebuilt =
            super::load_or_update_vendor_cache(&cache_root, "test", vec![source], -1, |_| {
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
        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            source_files.clone(),
            -1,
            parse,
        );
        let cached =
            super::load_or_update_vendor_cache(&cache_root, "test", source_files, -1, |_| {
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
        let refreshed =
            super::refresh_full_vendor_cache(&cache_root, "test", source_files, -1, parse);
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
        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            source_files.clone(),
            -1,
            parse,
        );
        let cached =
            super::load_or_update_vendor_cache(&cache_root, "test", source_files, -1, |_| {
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

        let _ = super::load_or_update_vendor_cache(
            &cache_root,
            "test",
            vec![source.clone()],
            -1,
            |_| vec![usage_record("stable-key", "2026-05-01T00:00:00Z", 42)],
        );
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
        fs::write(cache_root.join("entries").join("test.bin"), b"not binary")
            .expect("damage cache");

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
}
