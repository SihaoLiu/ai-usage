use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::data::{SourceUsageRecord, TokenUsage, UNKNOWN_FAST_TIER, UsageCost, UsageEntry};
use crate::model_id::{Vendor, is_reasoning_effort, parse_model_identity};
use crate::time_utils::parse_timestamp;

mod fast_tier;
mod index;
mod persistence;
mod scan;

use persistence::*;
pub(crate) use scan::{VisitCachedRecordsError, try_for_each_vendor_persisted_record};

#[cfg(test)]
mod tests;
#[cfg(test)]
thread_local! {
    static CACHED_RECORD_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static REMOTE_RECORD_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static INDEXED_CACHE_BYTES_READ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static INDEX_FULL_VALIDATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static HOT_SNAPSHOT_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const CACHE_VERSION: u32 = 1;
const ENTRY_FILE_MAGIC: &[u8; 8] = b"AIUCACH1";
const REMOTE_FILE_MAGIC: &[u8; 8] = b"AIUREMT1";
const ENTRY_INDEX_MAGIC: &[u8; 8] = b"AIUIDX01";
const REMOTE_INDEX_MAGIC: &[u8; 8] = b"AIURIDX1";
const HOT_SNAPSHOT_FILE: &str = "hot-snapshot.bin";
const HOT_SNAPSHOT_MAGIC: &[u8; 8] = b"AIUHOT01";
const MANIFEST_FILE: &str = "manifest.json";
const ENTRIES_DIR: &str = "entries";
const REMOTE_DIR: &str = "remote";
const SESSION_ID_PARSER_REVISION: u32 = 1;
const CLAUDE_PARSER_REVISION: u32 = 2;
const OMP_PARSER_REVISION: u32 = 3;

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
    /// Parser revision after the current active source set was parsed.
    /// Retained inactive records are intentionally excluded: their source
    /// files no longer exist and cannot be refreshed.
    #[serde(default, rename = "session_metadata_revision")]
    parser_revision: u32,
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
        "claude" => CLAUDE_PARSER_REVISION,
        "codex" | "gemini" | "kimi" => SESSION_ID_PARSER_REVISION,
        _ => 0,
    }
}

/// Whether a nonempty active-source manifest uses the current parser revision.
pub(crate) fn vendor_parser_revision_is_current(cache_root: &Path, vendor: &str) -> bool {
    let manifest = read_manifest(&cache_root.join(MANIFEST_FILE));
    manifest_vendor_parser_revision_is_current(&manifest, vendor)
}

pub(crate) fn local_parser_revisions_are_current(cache_root: &Path) -> bool {
    let manifest = read_manifest(&cache_root.join(MANIFEST_FILE));
    ["claude", "codex", "gemini", "kimi", "omp"]
        .into_iter()
        .all(|vendor| {
            !vendor_entries_path(cache_root, vendor).exists()
                || manifest_vendor_parser_revision_is_current(&manifest, vendor)
        })
}

fn manifest_vendor_parser_revision_is_current(manifest: &CacheManifest, vendor: &str) -> bool {
    let required_revision = parser_revision(vendor);
    required_revision == 0
        || manifest.vendors.get(vendor).is_some_and(|vendor_manifest| {
            vendor_manifest_uses_parser_revision(vendor_manifest, required_revision)
        })
}

fn vendor_manifest_uses_parser_revision(
    vendor_manifest: &VendorManifest,
    required_revision: u32,
) -> bool {
    vendor_manifest.parser_revision == required_revision
        && vendor_manifest
            .files
            .values()
            .all(|meta| meta.parser_revision == required_revision)
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
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedVendorRecords {
    format_version: u32,
    records: Vec<PersistedSourceRecord>,
}

/// Cache record layout before conversation ids were retained. Kept only for
/// an in-place migration of existing binary caches.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedVendorRecordsBeforeSession {
    format_version: u32,
    records: Vec<PersistedSourceRecordBeforeSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSourceRecordBeforeSession {
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

impl index::IndexableRecord for PersistedSourceRecord {
    fn index_timestamp(&self) -> &str {
        &self.timestamp
    }

    fn index_dedup_key(&self) -> Option<&str> {
        (!self.dedup_key.is_empty()).then_some(self.dedup_key.as_str())
    }
}

impl index::IndexableRecord for PersistedRemoteRecord {
    fn index_timestamp(&self) -> &str {
        &self.timestamp
    }

    fn append_index_context(&self, digests: &mut Vec<[u8; 32]>) {
        if self.vendor != "omp" || parse_omp_v220_key(&self.dedup_key).is_some() {
            return;
        }
        digests.push(omp_stable_key_digest(&self.dedup_key));
        if self.dedup_key.starts_with("omp:file:") {
            digests.push(omp_file_alias_digest(&remote_omp_file_alias(self)));
        }
    }

    fn index_duplicate_context(&self) -> Option<[u8; 32]> {
        if self.vendor != "omp" {
            return None;
        }
        let key = parse_omp_v220_key(&self.dedup_key)?;
        if !omp_v220_key_matches_remote_record(&key, self) {
            return None;
        }
        Some(
            omp_stable_key_from_v220_key(&key)
                .map(|stable_key| omp_stable_key_digest(&stable_key))
                .unwrap_or_else(|| omp_file_alias_digest(&remote_omp_file_alias_from_key(&key))),
        )
    }
}

impl PersistedRemoteRecord {
    fn collect_omp_stable_aliases(&self, aliases: &mut OmpStableAliases) {
        if self.vendor != "omp" {
            return;
        }
        if parse_omp_v220_key(&self.dedup_key).is_none() {
            aliases.keys.insert(self.dedup_key.clone());
        }
        if self.dedup_key.starts_with("omp:file:") {
            aliases.file_aliases.insert(remote_omp_file_alias(self));
        }
    }
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

#[derive(Debug, Deserialize)]
struct OmpV220Key {
    #[serde(rename = "message")]
    message_id: String,
    #[serde(rename = "response")]
    response_id: String,
    model: String,
    #[serde(rename = "input")]
    input_tokens: i64,
    #[serde(rename = "output")]
    output_tokens: i64,
    #[serde(rename = "cache_read")]
    cache_read_input_tokens: i64,
    #[serde(rename = "cache_write")]
    cache_creation_input_tokens: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OmpRemoteFileAlias {
    model: String,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
}

#[derive(Default)]
struct OmpStableAliases {
    keys: HashSet<String>,
    file_aliases: HashSet<OmpRemoteFileAlias>,
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
            session_id: None,
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
            session_id: None,
        }
    }
}

impl From<PersistedSourceRecordBeforeSession> for PersistedSourceRecord {
    fn from(record: PersistedSourceRecordBeforeSession) -> Self {
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
            cost_input: record.cost_input,
            cost_output: record.cost_output,
            cost_cache_read: record.cost_cache_read,
            cost_cache_creation: record.cost_cache_creation,
            session_id: None,
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
            session_id: record.entry.session_id,
        }
    }

    #[cfg(test)]
    fn to_usage_entry(&self) -> UsageEntry {
        UsageEntry {
            host_id: None,
            session_id: self.session_id.clone(),
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

    fn into_usage_entry(self) -> UsageEntry {
        let parsed_timestamp = parse_timestamp(&self.timestamp);
        UsageEntry {
            host_id: None,
            session_id: self.session_id,
            timestamp: self.timestamp,
            parsed_timestamp,
            session_start_time: self.session_start_time,
            session_end_time: self.session_end_time,
            model: self.model,
            effort: self.effort,
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

    fn into_cached_usage_record(mut self, vendor: &str) -> CachedUsageRecord {
        let source_path = std::mem::take(&mut self.source_path);
        let dedup_key = std::mem::take(&mut self.dedup_key);
        CachedUsageRecord {
            vendor: vendor.to_string(),
            source_path,
            dedup_key,
            entry: self.into_usage_entry(),
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

    fn into_remote_usage_record(self, host_id: &str) -> RemoteUsageRecord {
        let parsed_timestamp = parse_timestamp(&self.timestamp);
        RemoteUsageRecord {
            vendor: self.vendor,
            dedup_key: self.dedup_key,
            entry: UsageEntry {
                host_id: Some(host_id.to_string()),
                session_id: None,
                timestamp: self.timestamp,
                parsed_timestamp,
                session_start_time: self.session_start_time,
                session_end_time: self.session_end_time,
                model: self.model,
                effort: self.effort,
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

/// Read the compact, derived raw-data snapshot without touching source logs.
/// Its payload is versioned by the caller's type and protected by the same
/// checksum and atomic-write protocol as the authoritative record caches.
pub(crate) fn load_hot_snapshot<T: DeserializeOwned>(cache_root: &Path) -> io::Result<Option<T>> {
    #[cfg(test)]
    HOT_SNAPSHOT_READS.set(HOT_SNAPSHOT_READS.get() + 1);
    let path = cache_root.join(HOT_SNAPSHOT_FILE);
    match deserialize_framed(&path, HOT_SNAPSHOT_MAGIC) {
        Ok(snapshot) => Ok(Some(snapshot)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
pub(crate) fn reset_hot_snapshot_reads() {
    HOT_SNAPSHOT_READS.set(0);
}

#[cfg(test)]
pub(crate) fn hot_snapshot_reads() -> usize {
    HOT_SNAPSHOT_READS.get()
}

#[cfg(test)]
pub(crate) fn reset_cached_record_reads() {
    CACHED_RECORD_READS.set(0);
}

#[cfg(test)]
pub(crate) fn cached_record_reads() -> usize {
    CACHED_RECORD_READS.get()
}

/// Persist a compact derived snapshot. The canonical per-source cache files
/// remain the source of truth and can always rebuild this file.
pub(crate) fn write_hot_snapshot<T: Serialize>(cache_root: &Path, snapshot: &T) -> io::Result<()> {
    atomic_serialize_framed(
        &cache_root.join(HOT_SNAPSHOT_FILE),
        HOT_SNAPSHOT_MAGIC,
        snapshot,
    )
}

/// Load cached entries for one vendor without touching source files.
#[cfg(test)]
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
                .into_iter()
                .filter_map(|record| {
                    if !record.dedup_key.is_empty() && !seen.insert(record.dedup_key.clone()) {
                        return None;
                    }
                    Some(record.into_cached_usage_record(vendor))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn load_recent_vendor_cached_records(
    cache_root: &Path,
    vendor: &str,
    cutoff: DateTime<Local>,
) -> (Vec<(String, UsageEntry)>, bool) {
    load_vendor_cached_records_in_range(cache_root, vendor, cutoff, far_future())
}

pub(crate) fn load_vendor_cached_records_in_range(
    cache_root: &Path,
    vendor: &str,
    start: DateTime<Local>,
    end: DateTime<Local>,
) -> (Vec<(String, UsageEntry)>, bool) {
    let path = vendor_entries_path(cache_root, vendor);
    let loaded = read_cached_records_in_range(&path, start, end);
    match loaded {
        Ok((records, has_cached_records)) => {
            let mut seen = HashSet::new();
            let recent = records
                .into_iter()
                .filter_map(|record| {
                    let timestamp = parse_timestamp(&record.timestamp)?;
                    if timestamp < start
                        || timestamp > end
                        || (!record.dedup_key.is_empty() && !seen.insert(record.dedup_key.clone()))
                    {
                        return None;
                    }
                    let dedup_key = record.dedup_key.clone();
                    Some((dedup_key, record.into_usage_entry()))
                })
                .collect();
            (recent, has_cached_records)
        }
        Err(error) => {
            if error.kind() == io::ErrorKind::InvalidData {
                let _ = fs::remove_file(path);
            }
            (Vec::new(), false)
        }
    }
}

fn read_cached_records_in_range(
    path: &Path,
    start: DateTime<Local>,
    end: DateTime<Local>,
) -> io::Result<(Vec<PersistedSourceRecord>, bool)> {
    if let Ok(indexed) = index::read_range::<PersistedSourceRecord>(
        path,
        ENTRY_FILE_MAGIC,
        ENTRY_INDEX_MAGIC,
        start,
        end,
    ) {
        if indexed
            .records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cache entry has negative token count",
            ));
        }
        return Ok((indexed.records, indexed.has_records));
    }
    let records = read_cached_records(path)?;
    let has_records = !records.is_empty();
    let _ = index::ensure(
        path,
        ENTRY_FILE_MAGIC,
        ENTRY_INDEX_MAGIC,
        CACHE_VERSION,
        &records,
    );
    Ok((
        records
            .into_iter()
            .filter(|record| {
                parse_timestamp(&record.timestamp)
                    .is_some_and(|timestamp| timestamp >= start && timestamp <= end)
            })
            .collect(),
        has_records,
    ))
}

pub fn load_remote_entries(
    cache_root: &Path,
    hosts_filter: Option<&HashSet<String>>,
) -> Vec<RemoteUsageRecord> {
    let host_files = remote_host_files(cache_root, hosts_filter);

    // Decode and migrate one host at a time so old multi-gigabyte caches do
    // not multiply their peak memory during the one-time index build.
    let per_host: Vec<Vec<RemoteUsageRecord>> = host_files
        .iter()
        .map(|(host_id, path)| {
            let Ok(host_records) = read_remote_records(path) else {
                return Vec::new();
            };
            let _ = index::ensure(
                path,
                REMOTE_FILE_MAGIC,
                REMOTE_INDEX_MAGIC,
                CACHE_VERSION,
                &host_records,
            );
            let host_records = deduplicate_remote_omp_aliases(host_records);
            host_records
                .into_iter()
                .map(|record| record.into_remote_usage_record(host_id))
                .collect()
        })
        .collect();
    per_host.into_iter().flatten().collect()
}

#[cfg(test)]
pub(crate) fn load_recent_remote_entries(
    cache_root: &Path,
    hosts_filter: Option<&HashSet<String>>,
    cutoff: DateTime<Local>,
) -> Vec<RemoteUsageRecord> {
    load_remote_entries_in_range(cache_root, hosts_filter, cutoff, far_future()).0
}

pub(crate) fn load_remote_entries_in_range(
    cache_root: &Path,
    hosts_filter: Option<&HashSet<String>>,
    start: DateTime<Local>,
    end: DateTime<Local>,
) -> (Vec<RemoteUsageRecord>, bool) {
    let host_files = remote_host_files(cache_root, hosts_filter);
    let mut entries = Vec::new();
    let mut has_source_data = false;
    for (host_id, path) in host_files {
        let (records, host_has_records) = read_remote_records_in_range(&path, start, end)
            .unwrap_or_else(|_| {
                let records = read_remote_records(&path).unwrap_or_default();
                let has_records = !records.is_empty();
                let records = deduplicate_remote_omp_aliases(records)
                    .into_iter()
                    .filter(|record| {
                        parse_timestamp(&record.timestamp)
                            .is_some_and(|timestamp| timestamp >= start && timestamp <= end)
                    })
                    .collect();
                (records, has_records)
            });
        has_source_data |= host_has_records;
        entries.extend(
            records
                .into_iter()
                .map(|record| record.into_remote_usage_record(&host_id)),
        );
    }
    (entries, has_source_data)
}

fn read_remote_records_in_range(
    path: &Path,
    start: DateTime<Local>,
    end: DateTime<Local>,
) -> io::Result<(Vec<PersistedRemoteRecord>, bool)> {
    if let Ok(indexed) = index::read_range::<PersistedRemoteRecord>(
        path,
        REMOTE_FILE_MAGIC,
        REMOTE_INDEX_MAGIC,
        start,
        end,
    ) {
        if indexed
            .records
            .iter()
            .any(|record| !record.has_non_negative_token_usage())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "remote cache entry has negative token count",
            ));
        }
        return Ok((
            deduplicate_remote_omp_aliases(indexed.records),
            indexed.has_records,
        ));
    }
    let records = read_remote_records(path)?;
    let has_records = !records.is_empty();
    let _ = index::ensure(
        path,
        REMOTE_FILE_MAGIC,
        REMOTE_INDEX_MAGIC,
        CACHE_VERSION,
        &records,
    );
    Ok((
        deduplicate_remote_omp_aliases(records)
            .into_iter()
            .filter(|record| {
                parse_timestamp(&record.timestamp)
                    .is_some_and(|timestamp| timestamp >= start && timestamp <= end)
            })
            .collect(),
        has_records,
    ))
}

#[cfg(test)]
fn far_future() -> DateTime<Local> {
    DateTime::from_timestamp(253_402_300_799, 0)
        .expect("year 9999 timestamp")
        .with_timezone(&Local)
}

fn remote_host_files(
    cache_root: &Path,
    hosts_filter: Option<&HashSet<String>>,
) -> Vec<(String, PathBuf)> {
    let remote_root = cache_root.join(REMOTE_DIR);
    let Ok(entries) = fs::read_dir(remote_root) else {
        return Vec::new();
    };
    let mut host_files: Vec<(String, PathBuf)> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if !path.is_file() || path.extension().and_then(|value| value.to_str()) != Some("bin") {
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
    host_files
}

pub(crate) fn remote_host_ids(cache_root: &Path) -> Vec<String> {
    remote_host_files(cache_root, None)
        .into_iter()
        .map(|(host_id, _)| host_id)
        .collect()
}

/// Remove every per-host data file and derived index, reporting the number of
/// host data files removed. Returns Ok(0) if the directory does not exist.
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
            if path.extension().and_then(|value| value.to_str()) == Some("bin") {
                removed += 1;
            }
            fs::remove_file(&path)?;
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
    let mut positions: HashMap<(String, String), usize> = existing
        .iter()
        .enumerate()
        .map(|(idx, record)| ((record.vendor.clone(), record.dedup_key.clone()), idx))
        .collect();

    for record in records {
        let key = (record.vendor.clone(), record.dedup_key.clone());
        let incoming = PersistedRemoteRecord::from_remote_record(record);
        if let Some(idx) = positions.get(&key).copied() {
            refresh_remote_record(&mut existing[idx], incoming);
        } else {
            positions.insert(key, existing.len());
            existing.push(incoming);
        }
    }

    existing = deduplicate_remote_omp_aliases(existing);
    write_remote_records(&path, &existing)
}

fn refresh_remote_record(existing: &mut PersistedRemoteRecord, incoming: PersistedRemoteRecord) {
    *existing = incoming;
}

fn deduplicate_remote_omp_aliases(
    records: Vec<PersistedRemoteRecord>,
) -> Vec<PersistedRemoteRecord> {
    if records.len() < 2 {
        return records;
    }

    let mut stable_aliases = OmpStableAliases::default();
    for record in &records {
        record.collect_omp_stable_aliases(&mut stable_aliases);
    }
    deduplicate_remote_omp_aliases_with_stable_aliases(records, &stable_aliases)
}

fn deduplicate_remote_omp_aliases_with_stable_aliases(
    records: Vec<PersistedRemoteRecord>,
    stable_aliases: &OmpStableAliases,
) -> Vec<PersistedRemoteRecord> {
    let stale_indexes = records
        .iter()
        .enumerate()
        .filter_map(|(idx, record)| {
            if record.vendor != "omp" {
                return None;
            }
            let key = parse_omp_v220_key(&record.dedup_key)?;
            if !omp_v220_key_matches_remote_record(&key, record) {
                return None;
            }
            let duplicate_exists = omp_stable_key_from_v220_key(&key)
                .map(|stable_key| stable_aliases.keys.contains(&stable_key))
                .unwrap_or_else(|| {
                    stable_aliases
                        .file_aliases
                        .contains(&remote_omp_file_alias_from_key(&key))
                });
            duplicate_exists.then_some(idx)
        })
        .collect::<HashSet<_>>();

    if stale_indexes.is_empty() {
        return records;
    }

    records
        .into_iter()
        .enumerate()
        .filter_map(|(idx, record)| (!stale_indexes.contains(&idx)).then_some(record))
        .collect()
}

fn parse_omp_v220_key(dedup_key: &str) -> Option<OmpV220Key> {
    serde_json::from_str(dedup_key).ok()
}

fn omp_v220_key_matches_remote_record(key: &OmpV220Key, record: &PersistedRemoteRecord) -> bool {
    key.input_tokens == record.input_tokens
        && key.output_tokens == record.output_tokens
        && key.cache_read_input_tokens == record.cache_read_input_tokens
        && key.cache_creation_input_tokens == record.cache_creation_input_tokens
        && omp_model_candidates_for(&record.model, record.effort.as_deref())
            .into_iter()
            .any(|model| model == key.model)
}

fn omp_stable_key_from_v220_key(key: &OmpV220Key) -> Option<String> {
    match (key.message_id.is_empty(), key.response_id.is_empty()) {
        (false, false) => Some(format!(
            "omp:message:{}:response:{}",
            key.message_id, key.response_id
        )),
        (false, true) => Some(format!("omp:message:{}", key.message_id)),
        (true, false) => Some(format!("omp:response:{}", key.response_id)),
        (true, true) => None,
    }
}

fn remote_omp_file_alias(record: &PersistedRemoteRecord) -> OmpRemoteFileAlias {
    OmpRemoteFileAlias {
        model: record.model.clone(),
        input_tokens: record.input_tokens,
        output_tokens: record.output_tokens,
        cache_read_input_tokens: record.cache_read_input_tokens,
        cache_creation_input_tokens: record.cache_creation_input_tokens,
    }
}

fn remote_omp_file_alias_from_key(key: &OmpV220Key) -> OmpRemoteFileAlias {
    OmpRemoteFileAlias {
        model: omp_normalized_model(&key.model).to_string(),
        input_tokens: key.input_tokens,
        output_tokens: key.output_tokens,
        cache_read_input_tokens: key.cache_read_input_tokens,
        cache_creation_input_tokens: key.cache_creation_input_tokens,
    }
}

fn omp_stable_key_digest(key: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"omp-stable-key\0");
    hasher.update(key.as_bytes());
    hasher.finalize().into()
}

fn omp_file_alias_digest(alias: &OmpRemoteFileAlias) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"omp-file-alias\0");
    hasher.update((alias.model.len() as u64).to_le_bytes());
    hasher.update(alias.model.as_bytes());
    for value in [
        alias.input_tokens,
        alias.output_tokens,
        alias.cache_read_input_tokens,
        alias.cache_creation_input_tokens,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

fn omp_model_candidates_for(model: &str, effort: Option<&str>) -> Vec<String> {
    let mut models = vec![model.to_string()];
    match parse_model_identity(model).vendor {
        Vendor::Anthropic => {
            models.push(format!("anthropic/{model}"));
            models.push(format!("claude/{model}"));
        }
        Vendor::Google => {
            models.push(format!("gemini/{model}"));
            models.push(format!("google/{model}"));
            models.push(format!("vertex/{model}"));
        }
        Vendor::OpenAI => {
            models.push(format!("openai/{model}"));
            models.push(format!("openai-codex/{model}"));
        }
        _ => {}
    }
    if let Some(provider) = effort.filter(|value| !value.is_empty() && !is_reasoning_effort(value))
    {
        models.push(format!("{provider}/{model}"));
    }
    models.sort();
    models.dedup();
    models
}

fn omp_normalized_model(raw_model: &str) -> &str {
    raw_model
        .split_once('/')
        .and_then(|(_, model)| (!model.is_empty()).then_some(model))
        .unwrap_or(raw_model)
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
    let active_sources = current_sources(source_files);
    let records = load_or_update_vendor_cache_inner(
        cache_root,
        vendor,
        active_sources,
        current_fast_tier,
        parse_file,
        true,
    );
    aggregate_persisted_records(records.iter())
}

#[cfg(test)]
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
    let active_sources = current_sources(source_files);
    let records = load_or_update_vendor_cache_inner(
        cache_root,
        vendor,
        active_sources,
        current_fast_tier,
        parse_file,
        false,
    );
    aggregate_persisted_records(records.iter())
}

pub fn refresh_retaining_vendor_cache<F>(
    cache_root: &Path,
    vendor: &str,
    source_files: Vec<PathBuf>,
    current_fast_tier: i8,
    parse_file: F,
) where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    let active_sources = current_sources(source_files);
    if retaining_vendor_cache_is_current(cache_root, vendor, &active_sources) {
        return;
    }
    let _ = load_or_update_vendor_cache_inner(
        cache_root,
        vendor,
        active_sources,
        current_fast_tier,
        parse_file,
        true,
    );
}

fn retaining_vendor_cache_is_current(
    cache_root: &Path,
    vendor: &str,
    active_sources: &[CurrentSource],
) -> bool {
    let entries_path = vendor_entries_path(cache_root, vendor);
    let manifest_path = cache_root.join(MANIFEST_FILE);
    if !cache_artifact_precedes_manifest(&entries_path, &manifest_path) {
        return false;
    }
    let manifest = read_manifest(&manifest_path);
    let Some(vendor_manifest) = manifest.vendors.get(vendor) else {
        return false;
    };
    let required_revision = parser_revision(vendor);
    index::matches_source_generation(&entries_path, ENTRY_FILE_MAGIC, ENTRY_INDEX_MAGIC)
        && vendor_manifest_uses_parser_revision(vendor_manifest, required_revision)
        && active_sources.iter().all(|source| {
            vendor_manifest.files.get(&source.key).is_some_and(|meta| {
                meta.parser_revision == required_revision && meta.matches_stat(&source.stat)
            })
        })
}

fn cache_artifact_precedes_manifest(entries_path: &Path, manifest_path: &Path) -> bool {
    let Some(entries_stat) = stat_source_file(entries_path) else {
        return false;
    };
    let Some(manifest_stat) = stat_source_file(manifest_path) else {
        return false;
    };
    (entries_stat.modified_secs, entries_stat.modified_nanos)
        < (manifest_stat.modified_secs, manifest_stat.modified_nanos)
}

fn load_or_update_vendor_cache_inner<F>(
    cache_root: &Path,
    vendor: &str,
    active_sources: Vec<CurrentSource>,
    current_fast_tier: i8,
    parse_file: F,
    retain_inactive_sources: bool,
) -> Vec<PersistedSourceRecord>
where
    F: Fn(&Path) -> Vec<SourceUsageRecord> + Sync,
{
    if fs::create_dir_all(cache_root).is_err()
        || fs::create_dir_all(cache_root.join(ENTRIES_DIR)).is_err()
    {
        return parse_active_sources(&active_sources, current_fast_tier, &parse_file);
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
    if parser_revision != 0 && next_vendor_manifest.parser_revision != parser_revision {
        cache_changed = true;
    }

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
            let identity_may_change = next_vendor_manifest
                .files
                .get(&source.key)
                .is_some_and(|meta| meta.parser_revision != parser_revision);
            let mut fast_tiers =
                fast_tier::FastTierMatcher::new(&previous_records, identity_may_change);
            parse_file(&source.path)
                .into_iter()
                .map(|record| {
                    let fast_tier = fast_tiers.take(&record).unwrap_or(current_fast_tier);
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

    if retain_inactive_sources {
        for (source_path, records) in records_by_path {
            if next_vendor_manifest
                .files
                .get(&source_path)
                .is_some_and(|meta| meta.parser_revision == parser_revision)
            {
                active_records.extend(records);
            } else {
                cache_changed = true;
                next_vendor_manifest.files.remove(&source_path);
            }
        }
        let manifest_file_count = next_vendor_manifest.files.len();
        next_vendor_manifest.files.retain(|source_path, meta| {
            active_keys.contains(source_path) || meta.parser_revision == parser_revision
        });
        cache_changed |= next_vendor_manifest.files.len() != manifest_file_count;
    }

    if parser_revision != 0 {
        next_vendor_manifest.parser_revision = parser_revision;
    }
    if !retain_inactive_sources {
        next_vendor_manifest
            .files
            .retain(|source_path, _| active_keys.contains(source_path));
    }
    manifest
        .vendors
        .insert(vendor.to_string(), next_vendor_manifest);
    let rewrite_entries = cache_changed
        || index::ensure(
            &entries_path,
            ENTRY_FILE_MAGIC,
            ENTRY_INDEX_MAGIC,
            CACHE_VERSION,
            &active_records,
        )
        .is_err();
    if rewrite_entries && write_cached_records(&entries_path, &active_records).is_err() {
        return active_records;
    }
    if cache_changed || !cache_artifact_precedes_manifest(&entries_path, &manifest_path) {
        let _ = write_manifest(&manifest_path, &manifest);
    }

    active_records
}

fn records_by_path(
    records: Vec<PersistedSourceRecord>,
) -> BTreeMap<String, Vec<PersistedSourceRecord>> {
    let mut result: BTreeMap<String, Vec<PersistedSourceRecord>> = BTreeMap::new();
    for record in records {
        if let Some(records) = result.get_mut(record.source_path.as_str()) {
            records.push(record);
        } else {
            result.insert(record.source_path.clone(), vec![record]);
        }
    }
    result
}

fn record_stats_by_path(records: &[PersistedSourceRecord]) -> HashMap<String, RecordStats> {
    let mut stats: HashMap<String, RecordStats> = HashMap::new();
    for record in records {
        let entry = if let Some(entry) = stats.get_mut(record.source_path.as_str()) {
            entry
        } else {
            stats.entry(record.source_path.clone()).or_default()
        };
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

#[cfg(test)]
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
