use crate::data::cache::CachedUsageRecord;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub(crate) const FALLBACK_DEDUP_KEY_PREFIX: &str = "fallback:v1:";

#[derive(Debug, Clone)]
pub(crate) struct SyncKeyedRecord {
    pub(crate) record: CachedUsageRecord,
    pub(crate) dedup_key: String,
}

#[derive(Default)]
pub(crate) struct SyncKeyAssigner {
    fallback_counts: HashMap<String, usize>,
}

impl SyncKeyAssigner {
    pub(crate) fn assign(&mut self, record: CachedUsageRecord) -> SyncKeyedRecord {
        let dedup_key = if record.dedup_key.is_empty() {
            let fingerprint = fallback_fingerprint(&record);
            let occurrence = self.fallback_counts.entry(fingerprint.clone()).or_insert(0);
            let dedup_key = format!("{FALLBACK_DEDUP_KEY_PREFIX}{fingerprint}:{occurrence}");
            *occurrence += 1;
            dedup_key
        } else {
            record.dedup_key.clone()
        };
        SyncKeyedRecord { record, dedup_key }
    }
}

pub(crate) fn assign_sync_dedup_keys(records: Vec<CachedUsageRecord>) -> Vec<SyncKeyedRecord> {
    let mut assigner = SyncKeyAssigner::default();
    records
        .into_iter()
        .map(|record| assigner.assign(record))
        .collect()
}

fn fallback_fingerprint(record: &CachedUsageRecord) -> String {
    let mut hasher = Sha256::new();
    let mut usage = record.entry.usage.clone();
    usage.normalize_cache_creation_buckets();
    for value in [
        record.vendor.as_str(),
        record.source_path.as_str(),
        record.entry.timestamp.as_str(),
        record.entry.session_start_time.as_str(),
        record.entry.session_end_time.as_str(),
        record.entry.model.as_str(),
        record.entry.effort.as_deref().unwrap_or(""),
    ] {
        update_string(&mut hasher, value);
    }
    for value in [
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_creation_5m_input_tokens,
        usage.cache_creation_1h_input_tokens,
        usage.reasoning_output_tokens,
    ] {
        hasher.update(value.to_be_bytes());
    }
    hex_digest(hasher.finalize())
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
