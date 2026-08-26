use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

use super::{PersistedSourceRecord, SourceUsageRecord};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RecordFingerprint {
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
    cache_creation_5m_input_tokens: i64,
    cache_creation_1h_input_tokens: i64,
    reasoning_output_tokens: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RecordFingerprintWithoutEffort {
    dedup_key: String,
    timestamp: String,
    session_start_time: String,
    session_end_time: String,
    model: String,
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    cache_creation_5m_input_tokens: i64,
    cache_creation_1h_input_tokens: i64,
    reasoning_output_tokens: i64,
}

pub(super) struct FastTierMatcher {
    fast_tiers: Vec<i8>,
    claimed: Vec<bool>,
    exact: HashMap<RecordFingerprint, VecDeque<usize>>,
    without_effort: HashMap<RecordFingerprintWithoutEffort, VecDeque<usize>>,
    without_identity: Option<HashMap<RecordFingerprint, VecDeque<usize>>>,
}

impl FastTierMatcher {
    pub(super) fn new(records: &[PersistedSourceRecord], identity_may_change: bool) -> Self {
        Self {
            fast_tiers: records.iter().map(|record| record.fast_tier).collect(),
            claimed: vec![false; records.len()],
            exact: tiers_by_fingerprint(records, false),
            without_effort: tiers_by_fingerprint_without_effort(records),
            without_identity: identity_may_change.then(|| tiers_by_fingerprint(records, true)),
        }
    }

    pub(super) fn take(&mut self, record: &SourceUsageRecord) -> Option<i8> {
        let mut fingerprint = source_record_fingerprint(record);
        if let Some(tier) = claim_tier(
            &mut self.exact,
            &fingerprint,
            &self.fast_tiers,
            &mut self.claimed,
        ) {
            return Some(tier);
        }

        let fingerprint_without_effort = source_record_fingerprint_without_effort(record);
        if let Some(tier) = claim_tier(
            &mut self.without_effort,
            &fingerprint_without_effort,
            &self.fast_tiers,
            &mut self.claimed,
        ) {
            return Some(tier);
        }

        fingerprint.dedup_key.clear();
        self.without_identity
            .as_mut()
            .and_then(|index| claim_tier(index, &fingerprint, &self.fast_tiers, &mut self.claimed))
    }
}

fn claim_tier<K: Eq + Hash>(
    index: &mut HashMap<K, VecDeque<usize>>,
    key: &K,
    fast_tiers: &[i8],
    claimed: &mut [bool],
) -> Option<i8> {
    let candidates = index.get_mut(key)?;
    while let Some(record_index) = candidates.pop_front() {
        if !claimed[record_index] {
            claimed[record_index] = true;
            return Some(fast_tiers[record_index]);
        }
    }
    None
}

fn tiers_by_fingerprint(
    records: &[PersistedSourceRecord],
    ignore_identity: bool,
) -> HashMap<RecordFingerprint, VecDeque<usize>> {
    let mut tiers = HashMap::new();
    for (record_index, record) in records.iter().enumerate() {
        let mut fingerprint = persisted_record_fingerprint(record);
        if ignore_identity {
            fingerprint.dedup_key.clear();
        }
        tiers
            .entry(fingerprint)
            .or_insert_with(VecDeque::new)
            .push_back(record_index);
    }
    tiers
}

fn tiers_by_fingerprint_without_effort(
    records: &[PersistedSourceRecord],
) -> HashMap<RecordFingerprintWithoutEffort, VecDeque<usize>> {
    let mut tiers = HashMap::new();
    for (record_index, record) in records.iter().enumerate() {
        tiers
            .entry(persisted_record_fingerprint_without_effort(record))
            .or_insert_with(VecDeque::new)
            .push_back(record_index);
    }
    tiers
}

fn source_record_fingerprint(record: &SourceUsageRecord) -> RecordFingerprint {
    let mut usage = record.entry.usage.clone();
    usage.normalize_cache_creation_buckets();
    RecordFingerprint {
        dedup_key: record.dedup_key.clone(),
        timestamp: record.entry.timestamp.clone(),
        session_start_time: record.entry.session_start_time.clone(),
        session_end_time: record.entry.session_end_time.clone(),
        model: record.entry.model.clone(),
        effort: record.entry.effort.clone(),
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        cache_creation_5m_input_tokens: usage.cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens: usage.cache_creation_1h_input_tokens,
        reasoning_output_tokens: usage.reasoning_output_tokens,
    }
}

fn source_record_fingerprint_without_effort(
    record: &SourceUsageRecord,
) -> RecordFingerprintWithoutEffort {
    let mut usage = record.entry.usage.clone();
    usage.normalize_cache_creation_buckets();
    RecordFingerprintWithoutEffort {
        dedup_key: record.dedup_key.clone(),
        timestamp: record.entry.timestamp.clone(),
        session_start_time: record.entry.session_start_time.clone(),
        session_end_time: record.entry.session_end_time.clone(),
        model: record.entry.model.clone(),
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        cache_creation_5m_input_tokens: usage.cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens: usage.cache_creation_1h_input_tokens,
        reasoning_output_tokens: usage.reasoning_output_tokens,
    }
}

fn persisted_record_fingerprint(record: &PersistedSourceRecord) -> RecordFingerprint {
    RecordFingerprint {
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
        cache_creation_5m_input_tokens: record.cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens: record.cache_creation_1h_input_tokens,
        reasoning_output_tokens: record.reasoning_output_tokens,
    }
}

fn persisted_record_fingerprint_without_effort(
    record: &PersistedSourceRecord,
) -> RecordFingerprintWithoutEffort {
    RecordFingerprintWithoutEffort {
        dedup_key: record.dedup_key.clone(),
        timestamp: record.timestamp.clone(),
        session_start_time: record.session_start_time.clone(),
        session_end_time: record.session_end_time.clone(),
        model: record.model.clone(),
        input_tokens: record.input_tokens,
        output_tokens: record.output_tokens,
        cache_read_input_tokens: record.cache_read_input_tokens,
        cache_creation_input_tokens: record.cache_creation_input_tokens,
        cache_creation_5m_input_tokens: record.cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens: record.cache_creation_1h_input_tokens,
        reasoning_output_tokens: record.reasoning_output_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{TokenUsage, UNKNOWN_FAST_TIER, UsageEntry};

    fn source_record(dedup_key: &str) -> SourceUsageRecord {
        SourceUsageRecord {
            dedup_key: dedup_key.to_string(),
            entry: UsageEntry {
                host_id: None,
                session_id: None,
                timestamp: "2026-05-01T00:00:00Z".to_string(),
                parsed_timestamp: None,
                session_start_time: "2026-05-01T00:00:00Z".to_string(),
                session_end_time: "2026-05-01T00:00:00Z".to_string(),
                model: "test-model".to_string(),
                effort: None,
                fast_tier: UNKNOWN_FAST_TIER,
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cache_read_input_tokens: 3,
                    cache_creation_input_tokens: 4,
                    cache_creation_5m_input_tokens: 0,
                    cache_creation_1h_input_tokens: 0,
                    reasoning_output_tokens: 5,
                },
                costs: None,
            },
        }
    }

    #[test]
    fn unchanged_revision_does_not_match_a_changed_identity() {
        let cached = PersistedSourceRecord::from_source_record(
            "source.jsonl".to_string(),
            source_record("old-key"),
            1,
        );
        let mut matcher = FastTierMatcher::new(&[cached], false);

        assert_eq!(matcher.take(&source_record("new-key")), None);
    }

    #[test]
    fn exact_match_cannot_be_claimed_again_without_effort() {
        let cached = PersistedSourceRecord::from_source_record(
            "source.jsonl".to_string(),
            source_record("stable-key"),
            1,
        );
        let mut matcher = FastTierMatcher::new(&[cached], false);

        assert_eq!(matcher.take(&source_record("stable-key")), Some(1));
        let mut changed_effort = source_record("stable-key");
        changed_effort.entry.effort = Some("high".to_string());
        assert_eq!(matcher.take(&changed_effort), None);
    }
}
