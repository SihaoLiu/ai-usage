use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fmt;

// Duration-specific cache fields are additive, so the existing wire schema
// and integrity algorithm remain compatible with deployed v3 clients.
pub const SCHEMA_VERSION: u32 = 1;
pub const INTEGRITY_ALGORITHM: &str = "usage-record-sha256-v1";

fn default_fast_tier() -> i8 {
    -1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireRecord {
    pub schema_version: u32,
    pub host_id: String,
    pub vendor: String,
    pub dedup_key: String,
    pub timestamp: String,
    pub session_start_time: String,
    pub session_end_time: String,
    pub model: String,
    pub effort: Option<String>,
    #[serde(default = "default_fast_tier")]
    pub fast_tier: i8,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    /// Claude cache writes retained for five minutes.
    #[serde(default)]
    pub cache_creation_5m_input_tokens: i64,
    /// Claude cache writes retained for one hour.
    #[serde(default)]
    pub cache_creation_1h_input_tokens: i64,
    pub reasoning_output_tokens: i64,
    #[serde(default)]
    pub cost_input: Option<f64>,
    #[serde(default)]
    pub cost_output: Option<f64>,
    #[serde(default)]
    pub cost_cache_read: Option<f64>,
    #[serde(default)]
    pub cost_cache_creation: Option<f64>,
    pub project_path_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequencedWireRecord {
    #[serde(flatten)]
    pub record: WireRecord,
    pub seq: u64,
    pub uploaded_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadResponse {
    pub accepted: usize,
    pub ignored: usize,
    pub max_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordKey {
    pub vendor: String,
    pub dedup_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordFingerprint {
    pub vendor: String,
    pub dedup_key: String,
    pub record_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotKeyBatch {
    pub host_id: String,
    pub snapshot_id: String,
    pub keys: Vec<RecordKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotDiffRequest {
    pub host_id: String,
    pub snapshot_id: String,
    pub records: Vec<RecordFingerprint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotDiffResponse {
    pub needed: Vec<RecordKey>,
    pub matched: usize,
    pub missing_or_changed: usize,
    pub max_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotRecordBatch {
    pub host_id: String,
    pub snapshot_id: String,
    pub records: Vec<WireRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotFinalizeRequest {
    pub host_id: String,
    pub snapshot_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotFinalizeResponse {
    pub deleted: usize,
    pub max_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PullResponse {
    pub records: Vec<SequencedWireRecord>,
    pub max_seq: u64,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineInfo {
    pub host_id: String,
    pub last_seen: String,
    pub record_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_revision: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineList {
    pub machines: Vec<MachineInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncPolicy {
    pub min_request_interval_ms: u64,
    pub request_phase_ms: u64,
    pub max_request_interval_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub version: String,
    pub schema_version: u32,
    pub uptime_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_policy: Option<SyncPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub host_id: String,
    pub algorithm: String,
    pub range_end_utc: String,
    pub record_count: u64,
    pub digest_sha256: String,
    pub computed_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegritySubmitResponse {
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrityReportList {
    pub reports: Vec<IntegrityReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    UnsupportedSchemaVersion(u32),
    InvalidHostId,
    InvalidVendor,
    EmptyField(&'static str),
    FieldTooLong(&'static str),
    InvalidTimestamp(&'static str),
    NegativeTokenCount(&'static str),
    InvalidCost(&'static str),
    InvalidProjectHash,
    InvalidIntegrityAlgorithm,
    InvalidDigest,
    InvalidSnapshotId,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported schema version {version}")
            }
            Self::InvalidHostId => f.write_str("invalid host_id"),
            Self::InvalidVendor => f.write_str("invalid vendor"),
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::FieldTooLong(field) => write!(f, "{field} is too long"),
            Self::InvalidTimestamp(field) => write!(f, "{field} must be RFC3339"),
            Self::NegativeTokenCount(field) => write!(f, "{field} must be non-negative"),
            Self::InvalidCost(field) => write!(f, "{field} must be finite and non-negative"),
            Self::InvalidProjectHash => f.write_str("project_path_sha256 must be lowercase hex"),
            Self::InvalidIntegrityAlgorithm => f.write_str("invalid integrity algorithm"),
            Self::InvalidDigest => f.write_str("digest_sha256 must be lowercase hex"),
            Self::InvalidSnapshotId => f.write_str("invalid snapshot_id"),
        }
    }
}

impl std::error::Error for ValidationError {}

impl WireRecord {
    /// Convert a legacy aggregate-only record to the current wire shape.
    /// Duration-specific buckets are authoritative when present; otherwise
    /// the aggregate is retained as a five-minute cache write.
    pub fn normalize_for_current_schema(mut self) -> Self {
        let split_total = self
            .cache_creation_5m_input_tokens
            .saturating_add(self.cache_creation_1h_input_tokens);
        if split_total > 0 {
            self.cache_creation_input_tokens = split_total;
        } else if self.cache_creation_input_tokens > 0 {
            self.cache_creation_5m_input_tokens = self.cache_creation_input_tokens;
        }
        self
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        validate_host_id(&self.host_id)?;
        validate_vendor(&self.vendor)?;
        validate_required_text("dedup_key", &self.dedup_key, 512)?;
        validate_required_timestamp("timestamp", &self.timestamp)?;
        validate_required_timestamp("session_start_time", &self.session_start_time)?;
        validate_required_timestamp("session_end_time", &self.session_end_time)?;
        validate_required_text("model", &self.model, 256)?;
        if let Some(effort) = &self.effort {
            validate_optional_text("effort", effort, 64)?;
        }
        for (field, value) in [
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("cache_read_input_tokens", self.cache_read_input_tokens),
            (
                "cache_creation_input_tokens",
                self.cache_creation_input_tokens,
            ),
            (
                "cache_creation_5m_input_tokens",
                self.cache_creation_5m_input_tokens,
            ),
            (
                "cache_creation_1h_input_tokens",
                self.cache_creation_1h_input_tokens,
            ),
            ("reasoning_output_tokens", self.reasoning_output_tokens),
        ] {
            if value < 0 {
                return Err(ValidationError::NegativeTokenCount(field));
            }
        }
        for (field, value) in [
            ("cost_input", self.cost_input),
            ("cost_output", self.cost_output),
            ("cost_cache_read", self.cost_cache_read),
            ("cost_cache_creation", self.cost_cache_creation),
        ] {
            if let Some(cost) = value
                && (!cost.is_finite() || cost < 0.0)
            {
                return Err(ValidationError::InvalidCost(field));
            }
        }
        if let Some(hash) = &self.project_path_sha256 {
            validate_project_hash(hash)?;
        }
        Ok(())
    }
}

impl SequencedWireRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.record.validate()?;
        validate_required_timestamp("uploaded_at", &self.uploaded_at)
    }
}

impl RecordKey {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_vendor(&self.vendor)?;
        validate_required_text("dedup_key", &self.dedup_key, 512)
    }
}

impl RecordFingerprint {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_vendor(&self.vendor)?;
        validate_required_text("dedup_key", &self.dedup_key, 512)?;
        validate_digest(&self.record_hash)
    }
}

impl SnapshotKeyBatch {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_host_id(&self.host_id)?;
        validate_snapshot_id(&self.snapshot_id)?;
        for key in &self.keys {
            key.validate()?;
        }
        Ok(())
    }
}

impl SnapshotDiffRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_host_id(&self.host_id)?;
        validate_snapshot_id(&self.snapshot_id)?;
        for record in &self.records {
            record.validate()?;
        }
        Ok(())
    }
}

impl SnapshotRecordBatch {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_host_id(&self.host_id)?;
        validate_snapshot_id(&self.snapshot_id)?;
        for record in &self.records {
            record.validate()?;
            if record.host_id != self.host_id {
                return Err(ValidationError::InvalidHostId);
            }
        }
        Ok(())
    }
}

impl SnapshotFinalizeRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_host_id(&self.host_id)?;
        validate_snapshot_id(&self.snapshot_id)
    }
}

impl IntegrityReport {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_host_id(&self.host_id)?;
        if self.algorithm != INTEGRITY_ALGORITHM {
            return Err(ValidationError::InvalidIntegrityAlgorithm);
        }
        validate_required_utc_timestamp("range_end_utc", &self.range_end_utc)?;
        validate_digest(&self.digest_sha256)?;
        validate_required_utc_timestamp("computed_at", &self.computed_at)
    }
}

pub fn is_valid_host_id(host_id: &str) -> bool {
    !host_id.is_empty()
        && host_id.len() <= 64
        && host_id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
}

pub fn is_valid_vendor(vendor: &str) -> bool {
    matches!(vendor, "claude" | "codex" | "gemini" | "kimi" | "omp")
}

pub fn is_globally_stable_usage_key(vendor: &str, dedup_key: &str) -> bool {
    match vendor {
        "claude" => has_nonempty_suffix(dedup_key, "claude:message:"),
        "codex" => is_codex_cumulative_transition_key(dedup_key),
        "gemini" => dedup_key
            .strip_prefix("gemini:")
            .and_then(|suffix| suffix.split_once(':'))
            .is_some_and(|(session_id, message_id)| {
                !session_id.is_empty()
                    && !matches!(session_id, "file" | "msg")
                    && !message_id.is_empty()
            }),
        "kimi" => has_nonempty_suffix(dedup_key, "kimi:response:"),
        "omp" => {
            if let Some(message_id) = dedup_key.strip_prefix("omp:message:") {
                match message_id.split_once(":response:") {
                    Some((message_id, response_id)) => {
                        !message_id.is_empty() && !response_id.is_empty()
                    }
                    None => !message_id.is_empty(),
                }
            } else {
                has_nonempty_suffix(dedup_key, "omp:response:")
            }
        }
        _ => false,
    }
}

pub fn is_valid_codex_usage_snapshot(
    input_tokens: i64,
    cached_input_tokens: i64,
    cache_write_input_tokens: i64,
    output_tokens: i64,
    reasoning_output_tokens: i64,
    total_tokens: i64,
) -> bool {
    if [
        input_tokens,
        cached_input_tokens,
        cache_write_input_tokens,
        output_tokens,
        reasoning_output_tokens,
        total_tokens,
    ]
    .into_iter()
    .any(|value| value < 0)
    {
        return false;
    }

    cached_input_tokens
        .checked_add(cache_write_input_tokens)
        .is_some_and(|cached| cached <= input_tokens)
        && input_tokens.checked_add(output_tokens) == Some(total_tokens)
        && reasoning_output_tokens <= output_tokens
}

fn is_codex_cumulative_transition_key(dedup_key: &str) -> bool {
    let Some((turn_id, transition)) = dedup_key
        .strip_prefix("codex:turn:")
        .and_then(|suffix| suffix.rsplit_once(":cumulative:"))
    else {
        return false;
    };
    let Some((previous, current)) = transition.split_once("->") else {
        return false;
    };

    !turn_id.is_empty()
        && (previous == "start" || is_codex_cumulative_snapshot(previous))
        && is_codex_cumulative_snapshot(current)
}

fn is_codex_cumulative_snapshot(snapshot: &str) -> bool {
    let mut values = snapshot.split(',');
    let mut parsed = [0_i64; 6];
    for value in &mut parsed {
        let Some(next) = values.next().and_then(|value| value.parse::<i64>().ok()) else {
            return false;
        };
        *value = next;
    }
    values.next().is_none()
        && is_valid_codex_usage_snapshot(
            parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5],
        )
}

fn has_nonempty_suffix(value: &str, prefix: &str) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|suffix| !suffix.is_empty())
}

fn validate_host_id(host_id: &str) -> Result<(), ValidationError> {
    if is_valid_host_id(host_id) {
        Ok(())
    } else {
        Err(ValidationError::InvalidHostId)
    }
}

fn validate_vendor(vendor: &str) -> Result<(), ValidationError> {
    if is_valid_vendor(vendor) {
        Ok(())
    } else {
        Err(ValidationError::InvalidVendor)
    }
}

fn validate_required_text(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::EmptyField(field));
    }
    if value.len() > max_len {
        return Err(ValidationError::FieldTooLong(field));
    }
    Ok(())
}

fn validate_optional_text(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), ValidationError> {
    if value.len() > max_len {
        return Err(ValidationError::FieldTooLong(field));
    }
    Ok(())
}

fn validate_required_timestamp(field: &'static str, value: &str) -> Result<(), ValidationError> {
    validate_required_text(field, value, 64)?;
    DateTime::parse_from_rfc3339(value).map_err(|_| ValidationError::InvalidTimestamp(field))?;
    Ok(())
}

fn validate_required_utc_timestamp(
    field: &'static str,
    value: &str,
) -> Result<(), ValidationError> {
    validate_required_text(field, value, 64)?;
    let parsed = DateTime::parse_from_rfc3339(value)
        .map_err(|_| ValidationError::InvalidTimestamp(field))?;
    if parsed.offset().local_minus_utc() == 0 {
        Ok(())
    } else {
        Err(ValidationError::InvalidTimestamp(field))
    }
}

fn validate_project_hash(hash: &str) -> Result<(), ValidationError> {
    if hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ValidationError::InvalidProjectHash)
    }
}

fn validate_snapshot_id(snapshot_id: &str) -> Result<(), ValidationError> {
    if snapshot_id.is_empty()
        || snapshot_id.len() > 128
        || !snapshot_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        Err(ValidationError::InvalidSnapshotId)
    } else {
        Ok(())
    }
}

fn validate_digest(digest: &str) -> Result<(), ValidationError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ValidationError::InvalidDigest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_record() -> WireRecord {
        WireRecord {
            schema_version: SCHEMA_VERSION,
            host_id: "workstation-home".to_string(),
            vendor: "claude".to_string(),
            dedup_key: "source-record-1".to_string(),
            timestamp: "2026-05-18T12:34:56Z".to_string(),
            session_start_time: "2026-05-18T12:30:00Z".to_string(),
            session_end_time: "2026-05-18T12:34:56Z".to_string(),
            model: "claude-sonnet-4".to_string(),
            effort: Some("high".to_string()),
            fast_tier: 1,
            input_tokens: 10,
            output_tokens: 20,
            cache_read_input_tokens: 30,
            cache_creation_input_tokens: 40,
            cache_creation_5m_input_tokens: 40,
            cache_creation_1h_input_tokens: 0,
            reasoning_output_tokens: 50,
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_creation: None,
            project_path_sha256: Some("a".repeat(64)),
        }
    }

    #[test]
    fn accepts_valid_wire_record() {
        assert!(valid_record().validate().is_ok());
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let mut record = valid_record();
        record.schema_version = SCHEMA_VERSION + 1;

        assert!(record.validate().is_err());
    }

    #[test]
    fn rejects_invalid_host_ids() {
        for host_id in [
            "",
            "Workstation",
            "has.dot",
            "has/slash",
            "a".repeat(65).as_str(),
        ] {
            let mut record = valid_record();
            record.host_id = host_id.to_string();

            assert!(record.validate().is_err(), "host_id={host_id:?}");
        }
    }

    #[test]
    fn rejects_unknown_vendors() {
        let mut record = valid_record();
        record.vendor = "unknown".to_string();

        assert!(record.validate().is_err());
    }

    #[test]
    fn accepts_every_tracked_vendor() {
        for vendor in ["claude", "codex", "gemini", "kimi", "omp"] {
            let mut record = valid_record();
            record.vendor = vendor.to_string();

            assert!(record.validate().is_ok(), "vendor={vendor:?}");
        }
    }

    #[test]
    fn globally_stable_usage_keys_match_semantic_identities() {
        let cases = [
            ("claude message", "claude", "claude:message:message-a", true),
            (
                "claude file fallback",
                "claude",
                "claude:file:/workspace/session.jsonl:12",
                false,
            ),
            (
                "obsolete codex single total",
                "codex",
                "codex:turn:turn-a:total:10,2,3,4,5,24",
                false,
            ),
            (
                "codex cumulative transition",
                "codex",
                "codex:turn:turn-a:cumulative:10,2,0,3,1,13->20,4,1,6,2,26",
                true,
            ),
            (
                "first codex cumulative transition",
                "codex",
                "codex:turn:turn-a:cumulative:start->10,2,0,3,1,13",
                true,
            ),
            (
                "gemini session message",
                "gemini",
                "gemini:session-a:message-a",
                true,
            ),
            (
                "gemini message fallback",
                "gemini",
                "gemini:msg:message-a",
                false,
            ),
            ("kimi response", "kimi", "kimi:response:response-a", true),
            ("omp message", "omp", "omp:message:message-a", true),
            (
                "omp message response",
                "omp",
                "omp:message:message-a:response:response-a",
                true,
            ),
            ("omp response", "omp", "omp:response:response-a", true),
            ("empty key", "claude", "", false),
            (
                "vendor prefix mismatch",
                "claude",
                "codex:turn:turn-a:cumulative:start->10,2,0,3,1,13",
                false,
            ),
            (
                "unknown vendor",
                "unknown",
                "claude:message:message-a",
                false,
            ),
            (
                "legacy codex tuple",
                "codex",
                r#"["turn-a",10,2,3,1]"#,
                false,
            ),
            (
                "kimi timestamp ordinal",
                "kimi",
                "kimi:session-a:agent-a:1785000000000:0",
                false,
            ),
            (
                "gemini file fallback",
                "gemini",
                "gemini:file:/sessions/a.json:0",
                false,
            ),
            (
                "kimi file fallback",
                "kimi",
                "kimi:file:/sessions/a.jsonl:0",
                false,
            ),
            (
                "omp file fallback",
                "omp",
                "omp:file:/sessions/a.jsonl:0",
                false,
            ),
            ("empty claude message", "claude", "claude:message:", false),
            (
                "empty codex turn",
                "codex",
                "codex:turn::cumulative:start->10,2,0,3,1,13",
                false,
            ),
            (
                "missing codex previous snapshot",
                "codex",
                "codex:turn:turn-a:cumulative:->10,2,0,3,1,13",
                false,
            ),
            (
                "missing codex current snapshot",
                "codex",
                "codex:turn:turn-a:cumulative:10,2,0,3,1,13->",
                false,
            ),
            (
                "codex sentinel as current snapshot",
                "codex",
                "codex:turn:turn-a:cumulative:10,2,0,3,1,13->start",
                false,
            ),
            (
                "short codex previous snapshot",
                "codex",
                "codex:turn:turn-a:cumulative:1,2,3,4,5->10,2,0,3,1,13",
                false,
            ),
            (
                "long codex current snapshot",
                "codex",
                "codex:turn:turn-a:cumulative:10,2,0,3,1,13->20,4,1,6,2,26,27",
                false,
            ),
            (
                "non-decimal codex snapshot value",
                "codex",
                "codex:turn:turn-a:cumulative:10,2,0,3,1,13->20,4,1,6,2,nope",
                false,
            ),
            (
                "negative codex snapshot value",
                "codex",
                "codex:turn:turn-a:cumulative:start->10,2,0,3,-1,13",
                false,
            ),
            (
                "codex total mismatch",
                "codex",
                "codex:turn:turn-a:cumulative:start->10,2,0,3,1,99",
                false,
            ),
            (
                "codex cache exceeds input",
                "codex",
                "codex:turn:turn-a:cumulative:start->10,8,3,3,1,13",
                false,
            ),
            (
                "codex reasoning exceeds output",
                "codex",
                "codex:turn:turn-a:cumulative:start->10,2,0,3,4,13",
                false,
            ),
            ("empty gemini session", "gemini", "gemini::message-a", false),
            ("empty gemini message", "gemini", "gemini:session-a:", false),
            ("empty gemini msg", "gemini", "gemini:msg:", false),
            ("empty kimi response", "kimi", "kimi:response:", false),
            ("empty omp message", "omp", "omp:message:", false),
            ("empty omp response", "omp", "omp:response:", false),
            (
                "empty omp paired message",
                "omp",
                "omp:message::response:response-a",
                false,
            ),
            (
                "empty omp paired response",
                "omp",
                "omp:message:message-a:response:",
                false,
            ),
        ];

        for (name, vendor, dedup_key, expected) in cases {
            assert_eq!(
                is_globally_stable_usage_key(vendor, dedup_key),
                expected,
                "case={name:?}, vendor={vendor:?}, dedup_key={dedup_key:?}"
            );
        }
    }

    #[test]
    fn codex_usage_snapshot_requires_consistent_nonnegative_totals() {
        assert!(is_valid_codex_usage_snapshot(10, 2, 1, 3, 1, 13));
        assert!(!is_valid_codex_usage_snapshot(10, 2, 1, 3, 4, 13));
        assert!(!is_valid_codex_usage_snapshot(10, 8, 3, 3, 1, 13));
        assert!(!is_valid_codex_usage_snapshot(10, 2, 1, 3, 1, 99));
        assert!(!is_valid_codex_usage_snapshot(10, 2, -1, 3, 1, 13));
    }

    #[test]
    fn rejects_empty_required_strings() {
        for mutate in [
            |record: &mut WireRecord| record.dedup_key.clear(),
            |record: &mut WireRecord| record.timestamp.clear(),
            |record: &mut WireRecord| record.session_start_time.clear(),
            |record: &mut WireRecord| record.session_end_time.clear(),
            |record: &mut WireRecord| record.model.clear(),
        ] {
            let mut record = valid_record();
            mutate(&mut record);

            assert!(record.validate().is_err());
        }
    }

    #[test]
    fn rejects_bad_timestamps() {
        let mut record = valid_record();
        record.timestamp = "not-a-time".to_string();

        assert!(record.validate().is_err());
    }

    #[test]
    fn rejects_negative_token_counts() {
        let mut record = valid_record();
        record.output_tokens = -1;

        assert!(record.validate().is_err());
    }

    #[test]
    fn missing_fast_tier_defaults_to_unknown() {
        let record: WireRecord = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "host_id": "workstation-home",
                "vendor": "claude",
                "dedup_key": "source-record-1",
                "timestamp": "2026-05-18T12:34:56Z",
                "session_start_time": "2026-05-18T12:30:00Z",
                "session_end_time": "2026-05-18T12:34:56Z",
                "model": "claude-sonnet-4",
                "effort": "high",
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
                "reasoning_output_tokens": 50,
                "project_path_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }"#,
        )
        .expect("deserialize");

        assert_eq!(record.fast_tier, -1);
    }

    #[test]
    fn aggregate_only_wire_records_gain_a_five_minute_bucket() {
        let mut record: WireRecord = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "host_id": "workstation-home",
                "vendor": "claude",
                "dedup_key": "source-record-1",
                "timestamp": "2026-05-18T12:34:56Z",
                "session_start_time": "2026-05-18T12:30:00Z",
                "session_end_time": "2026-05-18T12:34:56Z",
                "model": "claude-sonnet-4",
                "effort": null,
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
                "reasoning_output_tokens": 0,
                "project_path_sha256": null
            }"#,
        )
        .expect("deserialize aggregate-only record");

        assert!(record.validate().is_ok());
        assert_eq!(record.cache_creation_5m_input_tokens, 0);
        assert_eq!(record.cache_creation_1h_input_tokens, 0);

        record = record.normalize_for_current_schema();
        assert_eq!(record.schema_version, SCHEMA_VERSION);
        assert_eq!(record.cache_creation_5m_input_tokens, 40);
        assert_eq!(record.cache_creation_1h_input_tokens, 0);
        assert_eq!(record.cache_creation_input_tokens, 40);
    }

    #[test]
    fn missing_costs_default_to_absent() {
        let record: WireRecord = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "host_id": "workstation-home",
                "vendor": "omp",
                "dedup_key": "source-record-1",
                "timestamp": "2026-05-18T12:34:56Z",
                "session_start_time": "2026-05-18T12:30:00Z",
                "session_end_time": "2026-05-18T12:34:56Z",
                "model": "gpt-5.5",
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
                "reasoning_output_tokens": 0,
                "project_path_sha256": null
            }"#,
        )
        .expect("deserialize record without costs");

        assert_eq!(record.cost_input, None);
        assert_eq!(record.cost_output, None);
        assert_eq!(record.cost_cache_read, None);
        assert_eq!(record.cost_cache_creation, None);
        assert!(record.validate().is_ok());
    }

    #[test]
    fn rejects_negative_costs() {
        let mut record = valid_record();
        record.cost_input = Some(-0.01);

        assert!(record.validate().is_err());
    }

    #[test]
    fn rejects_bad_project_hash() {
        for hash in ["abc", "g".repeat(64).as_str(), "A".repeat(64).as_str()] {
            let mut record = valid_record();
            record.project_path_sha256 = Some(hash.to_string());

            assert!(record.validate().is_err(), "hash={hash:?}");
        }
    }

    #[test]
    fn integrity_report_validates_algorithm_host_timestamps_and_digest() {
        let report = IntegrityReport {
            host_id: "workstation-home".to_string(),
            algorithm: INTEGRITY_ALGORITHM.to_string(),
            range_end_utc: "2026-06-01T00:00:00Z".to_string(),
            record_count: 2,
            digest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            computed_at: "2026-06-01T12:00:00Z".to_string(),
        };
        report.validate().expect("valid integrity report");

        let mut invalid_algorithm = report.clone();
        invalid_algorithm.algorithm = "other".to_string();
        assert_eq!(
            invalid_algorithm.validate(),
            Err(ValidationError::InvalidIntegrityAlgorithm)
        );

        let mut invalid_digest = report.clone();
        invalid_digest.digest_sha256 =
            "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef".to_string();
        assert_eq!(
            invalid_digest.validate(),
            Err(ValidationError::InvalidDigest)
        );
    }

    #[test]
    fn health_response_accepts_legacy_payload_without_instance_id() {
        let response: HealthResponse = serde_json::from_str(
            r#"{
                "ok": true,
                "version": "3.0.0",
                "schema_version": 1,
                "uptime_seconds": 42
            }"#,
        )
        .expect("deserialize legacy health response");

        assert_eq!(response.instance_id, None);
        assert_eq!(response.sync_policy, None);
    }

    #[test]
    fn health_response_round_trips_sync_policy() {
        let response = HealthResponse {
            ok: true,
            version: "3.3.0".to_string(),
            schema_version: SCHEMA_VERSION,
            uptime_seconds: 42,
            instance_id: Some("server-a".to_string()),
            sync_policy: Some(SyncPolicy {
                min_request_interval_ms: 1_500,
                request_phase_ms: 250,
                max_request_interval_ms: 60_000,
            }),
        };

        let encoded = serde_json::to_string(&response).expect("serialize health response");
        let decoded: HealthResponse =
            serde_json::from_str(&encoded).expect("deserialize health response");

        assert_eq!(decoded, response);
    }

    #[test]
    fn machine_info_accepts_legacy_payload_without_content_revision() {
        let machine: MachineInfo = serde_json::from_str(
            r#"{
                "host_id": "workstation-home",
                "last_seen": "2026-07-23T18:00:00Z",
                "record_count": 42
            }"#,
        )
        .expect("deserialize legacy machine info");

        assert_eq!(machine.content_revision, None);
    }

    #[test]
    fn legacy_machine_info_ignores_new_content_revision() {
        #[derive(Deserialize)]
        struct LegacyMachineInfo {
            host_id: String,
            last_seen: String,
            record_count: u64,
        }

        let current = MachineInfo {
            host_id: "workstation-home".to_string(),
            last_seen: "2026-07-23T18:00:00Z".to_string(),
            record_count: 42,
            content_revision: Some(84),
        };
        let legacy: LegacyMachineInfo = serde_json::from_value(
            serde_json::to_value(current).expect("serialize current machine info"),
        )
        .expect("deserialize with legacy machine info");

        assert_eq!(legacy.host_id, "workstation-home");
        assert_eq!(legacy.last_seen, "2026-07-23T18:00:00Z");
        assert_eq!(legacy.record_count, 42);
    }
}
