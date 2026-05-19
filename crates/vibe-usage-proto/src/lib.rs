use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub project_path_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineList {
    pub machines: Vec<MachineInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub schema_version: u32,
    pub uptime_seconds: u64,
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
    InvalidProjectHash,
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
            Self::InvalidProjectHash => f.write_str("project_path_sha256 must be lowercase hex"),
        }
    }
}

impl std::error::Error for ValidationError {}

impl WireRecord {
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
            ("reasoning_output_tokens", self.reasoning_output_tokens),
        ] {
            if value < 0 {
                return Err(ValidationError::NegativeTokenCount(field));
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

pub fn is_valid_host_id(host_id: &str) -> bool {
    !host_id.is_empty()
        && host_id.len() <= 64
        && host_id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
}

pub fn is_valid_vendor(vendor: &str) -> bool {
    matches!(vendor, "claude" | "codex" | "gemini")
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
            input_tokens: 10,
            output_tokens: 20,
            cache_read_input_tokens: 30,
            cache_creation_input_tokens: 40,
            reasoning_output_tokens: 50,
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
    fn rejects_bad_project_hash() {
        for hash in ["abc", "g".repeat(64).as_str(), "A".repeat(64).as_str()] {
            let mut record = valid_record();
            record.project_path_sha256 = Some(hash.to_string());

            assert!(record.validate().is_err(), "hash={hash:?}");
        }
    }
}
