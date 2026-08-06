use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const SYNC_STATE_SCHEMA_VERSION: u32 = 1;
pub const SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION: u32 = 2;

const SYNC_STATE_FILE: &str = "sync_state.json";
const SYNC_UPLOAD_LOG_FILE: &str = "sync_upload_log.bin";
const SYNC_SNAPSHOT_STATE_FILE: &str = "sync_snapshot_state.bin";
const SYNC_SNAPSHOT_MARKER_FILE: &str = "sync_snapshot_marker.json";
const SYNC_SNAPSHOT_ATTEMPT_FILE: &str = "sync_snapshot_attempt.json";
const SYNC_SNAPSHOT_PENDING_FILE: &str = "sync_snapshot_pending.bin";
const SNAPSHOT_MARKER_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub schema_version: u32,
    pub last_seen_seq: u64,
    #[serde(default)]
    pub pull_vendors: Vec<String>,
    #[serde(default)]
    pub pull_scope: String,
    #[serde(default)]
    pub last_full_pull: Option<String>,
    pub last_successful_sync: Option<String>,
    pub last_error: Option<String>,
    #[serde(default)]
    pub integrity_check: Option<IntegrityCheckState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrityCheckState {
    pub checked_at: String,
    pub range_end_utc: String,
    pub checked_hosts: usize,
    #[serde(default)]
    pub sync_scope: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotUploadState {
    pub schema_version: u32,
    pub full_hash: String,
    pub cache_generation: String,
    pub record_hashes: BTreeMap<(String, String), String>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SnapshotCacheMarker {
    schema_version: u32,
    cache_generation: String,
    sync_scope: String,
    record_count: usize,
    #[serde(default)]
    content_revision: Option<u64>,
    #[serde(default)]
    verified_at_secs: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PendingSnapshotRecord {
    schema_version: u32,
    snapshot_id: String,
    sync_scope: String,
    stable_data_changed: bool,
    state: SnapshotUploadState,
}

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotAttemptMarker {
    schema_version: u32,
    snapshot_id: String,
    sync_scope: String,
    cache_generation: String,
}

#[derive(Debug)]
pub struct PendingSnapshotUpload {
    pub snapshot_id: String,
    pub stable_data_changed: bool,
    pub state: SnapshotUploadState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotCacheReceipt {
    pub cache_generation: String,
    pub record_count: usize,
    pub content_revision: Option<u64>,
    pub verified_at_secs: u64,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            schema_version: SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 0,
            pull_vendors: Vec::new(),
            pull_scope: String::new(),
            last_full_pull: None,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        }
    }
}

pub fn load_sync_state(cache_root: &Path) -> SyncState {
    let path = cache_root.join(SYNC_STATE_FILE);
    let Ok(content) = fs::read_to_string(path) else {
        return SyncState::default();
    };
    let Ok(state) = serde_json::from_str::<SyncState>(&content) else {
        return SyncState::default();
    };
    if state.schema_version == SYNC_STATE_SCHEMA_VERSION {
        state
    } else {
        SyncState::default()
    }
}

pub fn save_sync_state(cache_root: &Path, state: &SyncState) -> io::Result<()> {
    let path = cache_root.join(SYNC_STATE_FILE);
    let content = serde_json::to_vec_pretty(state)?;
    atomic_write(&path, &content)
}

pub fn invalidate_pull_state(cache_root: &Path) -> io::Result<()> {
    let mut state = load_sync_state(cache_root);
    state.last_seen_seq = 0;
    state.pull_vendors.clear();
    state.pull_scope.clear();
    state.last_full_pull = None;
    state.integrity_check = None;
    save_sync_state(cache_root, &state)
}

/// Delete the persisted sync cursor. Returns whether a file was removed.
pub fn clear_sync_state(cache_root: &Path) -> io::Result<bool> {
    let path = cache_root.join(SYNC_STATE_FILE);
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

pub fn load_upload_log(cache_root: &Path) -> BTreeSet<(String, String)> {
    let path = cache_root.join(SYNC_UPLOAD_LOG_FILE);
    let Ok(content) = fs::read(path) else {
        return BTreeSet::new();
    };
    bincode::deserialize::<BTreeSet<(String, String)>>(&content).unwrap_or_default()
}

pub fn save_upload_log(cache_root: &Path, keys: &BTreeSet<(String, String)>) -> io::Result<()> {
    let path = cache_root.join(SYNC_UPLOAD_LOG_FILE);
    let content = bincode::serialize(keys).map_err(io::Error::other)?;
    atomic_write(&path, &content)
}

pub fn load_snapshot_upload_state(cache_root: &Path) -> SnapshotUploadState {
    let path = cache_root.join(SYNC_SNAPSHOT_STATE_FILE);
    let Ok(content) = fs::read(path) else {
        return SnapshotUploadState::default();
    };
    bincode::deserialize::<SnapshotUploadState>(&content).unwrap_or_default()
}

pub fn save_snapshot_upload_state(
    cache_root: &Path,
    state: &SnapshotUploadState,
    sync_scope: &str,
    content_revision: Option<u64>,
) -> io::Result<()> {
    let path = cache_root.join(SYNC_SNAPSHOT_STATE_FILE);
    let content = bincode::serialize(state).map_err(io::Error::other)?;
    atomic_write(&path, &content)?;
    save_snapshot_cache_marker(cache_root, state, sync_scope, content_revision)?;
    remove_if_exists(&cache_root.join(SYNC_SNAPSHOT_ATTEMPT_FILE))?;
    remove_if_exists(&cache_root.join(SYNC_SNAPSHOT_PENDING_FILE))
}

pub fn snapshot_attempt_id(
    cache_root: &Path,
    sync_scope: &str,
    cache_generation: &str,
    candidate: &str,
) -> io::Result<String> {
    let path = cache_root.join(SYNC_SNAPSHOT_ATTEMPT_FILE);
    if let Ok(content) = fs::read(&path)
        && let Ok(marker) = serde_json::from_slice::<SnapshotAttemptMarker>(&content)
        && marker.schema_version == SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION
        && marker.sync_scope == sync_scope
        && marker.cache_generation == cache_generation
    {
        return Ok(marker.snapshot_id);
    }
    let marker = SnapshotAttemptMarker {
        schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
        snapshot_id: candidate.to_string(),
        sync_scope: sync_scope.to_string(),
        cache_generation: cache_generation.to_string(),
    };
    atomic_write(&path, &serde_json::to_vec(&marker)?)?;
    Ok(marker.snapshot_id)
}

pub fn save_pending_snapshot_upload(
    cache_root: &Path,
    state: &SnapshotUploadState,
    sync_scope: &str,
    snapshot_id: &str,
    stable_data_changed: bool,
) -> io::Result<()> {
    let pending = PendingSnapshotRecord {
        schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
        snapshot_id: snapshot_id.to_string(),
        sync_scope: sync_scope.to_string(),
        stable_data_changed,
        state: state.clone(),
    };
    let content = bincode::serialize(&pending).map_err(io::Error::other)?;
    atomic_write(&cache_root.join(SYNC_SNAPSHOT_PENDING_FILE), &content)
}

pub fn pending_snapshot_upload(
    cache_root: &Path,
    sync_scope: &str,
    cache_generation: &str,
) -> Option<PendingSnapshotUpload> {
    let content = fs::read(cache_root.join(SYNC_SNAPSHOT_PENDING_FILE)).ok()?;
    let pending = bincode::deserialize::<PendingSnapshotRecord>(&content).ok()?;
    if pending.schema_version != SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION
        || pending.sync_scope != sync_scope
    {
        return None;
    }
    if pending.state.schema_version != SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION
        || pending.state.cache_generation != cache_generation
    {
        return None;
    }
    Some(PendingSnapshotUpload {
        snapshot_id: pending.snapshot_id,
        stable_data_changed: pending.stable_data_changed,
        state: pending.state,
    })
}

pub fn discard_snapshot_attempt(cache_root: &Path) -> io::Result<()> {
    remove_if_exists(&cache_root.join(SYNC_SNAPSHOT_ATTEMPT_FILE))?;
    remove_if_exists(&cache_root.join(SYNC_SNAPSHOT_PENDING_FILE))
}

pub fn commit_pending_snapshot_upload(
    cache_root: &Path,
    state: &SnapshotUploadState,
    sync_scope: &str,
    content_revision: Option<u64>,
) -> io::Result<()> {
    save_snapshot_upload_state(cache_root, state, sync_scope, content_revision)
}

fn save_snapshot_cache_marker(
    cache_root: &Path,
    state: &SnapshotUploadState,
    sync_scope: &str,
    content_revision: Option<u64>,
) -> io::Result<()> {
    let marker_path = cache_root.join(SYNC_SNAPSHOT_MARKER_FILE);
    if state.cache_generation.is_empty() || sync_scope.is_empty() {
        match fs::remove_file(marker_path) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        return Ok(());
    }
    let marker = SnapshotCacheMarker {
        schema_version: SNAPSHOT_MARKER_SCHEMA_VERSION,
        cache_generation: state.cache_generation.clone(),
        sync_scope: sync_scope.to_string(),
        record_count: state.record_hashes.len(),
        content_revision,
        verified_at_secs: unix_time_secs(),
    };
    let marker_content = serde_json::to_vec(&marker)?;
    atomic_write(&marker_path, &marker_content)
}

pub fn snapshot_cache_receipt(cache_root: &Path, sync_scope: &str) -> Option<SnapshotCacheReceipt> {
    let marker = load_snapshot_cache_marker(cache_root)?;
    (marker.sync_scope == sync_scope).then_some(SnapshotCacheReceipt {
        cache_generation: marker.cache_generation,
        record_count: marker.record_count,
        content_revision: marker.content_revision,
        verified_at_secs: marker.verified_at_secs,
    })
}

fn load_snapshot_cache_marker(cache_root: &Path) -> Option<SnapshotCacheMarker> {
    let content = fs::read(cache_root.join(SYNC_SNAPSHOT_MARKER_FILE)).ok()?;
    let marker = serde_json::from_slice::<SnapshotCacheMarker>(&content).ok()?;
    (marker.schema_version == SNAPSHOT_MARKER_SCHEMA_VERSION).then_some(marker)
}

fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("sync-state");
    let tmp_path = path.with_file_name(format!("{file_name}.tmp-{stamp}"));
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai-usage-state-test-{name}-{stamp}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn missing_sync_state_returns_default() {
        let cache_root = unique_temp_dir("missing-state");

        let state = load_sync_state(&cache_root);

        assert_eq!(
            state,
            SyncState {
                schema_version: SYNC_STATE_SCHEMA_VERSION,
                last_seen_seq: 0,
                pull_vendors: Vec::new(),
                pull_scope: String::new(),
                last_full_pull: None,
                last_successful_sync: None,
                last_error: None,
                integrity_check: None,
            }
        );
    }

    #[test]
    fn sync_state_round_trips_as_json() {
        let cache_root = unique_temp_dir("round-trip-state");
        let state = SyncState {
            schema_version: SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 42,
            pull_vendors: vec!["claude".to_string(), "codex".to_string()],
            pull_scope: "server-a".to_string(),
            last_full_pull: Some("2026-05-18T12:00:00Z".to_string()),
            last_successful_sync: Some("2026-05-18T12:34:56Z".to_string()),
            last_error: Some("temporary network error".to_string()),
            integrity_check: None,
        };

        save_sync_state(&cache_root, &state).expect("save state");

        assert_eq!(load_sync_state(&cache_root), state);
        let raw = fs::read_to_string(cache_root.join("sync_state.json")).expect("read json");
        assert!(raw.contains("\"last_seen_seq\": 42"));
    }

    #[test]
    fn pull_invalidation_preserves_upload_and_diagnostic_state() {
        let cache_root = unique_temp_dir("invalidate-pull");
        let state = SyncState {
            schema_version: SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 42,
            pull_vendors: vec!["claude".to_string(), "codex".to_string()],
            pull_scope: "server-a".to_string(),
            last_full_pull: Some("2026-05-18T12:00:00Z".to_string()),
            last_successful_sync: Some("2026-05-18T12:34:56Z".to_string()),
            last_error: Some("temporary network error".to_string()),
            integrity_check: Some(IntegrityCheckState {
                checked_at: "2026-05-18T12:34:56Z".to_string(),
                range_end_utc: "2026-05-18T00:00:00Z".to_string(),
                checked_hosts: 2,
                sync_scope: "scope-a".to_string(),
            }),
        };
        let snapshot = SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
            full_hash: "full-hash".to_string(),
            cache_generation: "generation-a".to_string(),
            record_hashes: BTreeMap::from([(
                ("claude".to_string(), "response-a".to_string()),
                "record-hash".to_string(),
            )]),
        };
        save_sync_state(&cache_root, &state).expect("save sync state");
        save_snapshot_upload_state(&cache_root, &snapshot, "scope-a", Some(9))
            .expect("save snapshot state");

        invalidate_pull_state(&cache_root).expect("invalidate pull state");

        assert_eq!(
            load_sync_state(&cache_root),
            SyncState {
                last_successful_sync: state.last_successful_sync,
                last_error: state.last_error,
                ..SyncState::default()
            }
        );
        assert_eq!(load_snapshot_upload_state(&cache_root), snapshot);
    }

    #[test]
    fn integrity_check_state_round_trips_with_sync_state() {
        let cache_root = unique_temp_dir("integrity-check-state");
        let state = SyncState {
            integrity_check: Some(IntegrityCheckState {
                checked_at: "2026-07-23T18:00:00Z".to_string(),
                range_end_utc: "2026-07-23T00:00:00Z".to_string(),
                checked_hosts: 5,
                sync_scope: "scope-a".to_string(),
            }),
            ..SyncState::default()
        };

        save_sync_state(&cache_root, &state).expect("save state");

        assert_eq!(load_sync_state(&cache_root), state);
    }

    #[test]
    fn legacy_integrity_check_without_sync_scope_stays_readable() {
        let cache_root = unique_temp_dir("legacy-integrity-check");
        fs::write(
            cache_root.join(SYNC_STATE_FILE),
            br#"{
                "schema_version": 1,
                "last_seen_seq": 42,
                "pull_vendors": ["claude", "codex", "gemini"],
                "last_successful_sync": null,
                "last_error": null,
                "integrity_check": {
                    "checked_at": "2026-07-23T18:00:00Z",
                    "range_end_utc": "2026-07-23T00:00:00Z",
                    "checked_hosts": 5
                }
            }"#,
        )
        .expect("write legacy state");

        let state = load_sync_state(&cache_root);

        assert_eq!(
            state
                .integrity_check
                .expect("legacy integrity check")
                .sync_scope,
            ""
        );
    }

    #[test]
    fn corrupt_sync_state_returns_default() {
        let cache_root = unique_temp_dir("corrupt-state");
        fs::write(cache_root.join("sync_state.json"), b"not-json").expect("write corrupt state");

        let state = load_sync_state(&cache_root);

        assert_eq!(state, SyncState::default());
    }

    #[test]
    fn upload_log_round_trips_as_bincode_set() {
        let cache_root = unique_temp_dir("round-trip-log");
        let mut keys = BTreeSet::new();
        keys.insert(("claude".to_string(), "a".to_string()));
        keys.insert(("codex".to_string(), "b".to_string()));

        save_upload_log(&cache_root, &keys).expect("save upload log");

        assert_eq!(load_upload_log(&cache_root), keys);
    }

    #[test]
    fn missing_or_corrupt_upload_log_returns_empty_set() {
        let missing_root = unique_temp_dir("missing-log");
        assert!(load_upload_log(&missing_root).is_empty());

        let corrupt_root = unique_temp_dir("corrupt-log");
        fs::write(corrupt_root.join("sync_upload_log.bin"), b"not-bincode")
            .expect("write corrupt log");

        assert!(load_upload_log(&corrupt_root).is_empty());
    }

    #[test]
    fn snapshot_state_keeps_the_previous_binary_layout() {
        #[derive(Serialize, Deserialize)]
        struct PreviousSnapshotUploadState {
            schema_version: u32,
            full_hash: String,
            key_set_hash: String,
            record_hashes: BTreeMap<(String, String), String>,
        }

        let previous = PreviousSnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
            full_hash: "full".to_string(),
            key_set_hash: "legacy-key-set".to_string(),
            record_hashes: BTreeMap::from([(
                ("claude".to_string(), "dedup".to_string()),
                "record".to_string(),
            )]),
        };
        let encoded = bincode::serialize(&previous).expect("serialize previous state");
        let current: SnapshotUploadState =
            bincode::deserialize(&encoded).expect("deserialize current state");

        assert_eq!(current.schema_version, SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION);
        assert_eq!(current.full_hash, "full");
        assert_eq!(current.cache_generation, "legacy-key-set");
        assert_eq!(
            current
                .record_hashes
                .get(&("claude".to_string(), "dedup".to_string())),
            Some(&"record".to_string())
        );

        let current_encoded = bincode::serialize(&current).expect("serialize current state");
        let previous_again: PreviousSnapshotUploadState =
            bincode::deserialize(&current_encoded).expect("deserialize previous state");
        assert_eq!(previous_again.key_set_hash, "legacy-key-set");
    }

    #[test]
    fn snapshot_cache_marker_is_a_scoped_upload_receipt() {
        let cache_root = unique_temp_dir("snapshot-cache-marker");
        let snapshot = SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
            full_hash: "full".to_string(),
            cache_generation: "generation-a".to_string(),
            record_hashes: BTreeMap::new(),
        };
        save_snapshot_upload_state(&cache_root, &snapshot, "scope-a", Some(42))
            .expect("save snapshot state");

        let receipt =
            snapshot_cache_receipt(&cache_root, "scope-a").expect("matching snapshot receipt");
        assert_eq!(receipt.cache_generation, "generation-a");
        assert_eq!(receipt.record_count, 0);
        assert_eq!(receipt.content_revision, Some(42));
        assert!(receipt.verified_at_secs > 0);
        assert_eq!(snapshot_cache_receipt(&cache_root, "scope-b"), None);

        fs::write(cache_root.join(SYNC_SNAPSHOT_STATE_FILE), b"corrupt state")
            .expect("replace snapshot state");
        assert_eq!(
            snapshot_cache_receipt(&cache_root, "scope-a")
                .expect("receipt remains independent")
                .record_count,
            0
        );
    }

    #[test]
    fn torn_pending_snapshot_pair_is_rejected() {
        let cache_root = unique_temp_dir("torn-pending-snapshot");
        let first = SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
            full_hash: "first".to_string(),
            cache_generation: "generation-a".to_string(),
            record_hashes: BTreeMap::new(),
        };
        save_pending_snapshot_upload(&cache_root, &first, "scope-a", "snapshot-a", true)
            .expect("save first pending snapshot");

        let second = SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_SCHEMA_VERSION,
            full_hash: "second".to_string(),
            cache_generation: "generation-b".to_string(),
            record_hashes: BTreeMap::new(),
        };
        let encoded = bincode::serialize(&second).expect("serialize second state");
        atomic_write(
            &cache_root.join("sync_snapshot_pending_state.bin"),
            &encoded,
        )
        .expect("replace pending state only");

        assert!(pending_snapshot_upload(&cache_root, "scope-a", "generation-b").is_none());
        let pending = pending_snapshot_upload(&cache_root, "scope-a", "generation-a")
            .expect("original pending snapshot");
        assert_eq!(pending.snapshot_id, "snapshot-a");
        assert_eq!(pending.state.full_hash, "first");
    }

    #[test]
    fn clear_sync_state_removes_existing_file_and_is_idempotent() {
        let cache_root = unique_temp_dir("clear-state");
        save_sync_state(
            &cache_root,
            &SyncState {
                schema_version: SYNC_STATE_SCHEMA_VERSION,
                last_seen_seq: 99,
                pull_vendors: Vec::new(),
                pull_scope: String::new(),
                last_full_pull: None,
                last_successful_sync: Some("2026-05-20T00:00:00Z".to_string()),
                last_error: None,
                integrity_check: None,
            },
        )
        .expect("save state");

        assert!(clear_sync_state(&cache_root).expect("first clear"));
        assert!(!cache_root.join("sync_state.json").exists());
        assert_eq!(load_sync_state(&cache_root), SyncState::default());

        assert!(!clear_sync_state(&cache_root).expect("second clear"));
    }

    #[test]
    fn atomic_writes_leave_only_final_files() {
        let cache_root = unique_temp_dir("atomic");
        let state = SyncState {
            schema_version: SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 7,
            pull_vendors: Vec::new(),
            pull_scope: String::new(),
            last_full_pull: None,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        };
        let mut keys = BTreeSet::new();
        keys.insert(("gemini".to_string(), "dedup".to_string()));

        save_sync_state(&cache_root, &state).expect("save state");
        save_upload_log(&cache_root, &keys).expect("save log");

        let names: BTreeSet<String> = fs::read_dir(&cache_root)
            .expect("read cache root")
            .map(|entry| {
                entry
                    .expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        assert!(names.contains("sync_state.json"));
        assert!(names.contains("sync_upload_log.bin"));
        assert!(names.iter().all(|name| !name.contains(".tmp-")));
    }
}
