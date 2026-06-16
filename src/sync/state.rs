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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub schema_version: u32,
    pub last_seen_seq: u64,
    #[serde(default)]
    pub pull_vendors: Vec<String>,
    pub last_successful_sync: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotUploadState {
    pub schema_version: u32,
    pub full_hash: String,
    pub key_set_hash: String,
    pub record_hashes: BTreeMap<(String, String), String>,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            schema_version: SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 0,
            pull_vendors: Vec::new(),
            last_successful_sync: None,
            last_error: None,
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
) -> io::Result<()> {
    let path = cache_root.join(SYNC_SNAPSHOT_STATE_FILE);
    let content = bincode::serialize(state).map_err(io::Error::other)?;
    atomic_write(&path, &content)
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
                last_successful_sync: None,
                last_error: None,
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
            last_successful_sync: Some("2026-05-18T12:34:56Z".to_string()),
            last_error: Some("temporary network error".to_string()),
        };

        save_sync_state(&cache_root, &state).expect("save state");

        assert_eq!(load_sync_state(&cache_root), state);
        let raw = fs::read_to_string(cache_root.join("sync_state.json")).expect("read json");
        assert!(raw.contains("\"last_seen_seq\": 42"));
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
    fn clear_sync_state_removes_existing_file_and_is_idempotent() {
        let cache_root = unique_temp_dir("clear-state");
        save_sync_state(
            &cache_root,
            &SyncState {
                schema_version: SYNC_STATE_SCHEMA_VERSION,
                last_seen_seq: 99,
                pull_vendors: Vec::new(),
                last_successful_sync: Some("2026-05-20T00:00:00Z".to_string()),
                last_error: None,
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
            last_successful_sync: None,
            last_error: None,
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
