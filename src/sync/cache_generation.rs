use sha2::{Digest, Sha256};
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::UNIX_EPOCH;

use crate::sync::config::EnabledSyncConfig;

pub(crate) fn local_cache_generation(cache_root: &Path, vendors: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sync-cache-generation-v1");
    for vendor in vendors {
        hasher.update((vendor.len() as u64).to_le_bytes());
        hasher.update(vendor.as_bytes());
        let path = cache_root.join("entries").join(format!("{vendor}.bin"));
        let Ok(metadata) = fs::metadata(path) else {
            hasher.update([0]);
            continue;
        };
        hasher.update([1]);
        hasher.update(metadata.len().to_le_bytes());
        let modified = metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
            .unwrap_or_default();
        hasher.update(modified.as_secs().to_le_bytes());
        hasher.update(modified.subsec_nanos().to_le_bytes());
        #[cfg(unix)]
        {
            hasher.update(metadata.ctime().to_le_bytes());
            hasher.update(metadata.ctime_nsec().to_le_bytes());
            hasher.update(metadata.dev().to_le_bytes());
            hasher.update(metadata.ino().to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(crate) fn sync_scope_fingerprint(
    config: &EnabledSyncConfig,
    server_instance_id: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sync-scope-v1");
    let server_scope = server_scope_fingerprint(config, server_instance_id);
    for value in [&server_scope, &config.machine_id] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update([u8::from(config.upload_project_hash)]);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(crate) fn server_scope_fingerprint(
    config: &EnabledSyncConfig,
    server_instance_id: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sync-server-scope-v1");
    for value in [
        config.server_url.trim_end_matches('/'),
        server_instance_id.unwrap_or(""),
    ] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
