use super::*;

#[test]
fn pull_downgrade_after_rollback_keeps_cached_records_and_cursor() {
    let cache_root = unique_temp_dir("pull-rollback");
    let cached_kimi =
        remote_usage_record("laptop", "kimi", "remote-kimi", "2026-05-18T12:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![cached_kimi])
        .expect("seed remote cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(
                &enabled_config("workstation"),
                None,
            ),
            last_full_pull: Some(Utc::now().to_rfc3339()),
            full_pull_in_progress: false,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        },
    )
    .expect("save migrated state");
    let transport = FakeTransport::new_rejecting_pull_vendor(
        vec![PullResponse {
            records: Vec::new(),
            max_seq: 12,
            truncated: false,
        }],
        "kimi",
    );

    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("rollback pull should downgrade without wiping");

    let requests = transport.pull_requests.borrow();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].0, 10);
    assert!(requests.iter().all(|request| request.1 == "workstation"));
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "remote-kimi");
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state.pull_vendors,
        pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS, "workstation")
    );
}

#[test]
fn pull_preserves_cache_and_downgrades_when_older_server_rejects_new_vendor() {
    let cache_root = unique_temp_dir("pull-older-server");
    let existing = remote_usage_record("laptop", "claude", "remote-a", "2026-05-18T12:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![existing])
        .expect("seed remote cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(
                &enabled_config("workstation"),
                None,
            ),
            last_full_pull: Some(Utc::now().to_rfc3339()),
            full_pull_in_progress: false,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        },
    )
    .expect("save pre-upgrade state");
    let new_remote =
        remote_usage_record("laptop", "claude", "remote-b", "2026-05-18T13:00:00Z", 20);
    let transport = FakeTransport::new_rejecting_pull_vendor(
        vec![PullResponse {
            records: vec![sequenced_remote_record(11, new_remote)],
            max_seq: 11,
            truncated: false,
        }],
        "kimi",
    );

    let mut events = Vec::new();
    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("pull should fall back to the previous vendor set");

    let requests = transport.pull_requests.borrow();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].3.contains(&"kimi".to_string()));
    assert!(!requests[1].3.contains(&"kimi".to_string()));
    assert_eq!(requests[1].0, 10);
    assert!(requests.iter().all(|request| request.1 == "workstation"));
    assert!(events.iter().any(|event| matches!(
        event,
        SyncProgress::PullVendorsUnavailable { vendors }
            if vendors == &vec!["kimi".to_string()]
    )));
    let mut remote_keys: Vec<String> = crate::data::cache::load_remote_entries(&cache_root, None)
        .iter()
        .map(|record| record.dedup_key.clone())
        .collect();
    remote_keys.sort();
    assert_eq!(remote_keys, ["remote-a", "remote-b"]);
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state.pull_vendors,
        pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS, "workstation")
    );
}

#[test]
fn pull_migrates_vendor_fingerprint_once_server_accepts_new_vendor() {
    let cache_root = unique_temp_dir("pull-migrate");
    let stale = remote_usage_record("laptop", "claude", "stale-a", "2026-05-18T12:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![stale])
        .expect("seed remote cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 10,
            pull_vendors: pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(
                &enabled_config("workstation"),
                None,
            ),
            last_full_pull: Some(Utc::now().to_rfc3339()),
            full_pull_in_progress: false,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        },
    )
    .expect("save pre-upgrade state");
    let refetched =
        remote_usage_record("laptop", "kimi", "remote-kimi", "2026-05-18T13:00:00Z", 20);
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![sequenced_remote_record(12, refetched)],
        max_seq: 12,
        truncated: false,
    }]);

    run_pull_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("pull should migrate to the full vendor set");

    let requests = transport.pull_requests.borrow();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0, 0);
    assert_eq!(requests[0].1, "workstation");
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "remote-kimi");
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state.pull_vendors,
        pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation")
    );
}
