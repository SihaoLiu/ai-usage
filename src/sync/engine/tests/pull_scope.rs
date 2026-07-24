use super::*;

#[test]
fn pull_resets_a_high_cursor_when_the_server_scope_changes() {
    let cache_root = unique_temp_dir("pull-server-scope");
    let config = enabled_config("workstation");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 500,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(
                &config,
                Some("server-a"),
            ),
            last_full_pull: None,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        },
    )
    .expect("save prior cursor");
    let pulled = sequenced_remote_record(
        5,
        remote_usage_record(
            "laptop",
            "claude",
            "lower-sequence",
            "2026-05-18T12:00:00Z",
            10,
        ),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![pulled],
        max_seq: 5,
        truncated: false,
    }])
    .with_server_instance("server-b");

    run_pull_once_with_progress(&cache_root, &config, &transport, |_| {})
        .expect("pull from replacement server");

    assert_eq!(transport.pull_requests.borrow()[0].0, 0);
    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(state.last_seen_seq, 5);
    assert_eq!(
        state.pull_scope,
        crate::sync::cache_generation::server_scope_fingerprint(&config, Some("server-b"))
    );
}

#[test]
fn legacy_pull_periodically_backfills_when_the_scope_is_unchanged() {
    let cache_root = unique_temp_dir("pull-legacy-backfill");
    let config = enabled_config("workstation");
    std::fs::write(
        cache_root.join("sync_state.json"),
        serde_json::json!({
            "schema_version": crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            "last_seen_seq": 500,
            "pull_vendors": pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS),
            "pull_scope": crate::sync::cache_generation::server_scope_fingerprint(
                &config,
                None,
            ),
            "last_full_pull": "2000-01-01T00:00:00Z",
            "last_successful_sync": null,
            "last_error": null,
            "integrity_check": null,
        })
        .to_string(),
    )
    .expect("save legacy cursor");
    let pulled = sequenced_remote_record(
        5,
        remote_usage_record(
            "laptop",
            "claude",
            "lower-sequence",
            "2026-05-18T12:00:00Z",
            10,
        ),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![pulled],
        max_seq: 5,
        truncated: false,
    }]);

    run_pull_once_with_progress(&cache_root, &config, &transport, |_| {})
        .expect("periodic legacy backfill");

    assert_eq!(transport.pull_requests.borrow()[0].0, 0);
    assert_eq!(
        crate::sync::state::load_sync_state(&cache_root).last_seen_seq,
        5
    );

    let incremental = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 5,
        truncated: false,
    }]);
    run_pull_once_with_progress(&cache_root, &config, &incremental, |_| {})
        .expect("incremental legacy pull");

    assert_eq!(incremental.pull_requests.borrow()[0].0, 5);
}
