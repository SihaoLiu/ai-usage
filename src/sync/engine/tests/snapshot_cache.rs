use super::*;

fn populated_pull_state() -> crate::sync::state::SyncState {
    crate::sync::state::SyncState {
        schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
        last_seen_seq: 42,
        pull_vendors: SUPPORTED_PULL_VENDORS
            .iter()
            .map(|vendor| (*vendor).to_string())
            .collect(),
        pull_scope: "server-a".to_string(),
        last_full_pull: Some("2026-05-18T12:00:00Z".to_string()),
        last_successful_sync: Some("2026-05-18T12:34:56Z".to_string()),
        last_error: None,
        integrity_check: None,
    }
}

#[test]
fn local_cache_generation_tracks_vendor_cache_changes() {
    let cache_root = unique_temp_dir("snapshot-cache-generation");
    let missing = crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS);

    populate_vendor_cache(&cache_root, "claude", "first");
    let first = crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS);
    let unchanged = crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS);
    populate_vendor_cache(&cache_root, "codex", "second");
    let second = crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS);

    assert_ne!(missing, first);
    assert_eq!(first, unchanged);
    assert_ne!(first, second);
}

#[test]
fn snapshot_attempt_reuses_an_id_until_the_upload_is_confirmed() {
    let cache_root = unique_temp_dir("snapshot-attempt-id");
    let first_candidate = snapshot_id("workstation");
    let first = crate::sync::state::snapshot_attempt_id(
        &cache_root,
        "scope-a",
        "generation-a",
        &first_candidate,
    )
    .expect("first attempt");
    let repeated = crate::sync::state::snapshot_attempt_id(
        &cache_root,
        "scope-a",
        "generation-a",
        &snapshot_id("workstation"),
    )
    .expect("repeated attempt");
    assert_eq!(first, repeated);

    crate::sync::state::save_snapshot_upload_state(
        &cache_root,
        &crate::sync::state::SnapshotUploadState {
            schema_version: SNAPSHOT_UPLOAD_STATE_VERSION,
            full_hash: "full".to_string(),
            cache_generation: "generation-a".to_string(),
            record_hashes: BTreeMap::new(),
        },
        "scope-a",
        None,
    )
    .expect("confirm attempt");
    let next = crate::sync::state::snapshot_attempt_id(
        &cache_root,
        "scope-a",
        "generation-a",
        &snapshot_id("workstation"),
    )
    .expect("next attempt");

    assert_ne!(first, next);
    assert!(next.len() <= 128);
}

#[test]
fn snapshot_state_records_the_local_cache_generation() {
    let cache_root = unique_temp_dir("snapshot-state-generation");
    populate_vendor_cache(&cache_root, "claude", "first");
    let transport = DiffSnapshotTransport::new(Vec::new());

    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("snapshot upload");

    let state = crate::sync::state::load_snapshot_upload_state(&cache_root);
    assert_eq!(
        state.cache_generation,
        crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS)
    );
}

#[test]
fn full_snapshot_reconciliation_preserves_the_peer_pull_cursor() {
    let cache_root = unique_temp_dir("snapshot-preserves-peer-pull");
    let config = enabled_config("workstation");
    let server_instance = "server-instance";
    populate_vendor_cache(&cache_root, "claude", "first");
    let mut pull_state = populated_pull_state();
    pull_state.pull_vendors = pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation");
    pull_state.pull_scope =
        crate::sync::cache_generation::server_scope_fingerprint(&config, Some(server_instance));
    pull_state.integrity_check = Some(crate::sync::state::IntegrityCheckState {
        checked_at: "2026-05-18T12:34:56Z".to_string(),
        range_end_utc: "2026-05-18T00:00:00Z".to_string(),
        checked_hosts: 2,
        sync_scope: "scope-a".to_string(),
    });
    crate::sync::state::save_sync_state(&cache_root, &pull_state).expect("save pull state");

    run_upload_once_with_progress(
        &cache_root,
        &config,
        &DiffSnapshotTransport::new(Vec::new()).with_server_instance(server_instance),
        |_| {},
    )
    .expect("full snapshot upload");

    let state = crate::sync::state::load_sync_state(&cache_root);
    assert_eq!(
        state,
        crate::sync::state::SyncState {
            integrity_check: None,
            ..pull_state
        }
    );

    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 42,
        truncated: false,
    }])
    .with_server_instance(server_instance);
    run_pull_once_with_progress(&cache_root, &config, &transport, |_| {})
        .expect("incremental peer pull");
    let request = &transport.pull_requests.borrow()[0];
    assert_eq!(request.0, 42);
    assert_eq!(request.1, "workstation");
}

#[test]
fn reconciled_snapshot_rebuilds_remote_cache_without_stale_self_records() {
    let cache_root = unique_temp_dir("snapshot-rebuilds-remote");
    let config = enabled_config("workstation");
    populate_vendor_cache(&cache_root, "claude", "claude:message:response-a");
    crate::data::cache::merge_remote_records(
        &cache_root,
        "workstation",
        vec![remote_usage_record(
            "workstation",
            "claude",
            "fallback:v1:response-a:0",
            "2026-05-18T12:00:00Z",
            10,
        )],
    )
    .expect("seed stale remote cache");
    let mut pull_state = populated_pull_state();
    pull_state.pull_scope = crate::sync::cache_generation::server_scope_fingerprint(&config, None);
    crate::sync::state::save_sync_state(&cache_root, &pull_state).expect("save pull state");

    run_upload_once_with_progress(
        &cache_root,
        &config,
        &DiffSnapshotTransport::new(Vec::new()),
        |_| {},
    )
    .expect("full snapshot upload");

    let canonical = sequenced_remote_record(
        7,
        remote_usage_record(
            "workstation",
            "claude",
            "claude:message:response-a",
            "2026-05-18T12:00:00Z",
            10,
        ),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![canonical],
        max_seq: 7,
        truncated: false,
    }]);
    run_pull_once_with_progress(&cache_root, &config, &transport, |_| {})
        .expect("full pull after reconciliation");

    assert_eq!(transport.pull_requests.borrow()[0].0, 0);
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "claude:message:response-a");
}

#[test]
fn incremental_snapshot_keeps_the_pull_cursor() {
    let cache_root = unique_temp_dir("snapshot-keeps-pull");
    populate_vendor_cache(&cache_root, "claude", "first");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &DiffSnapshotTransport::new(Vec::new()),
        |_| {},
    )
    .expect("initial snapshot upload");
    let pull_state = populated_pull_state();
    crate::sync::state::save_sync_state(&cache_root, &pull_state).expect("save pull state");
    populate_vendor_cache_with_records(
        &cache_root,
        "claude",
        vec![
            usage_record("first", "2026-05-18T12:00:00Z", 10),
            usage_record("second", "2026-05-18T12:01:00Z", 20),
        ],
    );
    let transport = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "second".to_string(),
    }]);

    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("incremental snapshot upload");

    assert!(transport.snapshot_finalizations.borrow().is_empty());
    assert_eq!(crate::sync::state::load_sync_state(&cache_root), pull_state);
}

#[test]
fn snapshot_upload_does_not_materialize_indexed_vendor_caches() {
    let cache_root = unique_temp_dir("snapshot-streaming-cache");
    populate_vendor_cache_with_records(
        &cache_root,
        "claude",
        vec![
            usage_record("first", "2026-05-18T12:00:00Z", 10),
            usage_record("second", "2026-05-18T12:01:00Z", 20),
        ],
    );
    let transport = DiffSnapshotTransport::new(vec![
        RecordKey {
            vendor: "claude".to_string(),
            dedup_key: "first".to_string(),
        },
        RecordKey {
            vendor: "claude".to_string(),
            dedup_key: "second".to_string(),
        },
    ]);
    crate::data::cache::reset_cached_record_reads();

    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("streaming snapshot upload");

    assert_eq!(crate::data::cache::cached_record_reads(), 0);
    assert_eq!(transport.snapshot_record_batches.borrow().len(), 1);
    assert_eq!(
        transport.snapshot_record_batches.borrow()[0].records.len(),
        2
    );
}

#[test]
fn snapshot_upload_retries_when_the_cache_changes_between_scans() {
    let cache_root = unique_temp_dir("snapshot-generation-race");
    populate_vendor_cache(&cache_root, "claude", "first");
    let changed_root = cache_root.clone();
    let transport = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }])
    .with_after_first_diff(move || {
        populate_vendor_cache_with_records(
            &changed_root,
            "claude",
            vec![usage_record("first", "2026-05-18T12:00:00Z", 99)],
        );
    });

    let error = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect_err("cache mutation must retry");

    assert!(error.to_string().contains("cached records changed"));
    assert!(transport.snapshot_record_batches.borrow().is_empty());
    assert!(transport.snapshot_finalizations.borrow().is_empty());
}

#[test]
fn snapshot_upload_commits_the_captured_generation_while_new_records_arrive() {
    let cache_root = unique_temp_dir("snapshot-live-cache");
    populate_vendor_cache(&cache_root, "claude", "first");
    let captured_generation =
        crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS);
    let changed_root = cache_root.clone();
    let transport = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }])
    .with_after_first_diff(move || {
        populate_vendor_cache_with_records(
            &changed_root,
            "claude",
            vec![
                usage_record("first", "2026-05-18T12:00:00Z", 10),
                usage_record("second", "2026-05-18T12:01:00Z", 20),
            ],
        );
    });

    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("captured snapshot upload");

    assert_eq!(transport.snapshot_finalizations.borrow().len(), 1);
    assert_eq!(transport.snapshot_record_batches.borrow().len(), 1);
    assert_eq!(
        transport.snapshot_record_batches.borrow()[0].records[0].dedup_key,
        "first"
    );
    let completed = crate::sync::state::load_snapshot_upload_state(&cache_root);
    assert_eq!(completed.cache_generation, captured_generation);
    assert_ne!(
        completed.cache_generation,
        crate::sync::cache_generation::local_cache_generation(&cache_root, &VENDORS)
    );

    let catch_up = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "second".to_string(),
    }]);
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &catch_up,
        |_| {},
    )
    .expect("incremental catch-up");

    assert_eq!(catch_up.snapshot_diffs.borrow().len(), 1);
    assert_eq!(catch_up.snapshot_diffs.borrow()[0].records.len(), 1);
    assert_eq!(catch_up.snapshot_record_batches.borrow().len(), 1);
    assert_eq!(
        catch_up.snapshot_record_batches.borrow()[0].records[0].dedup_key,
        "second"
    );
    assert!(catch_up.snapshot_finalizations.borrow().is_empty());
}

#[test]
fn snapshot_finalize_retry_resumes_without_replaying_manifest() {
    let cache_root = unique_temp_dir("snapshot-finalize-resume");
    let config = enabled_config("workstation");
    populate_vendor_cache(&cache_root, "claude", "first");
    let mut pull_state = populated_pull_state();
    pull_state.pull_vendors = pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation");
    pull_state.pull_scope = crate::sync::cache_generation::server_scope_fingerprint(&config, None);
    pull_state.integrity_check = Some(crate::sync::state::IntegrityCheckState {
        checked_at: "2026-05-18T12:34:56Z".to_string(),
        range_end_utc: "2026-05-18T00:00:00Z".to_string(),
        checked_hosts: 2,
        sync_scope: "scope-a".to_string(),
    });
    crate::sync::state::save_sync_state(&cache_root, &pull_state).expect("save pull state");
    let first = DiffSnapshotTransport::new(Vec::new()).with_finalize_error("timeout: global");

    let error = run_upload_once_with_progress(&cache_root, &config, &first, |_| {})
        .expect_err("finalize timeout");
    assert_eq!(error.to_string(), "snapshot finalize: timeout: global");
    assert_eq!(first.snapshot_diffs.borrow().len(), 1);
    assert_eq!(first.snapshot_finalizations.borrow().len(), 1);
    assert_eq!(crate::sync::state::load_sync_state(&cache_root), pull_state);
    let snapshot_id = first.snapshot_finalizations.borrow()[0].snapshot_id.clone();

    let retry = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(&cache_root, &config, &retry, |_| {}).expect("resume finalize");

    assert!(retry.snapshot_diffs.borrow().is_empty());
    assert!(retry.snapshot_record_batches.borrow().is_empty());
    assert_eq!(retry.snapshot_finalizations.borrow().len(), 1);
    assert_eq!(
        retry.snapshot_finalizations.borrow()[0].snapshot_id,
        snapshot_id
    );
    assert_eq!(
        crate::sync::state::load_sync_state(&cache_root),
        crate::sync::state::SyncState {
            integrity_check: None,
            ..pull_state
        }
    );
}

#[test]
fn superseded_snapshot_finalize_discards_the_pending_attempt() {
    let cache_root = unique_temp_dir("snapshot-finalize-superseded");
    populate_vendor_cache(&cache_root, "claude", "first");
    let superseded = DiffSnapshotTransport::new(Vec::new())
        .with_finalize_error("http status: 409: snapshot attempt is not current");

    let error = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &superseded,
        |_| {},
    )
    .expect_err("superseded finalize");
    assert_eq!(error.to_string(), "snapshot finalize: superseded");
    let stale_id = superseded.snapshot_finalizations.borrow()[0]
        .snapshot_id
        .clone();

    let retry = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &retry, |_| {})
        .expect("retry superseded snapshot");

    assert_eq!(retry.snapshot_diffs.borrow().len(), 1);
    assert_ne!(retry.snapshot_diffs.borrow()[0].snapshot_id, stale_id);
}

#[test]
fn superseded_snapshot_diff_discards_the_attempt() {
    let cache_root = unique_temp_dir("snapshot-diff-superseded");
    populate_vendor_cache(&cache_root, "claude", "first");
    let superseded = DiffSnapshotTransport::new(Vec::new())
        .with_diff_error("http status: 409: snapshot attempt is not current");

    let error = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &superseded,
        |_| {},
    )
    .expect_err("superseded diff");
    assert_eq!(error.to_string(), "snapshot diff: superseded");
    let stale_id = superseded.snapshot_diffs.borrow()[0].snapshot_id.clone();

    let retry = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &retry, |_| {})
        .expect("retry superseded snapshot");

    assert_ne!(retry.snapshot_diffs.borrow()[0].snapshot_id, stale_id);
}

#[test]
fn superseded_snapshot_record_batch_discards_the_attempt() {
    let cache_root = unique_temp_dir("snapshot-records-superseded");
    populate_vendor_cache(&cache_root, "claude", "first");
    let key = RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    };
    let superseded = DiffSnapshotTransport::new(vec![key.clone()])
        .with_record_error("http status: 409: snapshot attempt is not current");

    let error = run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &superseded,
        |_| {},
    )
    .expect_err("superseded record batch");
    assert_eq!(error.to_string(), "snapshot records: superseded");
    let stale_id = superseded.snapshot_diffs.borrow()[0].snapshot_id.clone();

    let retry = DiffSnapshotTransport::new(vec![key]);
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &retry, |_| {})
        .expect("retry superseded snapshot");

    assert_ne!(retry.snapshot_diffs.borrow()[0].snapshot_id, stale_id);
}

#[test]
fn snapshot_diff_retry_reuses_the_incomplete_attempt_id() {
    let cache_root = unique_temp_dir("snapshot-diff-attempt");
    populate_vendor_cache(&cache_root, "claude", "first");
    let first = DiffSnapshotTransport::new(Vec::new()).with_diff_error("timeout: global");

    let error =
        run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &first, |_| {})
            .expect_err("diff timeout");
    assert_eq!(error.to_string(), "snapshot diff: timeout: global");
    let first_snapshot_id = first.snapshot_diffs.borrow()[0].snapshot_id.clone();

    let retry = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &retry, |_| {})
        .expect("retry diff");

    assert_eq!(
        retry.snapshot_diffs.borrow()[0].snapshot_id,
        first_snapshot_id
    );
}

#[test]
fn cache_change_after_finalize_timeout_keeps_the_confirmed_manifest_baseline() {
    let cache_root = unique_temp_dir("snapshot-pending-baseline");
    populate_vendor_cache(&cache_root, "claude", "first");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &DiffSnapshotTransport::new(Vec::new()),
        |_| {},
    )
    .expect("initial upload");

    populate_vendor_cache(&cache_root, "claude", "second");
    let timed_out = DiffSnapshotTransport::new(Vec::new()).with_finalize_error("timeout: global");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &timed_out,
        |_| {},
    )
    .expect_err("finalize timeout");

    populate_vendor_cache_with_records(
        &cache_root,
        "claude",
        vec![
            usage_record("second", "2026-05-18T12:00:00Z", 10),
            usage_record("third", "2026-05-18T12:01:00Z", 20),
        ],
    );
    let changed = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &changed,
        |_| {},
    )
    .expect("reconcile changed cache");

    assert_eq!(changed.snapshot_diffs.borrow().len(), 1);
    assert_eq!(changed.snapshot_diffs.borrow()[0].records.len(), 2);
    assert_eq!(changed.snapshot_finalizations.borrow().len(), 1);
}

#[test]
fn snapshot_cache_does_not_cross_machine_id_changes() {
    let cache_root = unique_temp_dir("snapshot-machine-scope");
    populate_vendor_cache(&cache_root, "claude", "first");
    let initial = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &initial,
        |_| {},
    )
    .expect("initial upload");

    let next = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }]);
    run_upload_once_with_progress(&cache_root, &enabled_config("laptop"), &next, |_| {})
        .expect("upload after machine change");

    assert_eq!(next.snapshot_diffs.borrow().len(), 1);
    assert_eq!(next.snapshot_diffs.borrow()[0].host_id, "laptop");
    assert_eq!(next.snapshot_record_batches.borrow().len(), 1);
    assert_eq!(
        next.snapshot_record_batches.borrow()[0].records[0].host_id,
        "laptop"
    );
}

#[test]
fn snapshot_cache_does_not_cross_project_hash_policy_changes() {
    let cache_root = unique_temp_dir("snapshot-project-hash-scope");
    populate_vendor_cache(&cache_root, "claude", "first");
    let config = enabled_config("workstation");
    let initial = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(&cache_root, &config, &initial, |_| {}).expect("initial upload");

    let mut next_config = config;
    next_config.upload_project_hash = false;
    let next = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }]);
    run_upload_once_with_progress(&cache_root, &next_config, &next, |_| {})
        .expect("upload after project hash policy change");

    assert_eq!(next.snapshot_diffs.borrow().len(), 1);
    assert_eq!(next.snapshot_record_batches.borrow().len(), 1);
    assert!(
        next.snapshot_record_batches.borrow()[0].records[0]
            .project_path_sha256
            .is_none()
    );
}

#[test]
fn snapshot_cache_does_not_cross_server_instance_changes() {
    let cache_root = unique_temp_dir("snapshot-server-scope");
    populate_vendor_cache(&cache_root, "claude", "first");
    let initial = DiffSnapshotTransport::new(Vec::new()).with_server_instance("server-a");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &initial,
        |_| {},
    )
    .expect("initial upload");

    let next = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }])
    .with_server_instance("server-b");
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &next, |_| {})
        .expect("upload after server replacement");

    assert_eq!(next.snapshot_diffs.borrow().len(), 1);
    assert_eq!(next.snapshot_record_batches.borrow().len(), 1);
    assert_ne!(
        initial.snapshot_finalizations.borrow()[0].snapshot_id,
        next.snapshot_finalizations.borrow()[0].snapshot_id
    );
}

#[test]
fn snapshot_cache_reconciles_when_remote_host_state_is_missing() {
    let cache_root = unique_temp_dir("snapshot-remote-missing");
    populate_vendor_cache(&cache_root, "claude", "first");
    let initial = DiffSnapshotTransport::new(Vec::new()).with_server_instance("server-a");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &initial,
        |_| {},
    )
    .expect("initial upload");

    let next = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }])
    .with_server_instance("server-a")
    .with_remote_snapshot_state(RemoteSnapshotState::Missing);
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &next, |_| {})
        .expect("upload after remote state loss");

    assert_eq!(next.snapshot_diffs.borrow().len(), 1);
    assert_eq!(next.snapshot_record_batches.borrow().len(), 1);
}

#[test]
fn snapshot_cache_reconciles_when_remote_content_revision_changes() {
    let cache_root = unique_temp_dir("snapshot-remote-revision");
    populate_vendor_cache(&cache_root, "claude", "first");
    let initial = DiffSnapshotTransport::new(Vec::new())
        .with_server_instance("server-a")
        .with_remote_snapshot_state(RemoteSnapshotState::Present {
            record_count: 1,
            content_revision: Some(10),
        });
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &initial,
        |_| {},
    )
    .expect("initial upload");

    let next = DiffSnapshotTransport::new(vec![RecordKey {
        vendor: "claude".to_string(),
        dedup_key: "first".to_string(),
    }])
    .with_server_instance("server-a")
    .with_remote_snapshot_state(RemoteSnapshotState::Present {
        record_count: 1,
        content_revision: Some(9),
    });
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &next, |_| {})
        .expect("upload after remote rollback");

    assert_eq!(next.snapshot_diffs.borrow().len(), 1);
    assert_eq!(next.snapshot_record_batches.borrow().len(), 1);
    assert_ne!(
        initial.snapshot_finalizations.borrow()[0].snapshot_id,
        next.snapshot_finalizations.borrow()[0].snapshot_id
    );
}

#[test]
fn snapshot_cache_periodically_reconciles_with_a_legacy_server() {
    let cache_root = unique_temp_dir("snapshot-legacy-recheck");
    populate_vendor_cache(&cache_root, "claude", "first");
    let initial = DiffSnapshotTransport::new(Vec::new())
        .with_server_instance("server-a")
        .with_remote_snapshot_state(RemoteSnapshotState::Present {
            record_count: 1,
            content_revision: None,
        });
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &initial,
        |_| {},
    )
    .expect("initial upload");

    let marker_path = cache_root.join("sync_snapshot_marker.json");
    let mut marker: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker_path).expect("read receipt"))
            .expect("parse receipt");
    marker["verified_at_secs"] = serde_json::json!(0);
    std::fs::write(
        marker_path,
        serde_json::to_vec(&marker).expect("serialize expired receipt"),
    )
    .expect("expire receipt");

    let next = DiffSnapshotTransport::new(Vec::new())
        .with_server_instance("server-a")
        .with_remote_snapshot_state(RemoteSnapshotState::Present {
            record_count: 1,
            content_revision: None,
        });
    run_upload_once_with_progress(&cache_root, &enabled_config("workstation"), &next, |_| {})
        .expect("legacy server recheck");

    assert_eq!(next.snapshot_diffs.borrow().len(), 1);
}

#[test]
fn corrupt_rebuildable_snapshot_state_reconciles_on_the_next_cache_change() {
    let cache_root = unique_temp_dir("snapshot-state-rebuild");
    populate_vendor_cache(&cache_root, "claude", "first");
    let initial = DiffSnapshotTransport::new(Vec::new()).with_server_instance("server-a");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &initial,
        |_| {},
    )
    .expect("initial upload");
    std::fs::write(cache_root.join("sync_snapshot_state.bin"), b"corrupt")
        .expect("corrupt rebuildable state");

    let unchanged = DiffSnapshotTransport::new(Vec::new())
        .with_server_instance("server-a")
        .with_remote_snapshot_state(RemoteSnapshotState::Present {
            record_count: 1,
            content_revision: None,
        });
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &unchanged,
        |_| {},
    )
    .expect("unchanged upload");
    assert!(unchanged.snapshot_diffs.borrow().is_empty());

    populate_vendor_cache_with_records(
        &cache_root,
        "claude",
        vec![
            usage_record("first", "2026-05-18T12:00:00Z", 10),
            usage_record("second", "2026-05-18T12:01:00Z", 20),
        ],
    );
    let changed = DiffSnapshotTransport::new(Vec::new()).with_server_instance("server-a");
    run_upload_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &changed,
        |_| {},
    )
    .expect("changed upload");

    assert_eq!(changed.snapshot_diffs.borrow().len(), 1);
    assert_eq!(changed.snapshot_diffs.borrow()[0].records.len(), 2);
    assert_eq!(changed.snapshot_finalizations.borrow().len(), 1);
}

#[test]
fn fingerprint_budget_accounts_for_json_escaping() {
    let cache_root = unique_temp_dir("snapshot-escaped-fingerprint-budget");
    let escaped = "\\\"".repeat(240);
    let records = (0..3_000)
        .map(|index| usage_record(&format!("{index:04}-{escaped}"), "2026-05-18T12:00:00Z", 10))
        .collect();
    populate_vendor_cache_with_records(&cache_root, "claude", records);
    let snapshot = collect_snapshot_manifest(
        &cache_root,
        &enabled_config("workstation"),
        crate::sync::integrity::integrity_range_end_utc(Utc::now()),
    )
    .expect("snapshot");

    for chunk in snapshot_fingerprint_chunks(&snapshot) {
        let body = serde_json::to_vec(&SnapshotDiffRequest {
            host_id: "workstation".to_string(),
            snapshot_id: "workstation:20260723T120000Z".to_string(),
            records: chunk,
        })
        .expect("serialize request");
        assert!(
            body.len() <= 750 * 1024,
            "escaped fingerprint body was {} bytes",
            body.len()
        );
    }
}
