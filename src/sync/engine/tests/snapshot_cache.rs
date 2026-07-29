use super::*;

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
