use super::*;
use chrono::{DateTime, Duration, Utc};

fn seed_empty_snapshot(cache_root: &Path) {
    let transport = DiffSnapshotTransport::new(Vec::new());
    run_upload_once_with_progress(
        cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("seed snapshot state");
}

fn save_recent_integrity_check(cache_root: &Path, range_end_utc: String) {
    let config = enabled_config("workstation");
    crate::sync::state::save_sync_state(
        cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 0,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(&config, None),
            last_full_pull: Some(Utc::now().to_rfc3339()),
            last_successful_sync: None,
            last_error: None,
            integrity_check: Some(crate::sync::state::IntegrityCheckState {
                checked_at: (Utc::now() - Duration::minutes(1)).to_rfc3339(),
                range_end_utc,
                checked_hosts: 3,
                sync_scope: crate::sync::cache_generation::sync_scope_fingerprint(&config, None),
            }),
        },
    )
    .expect("save integrity state");
}

#[test]
fn recent_integrity_check_is_scoped_to_the_sync_identity() {
    let cache_root = unique_temp_dir("integrity-sync-scope");
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    let original = enabled_config("workstation");
    let mut changed = original.clone();
    changed.upload_project_hash = !changed.upload_project_hash;

    assert_eq!(
        reusable_integrity_check(&cache_root, &original, None, Utc::now()),
        Some(3)
    );
    assert_eq!(
        reusable_integrity_check(&cache_root, &changed, None, Utc::now()),
        None
    );
    assert_eq!(
        reusable_integrity_check(&cache_root, &original, Some("replacement"), Utc::now()),
        None
    );
}

#[test]
fn background_cycle_reuses_recent_integrity_check() {
    let cache_root = unique_temp_dir("integrity-reuse");
    seed_empty_snapshot(&cache_root);
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);
    let mut events = Vec::new();

    run_sync_cycle_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("background sync");

    assert!(transport.integrity_submissions.borrow().is_empty());
    assert!(events.contains(&SyncProgress::IntegrityCheckReused { checked_hosts: 3 }));
}

#[test]
fn background_cycle_rechecks_when_utc_range_changes() {
    let cache_root = unique_temp_dir("integrity-new-range");
    seed_empty_snapshot(&cache_root);
    save_recent_integrity_check(
        &cache_root,
        (crate::sync::integrity::integrity_range_end_utc(Utc::now()) - Duration::days(1))
            .to_rfc3339(),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("background sync");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
}

#[test]
fn background_cycle_rechecks_after_pulling_historical_data() {
    let cache_root = unique_temp_dir("integrity-historical-pull");
    seed_empty_snapshot(&cache_root);
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    let pulled = sequenced_remote_record(
        1,
        remote_usage_record("laptop", "claude", "historical", "2000-01-01T00:00:00Z", 10),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: vec![pulled],
        max_seq: 1,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("background sync");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
}

#[test]
fn background_cycle_rechecks_after_uploading_historical_data() {
    let cache_root = unique_temp_dir("integrity-historical-upload");
    seed_empty_snapshot(&cache_root);
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    populate_vendor_cache_with_record(
        &cache_root,
        "claude",
        usage_record("historical", "2000-01-01T00:00:00Z", 10),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("background sync");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
}

#[test]
fn manual_pull_forces_integrity_with_a_recent_check() {
    let cache_root = unique_temp_dir("integrity-manual-force");
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_pull_and_integrity_once_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |_| {},
    )
    .expect("manual pull");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
}

#[test]
fn background_cycle_rechecks_after_six_hours() {
    let cache_root = unique_temp_dir("integrity-expired");
    seed_empty_snapshot(&cache_root);
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    let mut sync_state = crate::sync::state::load_sync_state(&cache_root);
    sync_state
        .integrity_check
        .as_mut()
        .expect("integrity state")
        .checked_at = (Utc::now() - Duration::hours(7)).to_rfc3339();
    crate::sync::state::save_sync_state(&cache_root, &sync_state).expect("save expired state");
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("background sync");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
}

#[test]
fn successful_integrity_check_is_persisted() {
    let cache_root = unique_temp_dir("integrity-persisted");
    seed_empty_snapshot(&cache_root);
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("background sync");

    let check = crate::sync::state::load_sync_state(&cache_root)
        .integrity_check
        .expect("persisted integrity check");
    assert_eq!(check.checked_hosts, 0);
    assert_eq!(
        DateTime::parse_from_rfc3339(&check.range_end_utc)
            .expect("range end")
            .with_timezone(&Utc),
        crate::sync::integrity::integrity_range_end_utc(Utc::now())
    );
}

#[test]
fn background_cycle_rechecks_after_remote_cache_reset() {
    let cache_root = unique_temp_dir("integrity-cache-reset");
    seed_empty_snapshot(&cache_root);
    save_recent_integrity_check(
        &cache_root,
        crate::sync::integrity::integrity_range_end_utc(Utc::now()).to_rfc3339(),
    );
    let mut sync_state = crate::sync::state::load_sync_state(&cache_root);
    sync_state.pull_vendors = pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS);
    crate::sync::state::save_sync_state(&cache_root, &sync_state).expect("save previous vendors");
    crate::data::cache::merge_remote_records(
        &cache_root,
        "laptop",
        vec![remote_usage_record(
            "laptop",
            "claude",
            "historical",
            "2000-01-01T00:00:00Z",
            10,
        )],
    )
    .expect("seed remote cache");
    let transport = FakeTransport::new(vec![PullResponse {
        records: Vec::new(),
        max_seq: 0,
        truncated: false,
    }]);

    run_sync_cycle(&cache_root, &enabled_config("workstation"), &transport)
        .expect("background sync");

    assert_eq!(transport.integrity_submissions.borrow().len(), 1);
}
