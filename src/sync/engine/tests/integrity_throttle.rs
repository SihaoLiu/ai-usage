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
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(&config, None),
            last_full_pull: Some(Utc::now().to_rfc3339()),
            full_pull_in_progress: false,
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
fn recent_full_pull_keeps_integrity_failure_without_refetching() {
    let cache_root = unique_temp_dir("integrity-repair-recent-pull");
    let owner_cache = unique_temp_dir("integrity-repair-recent-pull-owner");
    let correct = remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
    let stale = remote_usage_record("laptop", "claude", "stale-a", "2000-01-01T00:00:00Z", 10);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![stale])
        .expect("seed stale cache");
    crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![correct.clone()])
        .expect("seed owner cache");
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
            integrity_check: Some(crate::sync::state::IntegrityCheckState {
                checked_at: Utc::now().to_rfc3339(),
                range_end_utc: crate::sync::integrity::integrity_range_end_utc(Utc::now())
                    .to_rfc3339(),
                checked_hosts: 1,
                sync_scope: crate::sync::cache_generation::sync_scope_fingerprint(
                    &enabled_config("workstation"),
                    None,
                ),
            }),
        },
    )
    .expect("save recent full pull");
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        crate::sync::integrity::integrity_range_end_utc(Utc::now()),
        Utc::now(),
    )
    .expect("owner report");
    let transport = FakeTransport::new_with_integrity(
        vec![PullResponse {
            records: Vec::new(),
            max_seq: 10,
            truncated: false,
        }],
        vec![owner_report],
    );
    let mut events = Vec::new();

    run_integrity_once_with_repair(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("integrity check");

    assert!(transport.pull_requests.borrow().is_empty());
    let integrity_events = events
        .iter()
        .filter_map(|event| match event {
            SyncProgress::IntegrityCheckFinished { verification } => Some(verification),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(integrity_events.len(), 1);
    assert!(matches!(
        integrity_events[0],
        crate::sync::integrity::IntegrityVerification::Failed { .. }
    ));
    let remote = crate::data::cache::load_remote_entries(&cache_root, None);
    assert_eq!(remote.len(), 1);
    assert_eq!(remote[0].dedup_key, "stale-a");

    assert!(
        crate::sync::state::load_sync_state(&cache_root)
            .integrity_check
            .is_none()
    );
    events.clear();
    run_sync_cycle_with_progress(
        &cache_root,
        &enabled_config("workstation"),
        &transport,
        |event| events.push(event.clone()),
    )
    .expect("following sync cycle");
    assert!(
        events
            .iter()
            .all(|event| !matches!(event, SyncProgress::IntegrityCheckReused { .. }))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        SyncProgress::IntegrityCheckFinished {
            verification: crate::sync::integrity::IntegrityVerification::Failed { .. }
        }
    )));
}

#[test]
fn recent_full_pull_only_throttles_current_complete_identity() {
    for (name, scope_matches, vendors_match) in
        [("stale-scope", false, true), ("stale-vendors", true, false)]
    {
        let cache_root = unique_temp_dir(name);
        let owner_cache = unique_temp_dir(&format!("{name}-owner"));
        let config = enabled_config("workstation");
        let correct =
            remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
        let stale = remote_usage_record("laptop", "claude", "stale-a", "2000-01-01T00:00:00Z", 10);
        crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![stale])
            .expect("seed stale cache");
        crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![correct.clone()])
            .expect("seed owner cache");
        crate::sync::state::save_sync_state(
            &cache_root,
            &crate::sync::state::SyncState {
                schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
                last_seen_seq: 5,
                pull_vendors: if vendors_match {
                    pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation")
                } else {
                    pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS, "workstation")
                },
                pull_scope: if scope_matches {
                    crate::sync::cache_generation::server_scope_fingerprint(&config, None)
                } else {
                    "stale-scope".to_string()
                },
                last_full_pull: Some(Utc::now().to_rfc3339()),
                full_pull_in_progress: false,
                last_successful_sync: None,
                last_error: None,
                integrity_check: None,
            },
        )
        .expect("save pull identity");
        let owner_report = crate::sync::integrity::build_remote_report_at(
            &owner_cache,
            "laptop",
            crate::sync::integrity::integrity_range_end_utc(Utc::now()),
            Utc::now(),
        )
        .expect("owner report");
        let transport = FakeTransport::new_with_integrity(
            vec![PullResponse {
                records: vec![sequenced_remote_record(5, correct)],
                max_seq: 5,
                truncated: false,
            }],
            vec![owner_report],
        );
        let mut events = Vec::new();

        run_integrity_once_with_repair(&cache_root, &config, &transport, |event| {
            events.push(event.clone())
        })
        .expect("integrity repair");

        let pull_seqs = transport
            .pull_requests
            .borrow()
            .iter()
            .map(|request| request.0)
            .collect::<Vec<_>>();
        assert_eq!(pull_seqs, vec![0], "case {name}");
        let integrity_events = events
            .iter()
            .filter_map(|event| match event {
                SyncProgress::IntegrityCheckFinished { verification } => Some(verification),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(integrity_events.len(), 2, "case {name}");
        assert!(matches!(
            integrity_events[0],
            crate::sync::integrity::IntegrityVerification::Failed { .. }
        ));
        assert!(matches!(
            integrity_events[1],
            crate::sync::integrity::IntegrityVerification::Checked { checked_hosts: 1 }
        ));
    }
}

#[test]
fn integrity_repair_resumes_a_current_partial_full_pull() {
    let cache_root = unique_temp_dir("integrity-resume-partial-pull");
    let owner_cache = unique_temp_dir("integrity-resume-partial-pull-owner");
    let config = enabled_config("workstation");
    let first = remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
    let second = remote_usage_record("laptop", "claude", "remote-b", "2000-01-01T00:01:00Z", 20);
    crate::data::cache::merge_remote_records(&cache_root, "laptop", vec![first.clone()])
        .expect("seed partial cache");
    crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![first, second.clone()])
        .expect("seed owner cache");
    let previous_full_pull = (Utc::now() - Duration::hours(7)).to_rfc3339();
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 5,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(&config, None),
            last_full_pull: Some(previous_full_pull.clone()),
            full_pull_in_progress: true,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        },
    )
    .expect("save partial pull state");
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        crate::sync::integrity::integrity_range_end_utc(Utc::now()),
        Utc::now(),
    )
    .expect("owner report");
    let transport = FakeTransport::new_with_integrity(
        vec![PullResponse {
            records: vec![sequenced_remote_record(6, second)],
            max_seq: 6,
            truncated: false,
        }],
        vec![owner_report],
    );
    let mut events = Vec::new();

    run_integrity_once_with_repair(&cache_root, &config, &transport, |event| {
        events.push(event.clone())
    })
    .expect("integrity repair");

    let pull_seqs = transport
        .pull_requests
        .borrow()
        .iter()
        .map(|request| request.0)
        .collect::<Vec<_>>();
    assert_eq!(pull_seqs, vec![5]);
    let completed_state = crate::sync::state::load_sync_state(&cache_root);
    assert!(!completed_state.full_pull_in_progress);
    assert_ne!(
        completed_state.last_full_pull.as_deref(),
        Some(previous_full_pull.as_str())
    );
    let integrity_events = events
        .iter()
        .filter_map(|event| match event {
            SyncProgress::IntegrityCheckFinished { verification } => Some(verification),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(integrity_events.len(), 2);
    assert!(matches!(
        integrity_events[0],
        crate::sync::integrity::IntegrityVerification::Failed { .. }
    ));
    assert!(matches!(
        integrity_events[1],
        crate::sync::integrity::IntegrityVerification::Checked { checked_hosts: 1 }
    ));
}

#[test]
fn degraded_integrity_repair_does_not_recheck_or_clear_on_the_next_attempt() {
    let cache_root = unique_temp_dir("integrity-degraded-repair");
    let owner_cache = unique_temp_dir("integrity-degraded-repair-owner");
    let config = enabled_config("workstation");
    let retained = remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
    let unavailable =
        remote_usage_record("laptop", "kimi", "remote-kimi", "2000-01-01T00:01:00Z", 20);
    crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![unavailable])
        .expect("seed owner cache");
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 9,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(&config, None),
            last_full_pull: Some((Utc::now() - Duration::hours(7)).to_rfc3339()),
            full_pull_in_progress: false,
            last_successful_sync: None,
            last_error: None,
            integrity_check: None,
        },
    )
    .expect("save stale pull receipt");
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        crate::sync::integrity::integrity_range_end_utc(Utc::now()),
        Utc::now(),
    )
    .expect("owner report");
    let transport = FakeTransport {
        reject_pull_vendor: Some("kimi"),
        ..FakeTransport::new_with_integrity(
            vec![
                PullResponse {
                    records: vec![sequenced_remote_record(5, retained)],
                    max_seq: 5,
                    truncated: false,
                },
                PullResponse {
                    records: Vec::new(),
                    max_seq: 5,
                    truncated: false,
                },
            ],
            vec![owner_report],
        )
    };

    for expected_requests in [vec![0, 0], vec![0, 0, 0, 5]] {
        let mut events = Vec::new();
        run_integrity_once_with_repair(&cache_root, &config, &transport, |event| {
            events.push(event.clone())
        })
        .expect("degraded integrity repair");

        let pull_seqs = transport
            .pull_requests
            .borrow()
            .iter()
            .map(|request| request.0)
            .collect::<Vec<_>>();
        assert_eq!(pull_seqs, expected_requests);
        let integrity_events = events
            .iter()
            .filter(|event| matches!(event, SyncProgress::IntegrityCheckFinished { .. }))
            .count();
        assert_eq!(integrity_events, 1);
        let remote = crate::data::cache::load_remote_entries(&cache_root, None);
        assert_eq!(remote.len(), 1);
        assert_eq!(remote[0].dedup_key, "remote-a");
    }
}

#[test]
fn full_pull_interval_uses_elapsed_time_across_offset_changes() {
    let spring_now = DateTime::parse_from_rfc3339("2026-03-08T03:30:00-07:00")
        .expect("spring timestamp")
        .with_timezone(&Utc);
    assert!(!full_pull_due(
        Some("2026-03-08T01:30:00-08:00"),
        spring_now,
    ));

    let fall_now = DateTime::parse_from_rfc3339("2026-11-01T05:30:00-08:00")
        .expect("fall timestamp")
        .with_timezone(&Utc);
    assert!(full_pull_due(Some("2026-11-01T00:30:00-07:00"), fall_now));
}

#[test]
fn resumed_full_pull_records_completion_before_integrity_check() {
    let cache_root = unique_temp_dir("resumed-full-pull");
    let owner_cache = unique_temp_dir("resumed-full-pull-owner");
    let config = enabled_config("workstation");
    let correct = remote_usage_record("laptop", "claude", "remote-a", "2000-01-01T00:00:00Z", 10);
    let stale = remote_usage_record("laptop", "claude", "stale-a", "2000-01-01T00:00:00Z", 10);
    crate::data::cache::merge_remote_records(&owner_cache, "laptop", vec![correct.clone()])
        .expect("seed owner cache");
    let previous_full_pull = (Utc::now() - Duration::hours(7)).to_rfc3339();
    crate::sync::state::save_sync_state(
        &cache_root,
        &crate::sync::state::SyncState {
            schema_version: crate::sync::state::SYNC_STATE_SCHEMA_VERSION,
            last_seen_seq: 99,
            pull_vendors: pull_state_fingerprint_for(&SUPPORTED_PULL_VENDORS, "workstation"),
            pull_scope: crate::sync::cache_generation::server_scope_fingerprint(&config, None),
            last_full_pull: Some(previous_full_pull.clone()),
            full_pull_in_progress: false,
            last_successful_sync: None,
            last_error: None,
            integrity_check: Some(crate::sync::state::IntegrityCheckState {
                checked_at: Utc::now().to_rfc3339(),
                range_end_utc: crate::sync::integrity::integrity_range_end_utc(Utc::now())
                    .to_rfc3339(),
                checked_hosts: 1,
                sync_scope: crate::sync::cache_generation::sync_scope_fingerprint(&config, None),
            }),
        },
    )
    .expect("save stale full-pull receipt");
    let owner_report = crate::sync::integrity::build_remote_report_at(
        &owner_cache,
        "laptop",
        crate::sync::integrity::integrity_range_end_utc(Utc::now()),
        Utc::now(),
    )
    .expect("owner report");
    let transport = FakeTransport::new_with_integrity(
        vec![
            PullResponse {
                records: vec![sequenced_remote_record(5, stale)],
                max_seq: 5,
                truncated: true,
            },
            PullResponse {
                records: Vec::new(),
                max_seq: 5,
                truncated: false,
            },
            PullResponse {
                records: vec![sequenced_remote_record(5, correct)],
                max_seq: 5,
                truncated: false,
            },
        ],
        vec![owner_report],
    )
    .with_pull_error_on_request(2);

    let first = run_pull_once_with_progress(&cache_root, &config, &transport, |_| {});
    assert_eq!(
        first.expect_err("interrupted pull").to_string(),
        "temporary pull failure"
    );
    let interrupted_state = crate::sync::state::load_sync_state(&cache_root);
    assert!(interrupted_state.full_pull_in_progress);
    assert_eq!(
        interrupted_state.last_full_pull.as_deref(),
        Some(previous_full_pull.as_str())
    );

    let mut events = Vec::new();
    run_sync_cycle_with_progress(&cache_root, &config, &transport, |event| {
        events.push(event.clone())
    })
    .expect("resumed pull");

    let pull_seqs = transport
        .pull_requests
        .borrow()
        .iter()
        .map(|request| request.0)
        .collect::<Vec<_>>();
    assert_eq!(pull_seqs, vec![0, 5, 5]);
    let completed_state = crate::sync::state::load_sync_state(&cache_root);
    assert!(!completed_state.full_pull_in_progress);
    assert_ne!(
        completed_state.last_full_pull.as_deref(),
        Some(previous_full_pull.as_str())
    );
    let integrity_events = events
        .iter()
        .filter_map(|event| match event {
            SyncProgress::IntegrityCheckFinished { verification } => Some(verification),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(integrity_events.len(), 1);
    assert!(matches!(
        integrity_events[0],
        crate::sync::integrity::IntegrityVerification::Failed { .. }
    ));
    assert!(
        events
            .iter()
            .all(|event| !matches!(event, SyncProgress::IntegrityCheckReused { .. }))
    );
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
    sync_state.pull_vendors = pull_state_fingerprint_for(&PREVIOUS_PULL_VENDORS, "workstation");
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
