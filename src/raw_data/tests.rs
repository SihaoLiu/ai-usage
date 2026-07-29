use super::*;
use crate::constants::{AllPricing, SubscriptionFees};
use crate::table_view::TableView;
use chrono::{Duration, TimeZone};
use std::collections::HashMap;

fn test_range(center: DateTime<Local>) -> RawDataRange {
    RawDataRange::from_bounds(
        center - Duration::days(10_000),
        center + Duration::days(10_000),
    )
}

fn state_with_cache(cache: RawDataCache, window: TimeWindow) -> AppState {
    AppState {
        tool: "all".to_string(),
        table_view: TableView::Flat,
        host: None,
        session_id: None,
        local_host_id: None,
        days: 3,
        time_window: window,
        monitor_interval: 3600,
        pricing: AllPricing::load_raw().finalize(),
        subscription_fees: SubscriptionFees::default(),
        fee_env_path: std::path::PathBuf::from(".fee.env"),
        version_cache: HashMap::new(),
        all_tool_prompt: None,
        raw_cache: Some(cache),
        raw_cache_last_used_at: None,
        raw_refresh: None,
        integrity_status: crate::IntegrityStatus::Checked {
            duration: std::time::Duration::ZERO,
        },
        integrity_started_at: None,
    }
}

fn usage_entry_at(timestamp: DateTime<Local>) -> UsageEntry {
    UsageEntry {
        host_id: None,
        session_id: None,
        timestamp: timestamp.to_rfc3339(),
        parsed_timestamp: Some(timestamp),
        session_start_time: String::new(),
        session_end_time: String::new(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        usage: data::TokenUsage::default(),
        costs: None,
    }
}

#[test]
fn weighted_cost_reuses_model_breakdown_totals() {
    let row = |tool: &str, tokens: i64, cost: f64| ModelBreakdownRow {
        model: format!("{tool}-model"),
        tool: tool.to_string(),
        count: 1,
        input: tokens,
        output: 0,
        cache_creation: 0,
        cache_read: 0,
        reasoning: 0,
        thinking: 0,
        total: tokens,
        total_with_cache: tokens,
        input_cost: cost,
        output_cost: 0.0,
        cache_read_cost: 0.0,
        cache_creation_cost: 0.0,
    };
    let rows = [
        row("claude", 1_000_000, 10.0),
        row("codex", 3_000_000, 30.0),
    ];
    let fees = SubscriptionFees {
        claude: 20.0,
        codex: 40.0,
        ..Default::default()
    };

    let (weighted_cost, savings) = calculate_weighted_cost_per_mtok(&rows, 10.0, &fees);

    assert!((weighted_cost - 5.0).abs() < 1e-9);
    assert!((savings - 60.0).abs() < 1e-9);
}

#[test]
fn hot_snapshot_only_covers_windows_inside_its_recent_history() {
    let now = Local
        .with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
        .single()
        .expect("fixed now");
    let current = raw_cache_visible_range(&TimeWindow::rolling_days(3), now);
    let stale = hot_cache_range(now - Duration::days(HOT_CACHE_HORIZON_DAYS + 1));

    assert!(hot_cache_range(now).covers(current));
    assert!(!stale.covers(current));
}

#[test]
fn historical_cache_load_skips_the_hot_snapshot() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let cache_root = std::env::temp_dir().join(format!("ai-usage-cold-range-{stamp}"));
    fs::create_dir_all(&cache_root).expect("create cache root");
    let start = time_utils::parse_timestamp("2020-01-01T00:00:00Z").expect("historical start");
    let range = RawDataRange::from_bounds(start, start + Duration::days(3));
    data::cache::reset_hot_snapshot_reads();

    let _ = read_cached_raw_data_for_window_from(&cache_root, None, None, range);

    assert_eq!(data::cache::hot_snapshot_reads(), 0);
    fs::remove_dir_all(cache_root).expect("remove cache root");
}

#[test]
fn hot_snapshot_round_trip_keeps_only_recent_entries() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = Local
        .with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
        .single()
        .expect("fixed now");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let cache_root = std::env::temp_dir().join(format!("ai-usage-hot-cache-{stamp}"));
    fs::create_dir_all(&cache_root).expect("create cache root");

    let entry = |timestamp: DateTime<Local>, input_tokens| UsageEntry {
        host_id: None,
        session_id: None,
        timestamp: timestamp.to_rfc3339(),
        parsed_timestamp: Some(timestamp),
        session_start_time: String::new(),
        session_end_time: String::new(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        usage: data::TokenUsage {
            input_tokens,
            ..Default::default()
        },
        costs: None,
    };
    let mut source = RawDataCache {
        claude: vec![
            entry(now - Duration::days(HOT_CACHE_HORIZON_DAYS + 1), 99),
            entry(now - Duration::days(1), 11),
        ],
        codex: vec![entry(now - Duration::days(2), 22)],
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: test_range(now),
        has_source_data: true,
        local_host_id: Some("workstation".to_string()),
        local_record_keys: HashMap::new(),
        persistent_generation: crate::sync::cache_generation::raw_data_generation(&cache_root),
        local_session_metadata_current: true,
    };
    sort_raw_cache(&mut source);

    write_hot_raw_snapshot(&cache_root, &source, now).expect("write hot snapshot");
    let required = raw_cache_visible_range(&TimeWindow::rolling_days(3), now);
    let snapshot = load_hot_raw_snapshot(&cache_root, Some("workstation"), required)
        .expect("load hot snapshot");

    assert_eq!(snapshot.claude.len(), 1);
    assert_eq!(snapshot.claude[0].usage.input_tokens, 11);
    assert_eq!(snapshot.codex.len(), 1);
    assert_eq!(snapshot.codex[0].usage.input_tokens, 22);
    assert_eq!(snapshot.range, hot_cache_range(now));
    assert!(snapshot.local_session_metadata_current);
    assert!(load_hot_raw_snapshot(&cache_root, Some("other-host"), required).is_none());

    fs::create_dir_all(cache_root.join("remote")).expect("create remote cache");
    fs::write(cache_root.join("remote").join("laptop.bin"), b"changed")
        .expect("change persistent cache generation");
    assert!(load_hot_raw_snapshot(&cache_root, Some("workstation"), required).is_none());

    fs::remove_dir_all(cache_root).expect("remove cache root");
}

#[test]
fn background_hot_snapshot_is_derived_from_the_resident_cache() {
    let now = Local
        .with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
        .single()
        .expect("fixed now");
    let entry = |timestamp: DateTime<Local>, input_tokens| UsageEntry {
        host_id: None,
        session_id: None,
        timestamp: timestamp.to_rfc3339(),
        parsed_timestamp: Some(timestamp),
        session_start_time: String::new(),
        session_end_time: String::new(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        usage: data::TokenUsage {
            input_tokens,
            ..Default::default()
        },
        costs: None,
    };
    let mut source = RawDataCache {
        claude: Vec::new(),
        codex: vec![
            entry(now - Duration::days(HOT_CACHE_HORIZON_DAYS + 1), 99),
            entry(now - Duration::days(1), 11),
            entry(now + Duration::days(2), 33),
        ],
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: hot_cache_range(now),
        has_source_data: true,
        local_host_id: Some("workstation".to_string()),
        local_record_keys: HashMap::new(),
        persistent_generation: "generation".to_string(),
        local_session_metadata_current: true,
    };
    sort_raw_cache(&mut source);

    let snapshot = prepare_hot_raw_snapshot(&source, now);

    assert_eq!(snapshot.cache.range, hot_cache_range(now));
    assert_eq!(snapshot.cache.codex.len(), 2);
    assert_eq!(snapshot.cache.codex[0].usage.input_tokens, 11);
    assert_eq!(snapshot.cache.codex[1].usage.input_tokens, 33);
    assert_eq!(snapshot.cache.persistent_generation, "generation");
}

#[test]
fn sorted_window_slice_borrows_only_the_requested_range() {
    let now = Local
        .with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
        .single()
        .expect("fixed now");
    let entry = |timestamp: DateTime<Local>| UsageEntry {
        host_id: None,
        session_id: None,
        timestamp: timestamp.to_rfc3339(),
        parsed_timestamp: Some(timestamp),
        session_start_time: String::new(),
        session_end_time: String::new(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        usage: data::TokenUsage::default(),
        costs: None,
    };
    let entries = vec![
        entry(now - Duration::days(9)),
        entry(now - Duration::days(2)),
        entry(now - Duration::days(1)),
    ];

    let visible = sorted_window_slice(&entries, &TimeWindow::rolling_days(3), now);

    assert_eq!(visible.len(), 2);
    assert_eq!(visible.as_ptr(), entries[1..].as_ptr());
}

#[test]
fn background_reload_refreshes_the_hot_snapshot_for_the_next_launch() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let cache_root = std::env::temp_dir().join(format!("ai-usage-hot-reload-{stamp}"));
    fs::create_dir_all(&cache_root).expect("create cache root");
    let now = Local::now();
    data::cache::merge_remote_records(
        &cache_root,
        "laptop",
        vec![data::cache::RemoteUsageRecord {
            vendor: "codex".to_string(),
            dedup_key: "recent-record".to_string(),
            entry: UsageEntry {
                host_id: None,
                session_id: None,
                timestamp: now.to_rfc3339(),
                parsed_timestamp: Some(now),
                session_start_time: String::new(),
                session_end_time: String::new(),
                model: "test-model".to_string(),
                effort: None,
                fast_tier: -1,
                usage: data::TokenUsage {
                    input_tokens: 42,
                    ..Default::default()
                },
                costs: None,
            },
        }],
    )
    .expect("write remote cache");
    let (tx, rx) = mpsc::sync_channel(0);
    let worker_root = cache_root.clone();
    let range = raw_cache_target_range(&TimeWindow::rolling_days(3), now);
    let worker = thread::spawn(move || {
        run_background_raw_load(worker_root, None, None, range, tx);
    });

    let BackgroundRawLoad::Refreshed(cache) = rx.recv().expect("receive resident cache");
    assert_eq!(cache.codex.len(), 1);
    worker.join().expect("join background reload");

    let required = raw_cache_visible_range(&TimeWindow::rolling_days(3), Local::now());
    let snapshot =
        load_hot_raw_snapshot(&cache_root, None, required).expect("load refreshed hot snapshot");
    assert_eq!(snapshot.codex.len(), 1);
    assert_eq!(snapshot.codex[0].usage.input_tokens, 42);

    fs::remove_dir_all(cache_root).expect("remove cache root");
}

#[test]
fn empty_window_cache_keeps_global_source_presence() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let cache_root = std::env::temp_dir().join(format!("ai-usage-empty-window-{stamp}"));
    fs::create_dir_all(&cache_root).expect("create cache root");
    let record_time =
        time_utils::parse_timestamp("2020-01-01T00:00:00Z").expect("record timestamp");
    data::cache::merge_remote_records(
        &cache_root,
        "laptop",
        vec![data::cache::RemoteUsageRecord {
            vendor: "codex".to_string(),
            dedup_key: "outside-window".to_string(),
            entry: UsageEntry {
                host_id: None,
                session_id: None,
                timestamp: record_time.to_rfc3339(),
                parsed_timestamp: Some(record_time),
                session_start_time: String::new(),
                session_end_time: String::new(),
                model: "test-model".to_string(),
                effort: None,
                fast_tier: -1,
                usage: data::TokenUsage::default(),
                costs: None,
            },
        }],
    )
    .expect("write remote cache");
    let start = time_utils::parse_timestamp("2030-01-01T00:00:00Z").expect("window start");
    let range = RawDataRange::from_bounds(start, start + Duration::days(3));

    let cache = load_scoped_persistent_raw_data_from(&cache_root, None, None, range);

    assert!(!raw_cache_has_any_tool_data(&cache));
    assert!(cache.has_source_data);
    fs::remove_dir_all(cache_root).expect("remove cache root");
}

#[test]
fn far_historical_prefetch_stays_bounded_around_the_visible_window() {
    let now = Local
        .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
        .single()
        .expect("fixed now");
    let window = TimeWindow::from_range("2020-01-01", "2020-01-03").expect("range");
    let visible = raw_cache_visible_range(&window, now);
    let target = raw_cache_target_range(&window, now);

    assert!(target.covers(visible));
    assert!(target.end() < now);
    assert!(target.end() - target.start() <= Duration::days(123));
}

#[test]
fn prefetch_refill_keeps_multiple_windows_ahead_of_held_navigation() {
    let now = Local
        .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
        .single()
        .expect("fixed now");
    let window = TimeWindow::from_range("2026-04-01", "2026-04-03").expect("range");
    let target = raw_cache_target_range(&window, now);
    let trigger = raw_cache_trigger_range(&window, now);
    let visible = raw_cache_visible_range(&window, now);
    let span = visible.end() - visible.start();

    assert!(trigger.covers(visible));
    assert!(target.start() <= trigger.start() - span * 4);
    assert!(target.end() >= trigger.end() + span * 4);
}

#[test]
fn large_window_refill_retains_half_a_viewport_of_headroom() {
    let now = Local
        .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
        .single()
        .expect("fixed now");
    let window = TimeWindow::rolling_days(6_000);
    let target = raw_cache_target_range(&window, now);
    let trigger = raw_cache_trigger_range(&window, now);
    let visible = raw_cache_visible_range(&window, now);
    let half_span = (visible.end() - visible.start()) / 2;

    assert!(target.start() <= trigger.start() - half_span);
    assert!(target.end() >= trigger.end() + half_span);
}

#[test]
fn prefetch_runway_exceeds_held_arrow_navigation() {
    let now = Local
        .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
        .single()
        .expect("fixed now");
    for days in [3, 365, 1_000, 6_000] {
        let window = TimeWindow::rolling_days(days);
        let visible = raw_cache_visible_range(&window, now);
        let trigger = raw_cache_trigger_range(&window, now);
        let arrow_step = Duration::minutes(crate::display_interval_minutes_for_window(
            &window, now, 160,
        ));
        let runway = visible.start() - trigger.start();

        assert!(
            runway >= arrow_step * 128,
            "days={days}, runway_steps={}",
            runway.num_minutes() / arrow_step.num_minutes()
        );
    }
}

#[test]
fn background_source_scan_leaves_cpu_capacity_for_interaction() {
    assert_eq!(background_refresh_parallelism(1), 1);
    assert_eq!(background_refresh_parallelism(2), 1);
    assert_eq!(background_refresh_parallelism(4), 1);
    assert_eq!(background_refresh_parallelism(8), 2);
    assert_eq!(background_refresh_parallelism(64), 4);
}

#[test]
fn resident_window_reuses_cache_storage_without_cloning() {
    let now = Local::now();
    let entry = |timestamp: DateTime<Local>| UsageEntry {
        host_id: None,
        session_id: None,
        timestamp: timestamp.to_rfc3339(),
        parsed_timestamp: Some(timestamp),
        session_start_time: String::new(),
        session_end_time: String::new(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        usage: data::TokenUsage::default(),
        costs: None,
    };
    let cache = RawDataCache {
        claude: vec![
            entry(now - Duration::days(5)),
            entry(now - Duration::days(1)),
        ],
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: hot_cache_range(now),
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let state = state_with_cache(cache, TimeWindow::rolling_days(3));
    let cached_ptr = std::ptr::from_ref(&state.raw_cache.as_ref().unwrap().claude[1]);

    let all_data = load_resident_all_tool_data(&state, now);

    assert_eq!(all_data.claude.len(), 1);
    assert_eq!(all_data.claude.as_ptr(), cached_ptr);
}

#[test]
fn idle_historical_cache_keeps_the_adjacent_window_on_each_side() {
    let now = Local
        .with_ymd_and_hms(2026, 7, 28, 12, 0, 0)
        .single()
        .expect("fixed now");
    let window = TimeWindow::ExplicitRange {
        start: now - Duration::days(10),
        end: now - Duration::days(7),
        projection_days: 3.0,
        page_step: Duration::days(3),
    };
    let cache = RawDataCache {
        claude: [-14, -13, -10, -7, -4, -3]
            .into_iter()
            .map(|days| usage_entry_at(now + Duration::days(days)))
            .collect(),
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: test_range(now),
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let mut state = state_with_cache(cache, window);
    let accessed_at = std::time::Instant::now();

    touch_raw_cache_at(&mut state, accessed_at);

    assert!(!retire_idle_raw_cache_at(
        &mut state,
        accessed_at + RAW_CACHE_IDLE_TTL - std::time::Duration::from_millis(1),
        now,
    ));
    assert!(retire_idle_raw_cache_at(
        &mut state,
        accessed_at + RAW_CACHE_IDLE_TTL,
        now,
    ));

    let cache = state
        .raw_cache
        .as_ref()
        .expect("adjacent windows remain hot");
    assert_eq!(cache.range.start(), now - Duration::days(13));
    assert_eq!(cache.range.end(), now - Duration::days(4));
    assert_eq!(
        cache
            .claude
            .iter()
            .map(|entry| entry_timestamp(entry).expect("entry timestamp"))
            .collect::<Vec<_>>(),
        [-13, -10, -7, -4]
            .into_iter()
            .map(|days| now + Duration::days(days))
            .collect::<Vec<_>>()
    );
    assert!(state.raw_cache_last_used_at.is_none());
}

#[test]
fn idle_latest_cache_keeps_only_the_previous_window_and_ages_it_forward() {
    let now = Local
        .with_ymd_and_hms(2026, 7, 28, 12, 0, 0)
        .single()
        .expect("fixed now");
    let cache = RawDataCache {
        claude: [-7, -6, -3, 0, 1]
            .into_iter()
            .map(|days| usage_entry_at(now + Duration::days(days)))
            .collect(),
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: test_range(now),
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let mut state = state_with_cache(cache, TimeWindow::rolling_days(3));
    let accessed_at = std::time::Instant::now();

    touch_raw_cache_at(&mut state, accessed_at);
    assert!(retire_idle_raw_cache_at(
        &mut state,
        accessed_at + RAW_CACHE_IDLE_TTL,
        now,
    ));

    let cache = state
        .raw_cache
        .as_ref()
        .expect("previous window remains hot");
    assert_eq!(cache.range.start(), now - Duration::days(6));
    assert_eq!(cache.range.end(), now);
    assert_eq!(cache.claude.len(), 3);

    touch_raw_cache_at(&mut state, accessed_at + RAW_CACHE_IDLE_TTL);
    assert!(retire_idle_raw_cache_at(
        &mut state,
        accessed_at + RAW_CACHE_IDLE_TTL * 2,
        now + Duration::days(1),
    ));

    let cache = state.raw_cache.as_ref().expect("rolling cache remains hot");
    assert_eq!(cache.range.start(), now - Duration::days(5));
    assert_eq!(cache.range.end(), now);
    assert_eq!(
        cache
            .claude
            .iter()
            .map(|entry| entry_timestamp(entry).expect("entry timestamp"))
            .collect::<Vec<_>>(),
        [-3, 0]
            .into_iter()
            .map(|days| now + Duration::days(days))
            .collect::<Vec<_>>()
    );
}

#[test]
fn dashboard_build_does_not_count_as_cache_activity() {
    let now = Local::now();
    let cache = RawDataCache {
        claude: Vec::new(),
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: hot_cache_range(now),
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let mut state = state_with_cache(cache, TimeWindow::rolling_days(3));
    assert!(state.raw_cache_last_used_at.is_none());

    let _dashboard = crate::tui::data::build(&mut state);

    assert!(state.raw_cache_last_used_at.is_none());
}

#[test]
fn cache_activity_is_recorded_before_the_cache_arrives() {
    let now = Local::now();
    let cache = RawDataCache {
        claude: Vec::new(),
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: hot_cache_range(now),
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let mut state = state_with_cache(cache, TimeWindow::rolling_days(3));
    state.raw_cache = None;

    touch_raw_cache_at(&mut state, std::time::Instant::now());

    assert!(state.raw_cache_last_used_at.is_some());
}

fn background_cache(now: DateTime<Local>) -> RawDataCache {
    RawDataCache {
        claude: vec![
            usage_entry_at(now - Duration::days(30)),
            usage_entry_at(now - Duration::days(1)),
        ],
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: test_range(now),
        has_source_data: true,
        local_host_id: Some("workstation".to_string()),
        local_record_keys: HashMap::from([(Tool::Claude, HashSet::from(["key".to_string()]))]),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    }
}

#[test]
fn active_background_completion_keeps_the_prefetch_range() {
    let now = Local::now();
    let cache = background_cache(now);
    let (tx, rx) = mpsc::channel();
    tx.send(BackgroundRawLoad::Refreshed(Box::new(cache)))
        .expect("send cache");
    let mut state = state_with_cache(background_cache(now), TimeWindow::rolling_days(3));
    state.raw_cache = None;
    state.raw_cache_last_used_at = Some(std::time::Instant::now());
    state.raw_refresh = Some(rx);
    state.integrity_status = crate::IntegrityStatus::Checking { percent: 0 };

    assert!(poll_background_raw_refresh(&mut state));
    assert_eq!(
        state.integrity_status,
        crate::IntegrityStatus::Checking { percent: 0 }
    );

    let cache = state.raw_cache.expect("foreground cache");
    assert_eq!(cache.range, test_range(now));
    assert_eq!(cache.claude.len(), 2);
    assert_eq!(cache.local_record_keys.len(), 1);

    state.raw_cache = Some(cache);
}

#[test]
fn idle_background_completion_keeps_only_the_retention_range() {
    let now = Local::now();
    let expected = idle_raw_cache_retention_range(&TimeWindow::rolling_days(3), now);
    let cache = background_cache(now);
    let (tx, rx) = mpsc::channel();
    tx.send(BackgroundRawLoad::Refreshed(Box::new(cache)))
        .expect("send cache");
    let mut state = state_with_cache(background_cache(now), TimeWindow::rolling_days(3));
    state.raw_cache = None;
    state.raw_refresh = Some(rx);

    assert!(poll_background_raw_refresh(&mut state));

    let cache = state.raw_cache.expect("retained cache");
    assert_eq!(
        cache.range.end() - cache.range.start(),
        expected.end() - expected.start()
    );
    assert!(
        (cache.range.end() - expected.end())
            .num_milliseconds()
            .abs()
            < 100
    );
    assert_eq!(cache.claude.len(), 1);
    assert!(state.raw_cache_last_used_at.is_none());
}

#[test]
fn idle_background_replacement_reclaims_after_the_old_cache_drops() {
    let now = Local::now();
    let desired = idle_raw_cache_retention_range(&TimeWindow::rolling_days(3), now);
    let incoming_range = RawDataRange::from_bounds(
        desired.start() - Duration::minutes(1),
        desired.end() + Duration::minutes(1),
    );
    let incoming = RawDataCache {
        claude: vec![usage_entry_at(now - Duration::days(1))],
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: incoming_range,
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let (tx, rx) = mpsc::channel();
    tx.send(BackgroundRawLoad::Refreshed(Box::new(incoming)))
        .expect("send cache");
    let mut state = state_with_cache(background_cache(now), TimeWindow::rolling_days(3));
    state.raw_refresh = Some(rx);
    IDLE_RECLAIM_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);

    assert!(poll_background_raw_refresh(&mut state));

    for _ in 0..100 {
        if IDLE_RECLAIM_CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(IDLE_RECLAIM_CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

#[test]
fn undersized_background_result_preserves_resident_prefetch_headroom() {
    let now = Local::now();
    let visible = raw_cache_visible_range(&TimeWindow::rolling_days(3), now);
    let resident_range = raw_cache_target_range(&TimeWindow::rolling_days(3), now);
    let stale_range = RawDataRange::from_bounds(
        visible.start() - Duration::days(100),
        visible.start() - Duration::days(90),
    );
    let cache = |range| RawDataCache {
        claude: Vec::new(),
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range,
        has_source_data: false,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    let mut state = state_with_cache(cache(resident_range), TimeWindow::rolling_days(3));
    state.raw_cache_last_used_at = Some(std::time::Instant::now());
    let (tx, rx) = mpsc::channel();
    tx.send(BackgroundRawLoad::Refreshed(Box::new(cache(stale_range))))
        .expect("send cache");
    state.raw_refresh = Some(rx);

    assert!(poll_background_raw_refresh(&mut state));
    assert_eq!(state.raw_cache.as_ref().unwrap().range, resident_range);
}

#[test]
fn window_data_state_distinguishes_empty_window_from_missing_source_data() {
    assert_eq!(
        classify_window_data(false, false),
        WindowDataState::NoSourceData
    );
    assert_eq!(
        classify_window_data(true, false),
        WindowDataState::EmptyWindow
    );
    assert_eq!(classify_window_data(true, true), WindowDataState::Populated);
}

#[test]
fn host_filter_includes_local_only_when_machine_matches() {
    assert!(include_local_for_host_filter(None, None));
    assert!(include_local_for_host_filter(
        Some("workstation"),
        Some("workstation")
    ));
    assert!(!include_local_for_host_filter(
        Some("laptop"),
        Some("workstation")
    ));
    assert!(!include_local_for_host_filter(Some("laptop"), None));
}

#[test]
fn remote_records_merge_into_tool_buckets() {
    let timestamp = "2026-05-18T12:00:00Z";
    let center = time_utils::parse_timestamp(timestamp).expect("timestamp");
    let mut cache = RawDataCache {
        claude: Vec::new(),
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: test_range(center),
        has_source_data: true,
        local_host_id: None,
        local_record_keys: HashMap::new(),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    merge_remote_records_into_raw_cache(
        &mut cache,
        vec![
            data::cache::RemoteUsageRecord {
                vendor: "claude".to_string(),
                dedup_key: "a".to_string(),
                entry: UsageEntry {
                    host_id: Some("laptop".to_string()),
                    session_id: None,
                    timestamp: timestamp.to_string(),
                    parsed_timestamp: time_utils::parse_timestamp(timestamp),
                    session_start_time: timestamp.to_string(),
                    session_end_time: timestamp.to_string(),
                    model: "model-a".to_string(),
                    effort: None,
                    fast_tier: -1,
                    usage: data::TokenUsage::default(),
                    costs: None,
                },
            },
            data::cache::RemoteUsageRecord {
                vendor: "codex".to_string(),
                dedup_key: "b".to_string(),
                entry: UsageEntry {
                    host_id: Some("laptop".to_string()),
                    session_id: None,
                    timestamp: timestamp.to_string(),
                    parsed_timestamp: time_utils::parse_timestamp(timestamp),
                    session_start_time: timestamp.to_string(),
                    session_end_time: timestamp.to_string(),
                    model: "model-b".to_string(),
                    effort: None,
                    fast_tier: -1,
                    usage: data::TokenUsage::default(),
                    costs: None,
                },
            },
        ],
    );

    assert_eq!(cache.claude.len(), 1);
    assert_eq!(cache.codex.len(), 1);
    assert!(cache.gemini.is_empty());
}

#[test]
fn remote_records_from_local_host_fill_missing_keys_without_duplicates() {
    let timestamp = "2026-05-18T12:00:00Z";
    let center = time_utils::parse_timestamp(timestamp).expect("timestamp");
    let mut cache = RawDataCache {
        claude: vec![UsageEntry {
            host_id: None,
            session_id: None,
            timestamp: timestamp.to_string(),
            parsed_timestamp: time_utils::parse_timestamp(timestamp),
            session_start_time: timestamp.to_string(),
            session_end_time: timestamp.to_string(),
            model: "local-model".to_string(),
            effort: None,
            fast_tier: -1,
            usage: data::TokenUsage::default(),
            costs: None,
        }],
        codex: Vec::new(),
        gemini: Vec::new(),
        kimi: Vec::new(),
        omp: Vec::new(),
        range: test_range(center),
        has_source_data: true,
        local_host_id: Some("laptop".to_string()),
        local_record_keys: HashMap::from([(Tool::Claude, HashSet::from(["a".to_string()]))]),
        persistent_generation: String::new(),
        local_session_metadata_current: true,
    };
    merge_remote_records_into_raw_cache(
        &mut cache,
        vec![
            data::cache::RemoteUsageRecord {
                vendor: "claude".to_string(),
                dedup_key: "a".to_string(),
                entry: UsageEntry {
                    host_id: Some("laptop".to_string()),
                    session_id: None,
                    timestamp: timestamp.to_string(),
                    parsed_timestamp: time_utils::parse_timestamp(timestamp),
                    session_start_time: timestamp.to_string(),
                    session_end_time: timestamp.to_string(),
                    model: "duplicate-model".to_string(),
                    effort: None,
                    fast_tier: -1,
                    usage: data::TokenUsage::default(),
                    costs: None,
                },
            },
            data::cache::RemoteUsageRecord {
                vendor: "claude".to_string(),
                dedup_key: "b".to_string(),
                entry: UsageEntry {
                    host_id: Some("laptop".to_string()),
                    session_id: None,
                    timestamp: timestamp.to_string(),
                    parsed_timestamp: time_utils::parse_timestamp(timestamp),
                    session_start_time: timestamp.to_string(),
                    session_end_time: timestamp.to_string(),
                    model: "missing-model".to_string(),
                    effort: None,
                    fast_tier: -1,
                    usage: data::TokenUsage::default(),
                    costs: None,
                },
            },
        ],
    );

    assert_eq!(cache.claude.len(), 2);
    assert!(
        cache
            .claude
            .iter()
            .any(|entry| entry.model == "local-model")
    );
    assert!(
        cache
            .claude
            .iter()
            .any(|entry| entry.model == "missing-model")
    );
    assert!(
        !cache
            .claude
            .iter()
            .any(|entry| entry.model == "duplicate-model")
    );
}
