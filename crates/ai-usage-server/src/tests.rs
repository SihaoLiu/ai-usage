use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_state(name: &str) -> AppState {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!("ai-usage-write-gate-{name}-{stamp}.db"));
    AppState::new(ServerConfig {
        listen: "127.0.0.1:0".to_string(),
        db_path,
        shared_token: "0123456789abcdef0123456789abcdef".to_string(),
        allowed_hosts: None,
        max_body_bytes: 1_048_576,
        max_batch_records: 1000,
        log_level: "info".to_string(),
        auto_update: AutoUpdateConfig::default(),
    })
    .expect("test state")
}

fn test_record() -> WireRecord {
    WireRecord {
        schema_version: SCHEMA_VERSION,
        host_id: "workstation".to_string(),
        vendor: "codex".to_string(),
        dedup_key: "record-a".to_string(),
        timestamp: "2026-08-04T12:00:00Z".to_string(),
        session_start_time: "2026-08-04T12:00:00Z".to_string(),
        session_end_time: "2026-08-04T12:01:00Z".to_string(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        input_tokens: 10,
        output_tokens: 20,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_creation_5m_input_tokens: 0,
        cache_creation_1h_input_tokens: 0,
        reasoning_output_tokens: 0,
        cost_input: None,
        cost_output: None,
        cost_cache_read: None,
        cost_cache_creation: None,
        project_path_sha256: None,
    }
}

#[tokio::test]
async fn write_gate_serializes_database_writers() {
    let state = test_state("serializes");
    let first = state.write_gate.lock().await;
    let waiting_state = state.clone();
    let second = tokio::spawn(async move {
        let _guard = waiting_state.write_gate.lock().await;
    });
    tokio::task::yield_now().await;

    assert!(!second.is_finished());
    drop(first);
    tokio::time::timeout(Duration::from_secs(1), second)
        .await
        .expect("second writer should proceed")
        .expect("second writer task");
}

#[test]
fn repeated_snapshot_marks_do_not_rewrite_records() {
    let state = test_state("idempotent-snapshot-mark");
    let mut conn = state.pool.get().expect("database connection");
    let tx = conn.transaction().expect("transaction");
    let record = test_record();
    upsert_record(&tx, &record, "2026-08-04T12:01:00Z").expect("insert record");

    let first = mark_snapshot_key(
        &tx,
        &record.host_id,
        &record.vendor,
        &record.dedup_key,
        "snapshot-a",
    )
    .expect("first mark");
    let repeated = mark_snapshot_key(
        &tx,
        &record.host_id,
        &record.vendor,
        &record.dedup_key,
        "snapshot-a",
    )
    .expect("repeated mark");

    assert_eq!(first, 1);
    assert_eq!(repeated, 0);
}

#[test]
fn upsert_outcome_distinguishes_insert_replace_and_noop() {
    let state = test_state("upsert-outcome");
    let mut conn = state.pool.get().expect("database connection");
    let tx = conn.transaction().expect("transaction");
    let record = test_record();

    assert_eq!(
        upsert_record(&tx, &record, "2026-08-04T12:01:00Z").expect("insert record"),
        UpsertOutcome::Inserted
    );
    assert_eq!(
        upsert_record(&tx, &record, "2026-08-04T12:02:00Z").expect("repeat record"),
        UpsertOutcome::Unchanged
    );
    let mut replacement = record;
    replacement.input_tokens += 1;
    assert_eq!(
        upsert_record(&tx, &replacement, "2026-08-04T12:03:00Z").expect("replace record"),
        UpsertOutcome::Replaced
    );
}

#[test]
fn cache_creation_duration_columns_migrate_aggregate_rows() {
    let conn = rusqlite::Connection::open_in_memory().expect("open database");
    conn.execute_batch(
        "CREATE TABLE records (
            schema_version INTEGER NOT NULL,
            cache_creation INTEGER NOT NULL
        )",
    )
    .expect("create legacy records table");
    conn.execute(
        "INSERT INTO records (schema_version, cache_creation) VALUES (?1, ?2)",
        params![SCHEMA_VERSION as i64, 11_i64],
    )
    .expect("insert aggregate row");

    ensure_cache_creation_duration_columns(&conn).expect("migrate duration columns");
    ensure_cache_creation_duration_columns(&conn).expect("migration is idempotent");

    let columns = conn
        .prepare("PRAGMA table_info(records)")
        .expect("prepare table info")
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query table info")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect table info");
    assert!(columns.iter().any(|column| column == "cache_creation_5m"));
    assert!(columns.iter().any(|column| column == "cache_creation_1h"));

    let row = conn
        .query_row(
            "SELECT cache_creation, cache_creation_5m, cache_creation_1h FROM records",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .expect("read migrated row");
    assert_eq!(row, (11, 11, 0));
}

#[test]
fn pull_limit_caps_legacy_large_pages() {
    assert_eq!(normalized_pull_limit(Some(20_000)), 5_000);
    assert_eq!(normalized_pull_limit(None), 5_000);
    assert_eq!(normalized_pull_limit(Some(250)), 250);
}

#[test]
fn rate_limit_bucket_storage_is_bounded() {
    let state = test_state("bounded-rate-limiters");
    for index in 0..2_000 {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {}", state.config.shared_token)
                .parse()
                .expect("authorization header"),
        );
        headers.insert(
            CLIENT_ID_HEADER,
            format!("client-{index}").parse().expect("client header"),
        );
        let _ = authorize(&state, &headers);
    }

    assert!(state.rate_limiter.lock().expect("rate limiter").len() <= MAX_RATE_LIMIT_BUCKETS);
}

#[test]
fn rotating_client_ids_still_share_the_server_capacity_limit() {
    let state = test_state("global-rate-limit");
    let mut limited = false;
    for index in 0..100 {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {}", state.config.shared_token)
                .parse()
                .expect("authorization header"),
        );
        headers.insert(
            CLIENT_ID_HEADER,
            format!("client-{index}").parse().expect("client header"),
        );
        if authorize(&state, &headers).is_err() {
            limited = true;
            break;
        }
    }

    assert!(limited);
}

#[test]
fn configured_hosts_own_their_rate_limit_buckets() {
    let mut state = test_state("configured-rate-limiters");
    Arc::make_mut(&mut state.config).allowed_hosts =
        Some(HashSet::from(["workstation".to_string()]));
    for index in 0..30 {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {}", state.config.shared_token)
                .parse()
                .expect("authorization header"),
        );
        headers.insert(
            CLIENT_ID_HEADER,
            format!("other-{index}").parse().expect("client header"),
        );
        authorize(&state, &headers).expect("legacy bucket request");
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::AUTHORIZATION,
        format!("Bearer {}", state.config.shared_token)
            .parse()
            .expect("authorization header"),
    );
    headers.insert(
        CLIENT_ID_HEADER,
        "other-final".parse().expect("client header"),
    );

    assert_eq!(
        authorize(&state, &headers)
            .expect_err("shared legacy bucket")
            .status,
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(
        state
            .rate_limiter
            .lock()
            .expect("rate limiter")
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec![LEGACY_RATE_LIMIT_KEY.to_string()]
    );
}

#[test]
fn negotiated_sync_policy_divides_global_capacity_into_stable_slots() {
    let mut state = test_state("sync-policy-slots");
    Arc::make_mut(&mut state.config).allowed_hosts = Some(HashSet::from([
        "client-a".to_string(),
        "client-b".to_string(),
        "client-c".to_string(),
        "client-d".to_string(),
        "client-e".to_string(),
        "client-f".to_string(),
    ]));

    let first = negotiated_sync_policy(&state.config, Some("client-a"));
    let repeated = negotiated_sync_policy(&state.config, Some("client-a"));
    let second = negotiated_sync_policy(&state.config, Some("client-b"));

    assert_eq!(first.min_request_interval_ms, 1_500);
    assert_eq!(first.request_phase_ms, 0);
    assert_eq!(second.request_phase_ms, 250);
    assert_eq!(first, repeated);
    assert!(first.max_request_interval_ms >= first.min_request_interval_ms);
}
