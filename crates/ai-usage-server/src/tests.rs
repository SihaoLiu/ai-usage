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
