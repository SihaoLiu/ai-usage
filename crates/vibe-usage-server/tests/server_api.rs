use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::json;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;
use vibe_usage_proto::{PullResponse, SCHEMA_VERSION, UploadResponse, WireRecord};
use vibe_usage_server::{AppState, ServerConfig, build_app};

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

fn unique_db_path(name: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("vibe-usage-server-test-{name}-{stamp}.db"))
}

fn config(name: &str) -> ServerConfig {
    ServerConfig {
        listen: "127.0.0.1:0".to_string(),
        db_path: unique_db_path(name),
        shared_token: TOKEN.to_string(),
        allowed_hosts: None,
        max_body_bytes: 1024 * 1024,
        max_batch_records: 1000,
        log_level: "info".to_string(),
    }
}

async fn app(name: &str) -> axum::Router {
    build_app(AppState::new(config(name)).expect("app state"))
}

fn record(host_id: &str, vendor: &str, dedup_key: &str, input_tokens: i64) -> WireRecord {
    WireRecord {
        schema_version: SCHEMA_VERSION,
        host_id: host_id.to_string(),
        vendor: vendor.to_string(),
        dedup_key: dedup_key.to_string(),
        timestamp: "2026-05-18T12:00:00Z".to_string(),
        session_start_time: "2026-05-18T12:00:00Z".to_string(),
        session_end_time: "2026-05-18T12:05:00Z".to_string(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: 1,
        input_tokens,
        output_tokens: 2,
        cache_read_input_tokens: 3,
        cache_creation_input_tokens: 4,
        reasoning_output_tokens: 5,
        project_path_sha256: None,
    }
}

fn ndjson(records: &[WireRecord]) -> String {
    records
        .iter()
        .map(|record| serde_json::to_string(record).expect("serialize record"))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn read_json<T: serde::de::DeserializeOwned>(response: axum::response::Response) -> T {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("json body")
}

fn authed_request(method: &str, uri: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(body.into())
        .expect("request")
}

#[tokio::test]
async fn health_is_public() {
    let response = app("health")
        .await
        .oneshot(
            Request::builder()
                .uri("/v1/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = read_json(response).await;
    assert_eq!(body["ok"], json!(true));
    assert_eq!(body["schema_version"], json!(SCHEMA_VERSION));
}

#[tokio::test]
async fn upload_requires_bearer_token() {
    let response = app("auth")
        .await
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/upload")
                .body(Body::from(ndjson(&[record("laptop", "claude", "a", 1)])))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    assert!(body.is_empty());
}

#[tokio::test]
async fn upload_deduplicates_without_advancing_sequence() {
    let app = app("dedup").await;
    let body = ndjson(&[record("laptop", "claude", "same", 10)]);

    let first = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", body.clone()))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: UploadResponse = read_json(first).await;
    assert_eq!(first_body.accepted, 1);
    assert_eq!(first_body.ignored, 0);
    assert_eq!(first_body.max_seq, 1);

    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", body))
        .await
        .expect("second response");
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 0);
    assert_eq!(second_body.ignored, 1);
    assert_eq!(second_body.max_seq, 1);
}

#[tokio::test]
async fn pull_filters_after_sequence_and_excluded_hosts() {
    let app = app("pull").await;
    let records = vec![
        record("laptop", "claude", "a", 10),
        record("workstation", "codex", "b", 20),
        record("server", "gemini", "c", 30),
    ];
    let upload = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&records)))
        .await
        .expect("upload response");
    assert_eq!(upload.status(), StatusCode::OK);

    let pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=1&exclude_host=server",
            Body::empty(),
        ))
        .await
        .expect("pull response");

    assert_eq!(pull.status(), StatusCode::OK);
    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 1);
    assert_eq!(body.records[0].seq, 2);
    assert_eq!(body.records[0].record.host_id, "workstation");
    assert_eq!(body.records[0].record.fast_tier, 1);
    assert_eq!(body.max_seq, 3);
    assert!(!body.truncated);
}

#[tokio::test]
async fn upload_rejects_batch_over_record_limit() {
    let mut cfg = config("batch-limit");
    cfg.max_batch_records = 1;
    let app = build_app(AppState::new(cfg).expect("app state"));
    let body = ndjson(&[
        record("laptop", "claude", "a", 10),
        record("laptop", "claude", "b", 20),
    ]);

    let response = app
        .oneshot(authed_request("POST", "/v1/upload", body))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn upload_rejects_disallowed_hosts() {
    let mut cfg = config("allowed-hosts");
    cfg.allowed_hosts = Some(HashSet::from(["workstation".to_string()]));
    let app = build_app(AppState::new(cfg).expect("app state"));

    let response = app
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&[record("laptop", "claude", "a", 10)]),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn machines_reports_last_seen_and_record_count() {
    let app = app("machines").await;
    let records = vec![
        record("laptop", "claude", "a", 10),
        record("laptop", "codex", "b", 20),
    ];
    let upload = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&records)))
        .await
        .expect("upload response");
    assert_eq!(upload.status(), StatusCode::OK);

    let response = app
        .oneshot(authed_request("GET", "/v1/machines", Body::empty()))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = read_json(response).await;
    assert_eq!(body["machines"][0]["host_id"], json!("laptop"));
    assert_eq!(body["machines"][0]["record_count"], json!(2));
}

#[tokio::test]
async fn oversized_body_returns_payload_too_large() {
    let mut cfg = config("body-limit");
    cfg.max_body_bytes = 16;
    let app = build_app(AppState::new(cfg).expect("app state"));

    let response = app
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&[record("laptop", "claude", "a", 10)]),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn concurrent_uploads_from_two_clients_succeed() {
    let app = app("concurrent").await;
    let first = app.clone().oneshot(authed_request(
        "POST",
        "/v1/upload",
        ndjson(&[record("laptop", "claude", "a", 10)]),
    ));
    let second = app.clone().oneshot(authed_request(
        "POST",
        "/v1/upload",
        ndjson(&[record("workstation", "codex", "b", 20)]),
    ));

    let (first, second) = tokio::join!(first, second);
    assert_eq!(first.expect("first response").status(), StatusCode::OK);
    assert_eq!(second.expect("second response").status(), StatusCode::OK);

    let pull = app
        .oneshot(authed_request("GET", "/v1/pull?after_seq=0", Body::empty()))
        .await
        .expect("pull response");
    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 2);
}

#[tokio::test]
async fn token_bucket_rejects_after_burst_is_spent() {
    let app = app("rate-limit").await;

    for _ in 0..30 {
        let response = app
            .clone()
            .oneshot(authed_request("GET", "/v1/machines", Body::empty()))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    let response = app
        .oneshot(authed_request("GET", "/v1/machines", Body::empty()))
        .await
        .expect("limited response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}
