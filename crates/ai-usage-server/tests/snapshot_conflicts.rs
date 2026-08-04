use ai_usage_server::{AppState, AutoUpdateConfig, ServerConfig, build_app};
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use serde_json::json;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

fn app(name: &str) -> (axum::Router, PathBuf) {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!("ai-usage-snapshot-{name}-{stamp}.db"));
    let state = AppState::new(ServerConfig {
        listen: "127.0.0.1:0".to_string(),
        db_path: db_path.clone(),
        shared_token: TOKEN.to_string(),
        allowed_hosts: None,
        max_body_bytes: 1024 * 1024,
        max_batch_records: 1000,
        log_level: "info".to_string(),
        auto_update: AutoUpdateConfig::default(),
    })
    .expect("app state");
    (build_app(state), db_path)
}

fn request(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .header("x-ai-usage-client", "laptop")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

#[tokio::test]
async fn unknown_snapshot_finalize_is_rejected_without_registration() {
    let (app, db_path) = app("unknown-finalize");
    let begin = app
        .clone()
        .oneshot(request(
            "/v1/snapshot/diff",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-b", "records": []}),
        ))
        .await
        .expect("begin current snapshot");
    assert_eq!(begin.status(), StatusCode::OK);
    let complete = app
        .clone()
        .oneshot(request(
            "/v1/snapshot/finalize",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-b"}),
        ))
        .await
        .expect("complete current snapshot");
    assert_eq!(complete.status(), StatusCode::OK);

    let unknown = app
        .oneshot(request(
            "/v1/snapshot/finalize",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-unknown"}),
        ))
        .await
        .expect("unknown snapshot finalize");
    assert_eq!(unknown.status(), StatusCode::CONFLICT);

    let conn = rusqlite::Connection::open(db_path).expect("open database");
    let attempts: i64 = conn
        .query_row("SELECT COUNT(*) FROM snapshot_attempts", [], |row| {
            row.get(0)
        })
        .expect("count snapshot attempts");
    assert_eq!(attempts, 1);
}

#[tokio::test]
async fn superseded_snapshot_requests_return_conflict() {
    let (app, _) = app("superseded-requests");
    for snapshot_id in ["snapshot-a", "snapshot-b"] {
        let response = app
            .clone()
            .oneshot(request(
                "/v1/snapshot/diff",
                json!({"host_id": "laptop", "snapshot_id": snapshot_id, "records": []}),
            ))
            .await
            .expect("begin snapshot");
        assert_eq!(response.status(), StatusCode::OK);
    }

    for (path, body) in [
        (
            "/v1/snapshot/diff",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-a", "records": []}),
        ),
        (
            "/v1/snapshot/records",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-a", "records": []}),
        ),
        (
            "/v1/snapshot/keys",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-a", "keys": []}),
        ),
        (
            "/v1/snapshot/finalize",
            json!({"host_id": "laptop", "snapshot_id": "snapshot-a"}),
        ),
    ] {
        let response = app
            .clone()
            .oneshot(request(path, body))
            .await
            .expect("superseded snapshot request");
        assert_eq!(response.status(), StatusCode::CONFLICT, "path={path}");
    }
}
