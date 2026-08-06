use ai_usage_proto::{
    MachineList, PullResponse, RecordKey, SCHEMA_VERSION, SnapshotDiffResponse,
    SnapshotFinalizeResponse, UploadResponse, WireRecord,
};
use ai_usage_server::{AppState, AutoUpdateConfig, ServerConfig, build_app};
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

fn config() -> ServerConfig {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    ServerConfig {
        listen: "127.0.0.1:0".to_string(),
        db_path: std::env::temp_dir().join(format!("ai-usage-claude-recount-{stamp}.db")),
        shared_token: TOKEN.to_string(),
        allowed_hosts: None,
        max_body_bytes: 1024 * 1024,
        max_batch_records: 1000,
        log_level: "info".to_string(),
        auto_update: AutoUpdateConfig::default(),
    }
}

fn record(dedup_key: &str, output_tokens: i64) -> WireRecord {
    WireRecord {
        schema_version: SCHEMA_VERSION,
        host_id: "workstation".to_string(),
        vendor: "claude".to_string(),
        dedup_key: dedup_key.to_string(),
        timestamp: "2026-08-05T16:43:15Z".to_string(),
        session_start_time: "2026-08-05T16:43:15Z".to_string(),
        session_end_time: "2026-08-05T16:43:15Z".to_string(),
        model: "claude-test".to_string(),
        effort: None,
        fast_tier: -1,
        input_tokens: 1,
        output_tokens,
        cache_read_input_tokens: 3,
        cache_creation_input_tokens: 4,
        reasoning_output_tokens: 0,
        cost_input: None,
        cost_output: None,
        cost_cache_read: None,
        cost_cache_creation: None,
        project_path_sha256: None,
    }
}

fn record_hash(record: &WireRecord) -> String {
    Sha256::digest(serde_json::to_vec(record).expect("serialize record"))
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn request(method: &str, uri: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .expect("request")
}

async fn read_json<T: serde::de::DeserializeOwned>(response: axum::response::Response) -> T {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("json body")
}

#[tokio::test]
async fn snapshot_recount_replaces_fallback_rows_with_one_response_row() {
    let app = build_app(AppState::new(config()).expect("app state"));
    let fallback = [
        record("fallback:v1:response-a:0", 2),
        record("fallback:v1:response-a:1", 5),
        record("fallback:v1:response-a:2", 8),
    ];
    let upload = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/upload",
            fallback
                .iter()
                .map(|record| serde_json::to_string(record).expect("serialize record"))
                .collect::<Vec<_>>()
                .join("\n"),
        ))
        .await
        .expect("legacy upload");
    assert_eq!(upload.status(), StatusCode::OK);
    let upload: UploadResponse = read_json(upload).await;
    assert_eq!(upload.accepted, 3);

    let canonical = record("claude:message:response-a", 8);
    let diff = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/diff",
            json!({
                "host_id": "workstation",
                "snapshot_id": "recount-a",
                "records": [{
                    "vendor": "claude",
                    "dedup_key": canonical.dedup_key,
                    "record_hash": record_hash(&canonical)
                }]
            })
            .to_string(),
        ))
        .await
        .expect("snapshot diff");
    assert_eq!(diff.status(), StatusCode::OK);
    let diff: SnapshotDiffResponse = read_json(diff).await;
    assert_eq!(
        diff.needed,
        vec![RecordKey {
            vendor: "claude".to_string(),
            dedup_key: canonical.dedup_key.clone(),
        }]
    );

    let records = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/records",
            json!({
                "host_id": "workstation",
                "snapshot_id": "recount-a",
                "records": [canonical]
            })
            .to_string(),
        ))
        .await
        .expect("snapshot records");
    assert_eq!(records.status(), StatusCode::OK);

    let finalize = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/finalize",
            json!({"host_id": "workstation", "snapshot_id": "recount-a"}).to_string(),
        ))
        .await
        .expect("snapshot finalize");
    assert_eq!(finalize.status(), StatusCode::OK);
    let finalize: SnapshotFinalizeResponse = read_json(finalize).await;
    assert_eq!(finalize.deleted, 3);

    let pull = app
        .clone()
        .oneshot(request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=claude",
            Body::empty(),
        ))
        .await
        .expect("pull");
    let pull: PullResponse = read_json(pull).await;
    assert_eq!(pull.records.len(), 1);
    assert_eq!(
        pull.records[0].record.dedup_key,
        "claude:message:response-a"
    );
    assert_eq!(pull.records[0].record.output_tokens, 8);

    let machines = app
        .oneshot(request("GET", "/v1/machines", Body::empty()))
        .await
        .expect("machines");
    let machines: MachineList = read_json(machines).await;
    assert_eq!(machines.machines.len(), 1);
    assert_eq!(machines.machines[0].record_count, 1);
}
