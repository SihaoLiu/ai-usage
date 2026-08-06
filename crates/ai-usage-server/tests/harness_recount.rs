use ai_usage_proto::{
    PullResponse, RecordKey, SCHEMA_VERSION, SnapshotDiffResponse, SnapshotFinalizeResponse,
    WireRecord,
};
use ai_usage_server::{AppState, AutoUpdateConfig, ServerConfig, build_app};
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

fn config(name: &str) -> ServerConfig {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let temp_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("temp");
    std::fs::create_dir_all(&temp_dir).expect("create temp directory");
    ServerConfig {
        listen: "127.0.0.1:0".to_string(),
        db_path: temp_dir.join(format!("ai-usage-{name}-{stamp}.db")),
        shared_token: TOKEN.to_string(),
        allowed_hosts: None,
        max_body_bytes: 1024 * 1024,
        max_batch_records: 1000,
        log_level: "info".to_string(),
        auto_update: AutoUpdateConfig::default(),
    }
}

fn record(host_id: &str, vendor: &str, dedup_key: &str, input_tokens: i64) -> WireRecord {
    WireRecord {
        schema_version: SCHEMA_VERSION,
        host_id: host_id.to_string(),
        vendor: vendor.to_string(),
        dedup_key: dedup_key.to_string(),
        timestamp: "2026-08-05T16:43:15Z".to_string(),
        session_start_time: "2026-08-05T16:43:15Z".to_string(),
        session_end_time: "2026-08-05T16:43:15Z".to_string(),
        model: "test-model".to_string(),
        effort: None,
        fast_tier: -1,
        input_tokens,
        output_tokens: 2,
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

#[tokio::test]
async fn changed_harness_snapshots_replace_legacy_parser_keys() {
    let app = build_app(AppState::new(config("harness-recount")).expect("app state"));
    let legacy = [
        record("workstation", "claude", "fallback:v1:legacy-claude:0", 5),
        record("workstation", "codex", "legacy-codex", 10),
        record("workstation", "gemini", "legacy-gemini", 20),
        record("workstation", "kimi", "legacy-kimi", 30),
    ];
    let upload = app
        .clone()
        .oneshot(request("POST", "/v1/upload", ndjson(&legacy)))
        .await
        .expect("legacy upload");
    assert_eq!(upload.status(), StatusCode::OK);

    let mut canonical = [
        record(
            "workstation",
            "claude",
            "claude:file:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:12",
            5,
        ),
        record(
            "workstation",
            "codex",
            "codex:turn:turn-a:cumulative:start->10,2,0,3,1,13",
            10,
        ),
        record("workstation", "gemini", "gemini:session-a:message-a", 20),
        record("workstation", "kimi", "kimi:response:response-a", 30),
    ];
    canonical[1].input_tokens = 8;
    canonical[1].output_tokens = 3;
    canonical[1].cache_read_input_tokens = 2;
    canonical[1].cache_creation_input_tokens = 0;
    canonical[1].reasoning_output_tokens = 1;
    let summaries = canonical
        .iter()
        .map(|record| {
            json!({
                "vendor": record.vendor,
                "dedup_key": record.dedup_key,
                "record_hash": record_hash(record),
            })
        })
        .collect::<Vec<_>>();
    let diff = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/diff",
            json!({
                "host_id": "workstation",
                "snapshot_id": "recount-a",
                "records": summaries,
            })
            .to_string(),
        ))
        .await
        .expect("snapshot diff");
    assert_eq!(diff.status(), StatusCode::OK);
    let diff: SnapshotDiffResponse = read_json(diff).await;
    assert_eq!(
        diff.needed,
        canonical
            .iter()
            .map(|record| RecordKey {
                vendor: record.vendor.clone(),
                dedup_key: record.dedup_key.clone(),
            })
            .collect::<Vec<_>>()
    );

    let records = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/records",
            json!({
                "host_id": "workstation",
                "snapshot_id": "recount-a",
                "records": canonical,
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
    assert_eq!(finalize.deleted, 4);

    let pull = app
        .oneshot(request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=claude,codex,gemini,kimi",
            Body::empty(),
        ))
        .await
        .expect("pull");
    assert_eq!(pull.status(), StatusCode::OK);
    let pull: PullResponse = read_json(pull).await;
    assert_eq!(pull.records.len(), 4);
    assert!(pull.records.iter().all(|record| {
        record.record.dedup_key.starts_with("claude:")
            || record.record.dedup_key.starts_with("codex:")
            || record.record.dedup_key.starts_with("gemini:")
            || record.record.dedup_key.starts_with("kimi:")
    }));
    let codex = pull
        .records
        .iter()
        .find(|record| record.record.vendor == "codex")
        .expect("codex record");
    assert_eq!(codex.record.input_tokens, 8);
    assert_eq!(codex.record.output_tokens, 3);
    assert_eq!(codex.record.cache_read_input_tokens, 2);
    assert_eq!(codex.record.cache_creation_input_tokens, 0);
    assert_eq!(codex.record.reasoning_output_tokens, 1);
}

#[tokio::test]
async fn host_snapshot_finalize_never_deletes_another_hosts_copy() {
    let app = build_app(AppState::new(config("cross-host-ownership")).expect("app state"));
    let stable_key = "codex:turn:turn-a:cumulative:start->10,2,0,3,1,13";
    let records = [
        record("host-a", "codex", stable_key, 10),
        record("host-b", "codex", stable_key, 20),
    ];
    let upload = app
        .clone()
        .oneshot(request("POST", "/v1/upload", ndjson(&records)))
        .await
        .expect("upload host copies");
    assert_eq!(upload.status(), StatusCode::OK);

    let keys = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/keys",
            json!({
                "host_id": "host-a",
                "snapshot_id": "empty-host-a",
                "keys": [],
            })
            .to_string(),
        ))
        .await
        .expect("snapshot keys");
    assert_eq!(keys.status(), StatusCode::OK);
    let finalize = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/snapshot/finalize",
            json!({"host_id": "host-a", "snapshot_id": "empty-host-a"}).to_string(),
        ))
        .await
        .expect("snapshot finalize");
    let finalize: SnapshotFinalizeResponse = read_json(finalize).await;
    assert_eq!(finalize.deleted, 1);

    let pull = app
        .oneshot(request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=codex",
            Body::empty(),
        ))
        .await
        .expect("pull remaining copy");
    let pull: PullResponse = read_json(pull).await;
    assert_eq!(pull.records.len(), 1);
    assert_eq!(pull.records[0].record.host_id, "host-b");
    assert_eq!(pull.records[0].record.dedup_key, stable_key);
    assert_eq!(pull.records[0].record.input_tokens, 20);
}
