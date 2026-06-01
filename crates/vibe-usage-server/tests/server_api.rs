use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use rusqlite::params;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;
use vibe_usage_proto::{
    INTEGRITY_ALGORITHM, IntegrityReport, IntegrityReportList, IntegritySubmitResponse,
    PullResponse, SCHEMA_VERSION, UploadResponse, WireRecord,
};
use vibe_usage_server::{AppState, AutoUpdateConfig, ServerConfig, build_app};

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
        auto_update: AutoUpdateConfig::default(),
    }
}

fn unique_config_path(name: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("vibe-usage-server-config-{name}-{stamp}.yaml"))
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
        cost_input: None,
        cost_output: None,
        cost_cache_read: None,
        cost_cache_creation: None,
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

fn record_hash(record: &WireRecord) -> String {
    let bytes = serde_json::to_vec(record).expect("serialize record");
    let digest = Sha256::digest(&bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn omp_v220_key(message_id: &str, response_id: &str, model: &str, record: &WireRecord) -> String {
    json!({
        "message": message_id,
        "response": response_id,
        "model": model,
        "input": record.input_tokens,
        "output": record.output_tokens,
        "cache_read": record.cache_read_input_tokens,
        "cache_write": record.cache_creation_input_tokens,
    })
    .to_string()
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

fn seed_existing_records(db_path: &Path, records: &[WireRecord]) {
    let conn = rusqlite::Connection::open(db_path).expect("open seed database");
    conn.execute_batch(include_str!("../migrations/0001_init.sql"))
        .expect("create schema");
    let uploaded_at = "2026-05-18T12:10:00Z";
    let mut touched_hosts = HashSet::new();
    for record in records {
        conn.execute(
            "INSERT INTO records (
                host_id, vendor, dedup_key, schema_version, timestamp_utc,
                session_start, session_end, model, effort, fast_tier, input_tokens,
                output_tokens, cache_read, cache_creation, reasoning_out,
                cost_input, cost_output, cost_cache_read, cost_cache_creation,
                project_hash, uploaded_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            params![
                record.host_id,
                record.vendor,
                record.dedup_key,
                i64::from(record.schema_version),
                record.timestamp,
                record.session_start_time,
                record.session_end_time,
                record.model,
                record.effort,
                i64::from(record.fast_tier),
                record.input_tokens,
                record.output_tokens,
                record.cache_read_input_tokens,
                record.cache_creation_input_tokens,
                record.reasoning_output_tokens,
                record.cost_input,
                record.cost_output,
                record.cost_cache_read,
                record.cost_cache_creation,
                record.project_path_sha256,
                uploaded_at,
            ],
        )
        .expect("seed record");
        touched_hosts.insert(record.host_id.clone());
    }
    for host_id in touched_hosts {
        conn.execute(
            "INSERT INTO machines (host_id, last_seen, record_count)
             VALUES (?1, ?2, (SELECT COUNT(*) FROM records WHERE host_id = ?1))",
            params![host_id, uploaded_at],
        )
        .expect("seed machine");
    }
}

#[test]
fn server_config_defaults_auto_update_disabled() {
    let path = unique_config_path("default-auto-update");
    std::fs::write(
        &path,
        format!(
            r#"
listen: "127.0.0.1:0"
db_path: "{}"
shared_token: "{TOKEN}"
"#,
            unique_db_path("default-auto-update").display()
        ),
    )
    .expect("write config");

    let cfg = ServerConfig::load_from_path(&path).expect("load config");

    assert!(!cfg.auto_update.enabled);
    assert_eq!(cfg.auto_update.interval_seconds, 3600);
}

#[test]
fn server_config_parses_auto_update_settings() {
    let path = unique_config_path("enabled-auto-update");
    std::fs::write(
        &path,
        format!(
            r#"
listen: "127.0.0.1:0"
db_path: "{}"
shared_token: "{TOKEN}"
auto_update:
  enabled: true
  interval_seconds: 7200
"#,
            unique_db_path("enabled-auto-update").display()
        ),
    )
    .expect("write config");

    let cfg = ServerConfig::load_from_path(&path).expect("load config");

    assert!(cfg.auto_update.enabled);
    assert_eq!(cfg.auto_update.interval_seconds, 7200);
}

#[test]
fn systemd_unit_uses_auto_update_compatible_restart_and_path() {
    let unit = include_str!("../deploy/vibe-usage-server.service.example");

    assert!(unit.contains("ExecStart=/var/lib/vibe-usage/bin/vibe-usage-server "));
    assert!(unit.contains("Restart=always"));
    assert!(unit.contains("ReadWritePaths=/var/lib/vibe-usage"));
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
    assert_eq!(body["version"], json!(env!("CARGO_PKG_VERSION")));
    assert_eq!(body["schema_version"], json!(SCHEMA_VERSION));
}

#[tokio::test]
async fn startup_removes_existing_omp_alias_duplicates_before_pull() {
    let cfg = config("startup-omp-alias-duplicates");
    let mut message_legacy = record("laptop", "omp", "placeholder", 10);
    message_legacy.model = "gpt-5.5".to_string();
    message_legacy.effort = Some("openai-codex".to_string());
    message_legacy.dedup_key =
        omp_v220_key("msg-a", "resp-a", "openai-codex/gpt-5.5", &message_legacy);
    let mut message_stable = message_legacy.clone();
    message_stable.dedup_key = "omp:message:msg-a:response:resp-a".to_string();

    let mut file_legacy = record("laptop", "omp", "placeholder", 20);
    file_legacy.dedup_key = omp_v220_key("", "", "test-model", &file_legacy);
    let file_stable_a = record("laptop", "omp", "omp:file:/tmp/omp.jsonl:0", 20);
    let file_stable_b = record("laptop", "omp", "omp:file:/tmp/omp.jsonl:1", 20);
    let claude = record("laptop", "claude", "claude-a", 30);
    seed_existing_records(
        &cfg.db_path,
        &[
            message_legacy,
            message_stable,
            file_legacy,
            file_stable_a,
            file_stable_b,
            claude,
        ],
    );
    let app = build_app(AppState::new(cfg).expect("app state"));

    let pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=omp",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    assert_eq!(pull.status(), StatusCode::OK);
    let body: PullResponse = read_json(pull).await;
    let keys = body
        .records
        .iter()
        .map(|record| record.record.dedup_key.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![
            "omp:message:msg-a:response:resp-a",
            "omp:file:/tmp/omp.jsonl:0",
            "omp:file:/tmp/omp.jsonl:1",
        ]
    );

    let machines = app
        .oneshot(authed_request("GET", "/v1/machines", Body::empty()))
        .await
        .expect("machines response");
    assert_eq!(machines.status(), StatusCode::OK);
    let body: vibe_usage_proto::MachineList = read_json(machines).await;
    assert_eq!(body.machines.len(), 1);
    assert_eq!(body.machines[0].record_count, 4);
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
async fn upload_updates_changed_existing_record() {
    let app = app("changed-record-upsert").await;
    let original = record("laptop", "codex", "same", 10);
    let mut changed = original.clone();
    changed.output_tokens = 99;
    changed.fast_tier = 0;

    let first = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[original])))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: UploadResponse = read_json(first).await;
    assert_eq!(first_body.accepted, 1);
    assert_eq!(first_body.max_seq, 1);

    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[changed])))
        .await
        .expect("second response");
    assert_eq!(second.status(), StatusCode::OK);
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 1);
    assert_eq!(second_body.ignored, 0);
    assert!(second_body.max_seq > first_body.max_seq);

    let pull = app
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=codex",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    assert_eq!(pull.status(), StatusCode::OK);
    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 1);
    assert_eq!(body.records[0].seq, second_body.max_seq);
    assert_eq!(body.records[0].record.output_tokens, 99);
    assert_eq!(body.records[0].record.fast_tier, 0);
}

#[tokio::test]
async fn snapshot_finalize_deletes_keys_missing_from_active_manifest() {
    let app = app("snapshot-delete-stale").await;
    let active = record("laptop", "claude", "active", 10);
    let stale = record("laptop", "claude", "stale", 20);
    let upload = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&[active.clone(), stale]),
        ))
        .await
        .expect("upload response");
    assert_eq!(upload.status(), StatusCode::OK);

    let keys = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/snapshot/keys",
            json!({
                "host_id": "laptop",
                "snapshot_id": "snapshot-a",
                "keys": [{"vendor": "claude", "dedup_key": "active"}]
            })
            .to_string(),
        ))
        .await
        .expect("keys response");
    assert_eq!(keys.status(), StatusCode::OK);

    let finalize = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/snapshot/finalize",
            json!({
                "host_id": "laptop",
                "snapshot_id": "snapshot-a"
            })
            .to_string(),
        ))
        .await
        .expect("finalize response");
    assert_eq!(finalize.status(), StatusCode::OK);
    let finalize_body: serde_json::Value = read_json(finalize).await;
    assert_eq!(finalize_body["deleted"], json!(1));

    let pull = app
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=claude",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    assert_eq!(pull.status(), StatusCode::OK);
    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 1);
    assert_eq!(body.records[0].record.dedup_key, active.dedup_key);
}

#[tokio::test]
async fn snapshot_diff_uploads_only_missing_or_changed_records_and_deletes_stale() {
    let app = app("snapshot-diff").await;
    let active = record("laptop", "claude", "active", 10);
    let old_changed = record("laptop", "claude", "changed", 20);
    let stale = record("laptop", "claude", "stale", 30);
    let mut changed = old_changed.clone();
    changed.input_tokens = 99;
    let missing = record("laptop", "codex", "missing", 40);

    let upload = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&[active.clone(), old_changed, stale]),
        ))
        .await
        .expect("upload response");
    assert_eq!(upload.status(), StatusCode::OK);

    let diff = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/snapshot/diff",
            json!({
                "host_id": "laptop",
                "snapshot_id": "snapshot-b",
                "records": [
                    {
                        "vendor": "claude",
                        "dedup_key": "active",
                        "record_hash": record_hash(&active)
                    },
                    {
                        "vendor": "claude",
                        "dedup_key": "changed",
                        "record_hash": record_hash(&changed)
                    },
                    {
                        "vendor": "codex",
                        "dedup_key": "missing",
                        "record_hash": record_hash(&missing)
                    }
                ]
            })
            .to_string(),
        ))
        .await
        .expect("diff response");
    assert_eq!(diff.status(), StatusCode::OK);
    let diff_body: serde_json::Value = read_json(diff).await;
    assert_eq!(diff_body["matched"], json!(1));
    assert_eq!(diff_body["needed"].as_array().expect("needed").len(), 2);
    assert!(
        diff_body["needed"]
            .as_array()
            .expect("needed")
            .contains(&json!({
                "vendor": "claude",
                "dedup_key": "changed"
            }))
    );
    assert!(
        diff_body["needed"]
            .as_array()
            .expect("needed")
            .contains(&json!({
                "vendor": "codex",
                "dedup_key": "missing"
            }))
    );

    let snapshot_upload = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/snapshot/records",
            json!({
                "host_id": "laptop",
                "snapshot_id": "snapshot-b",
                "records": [changed, missing]
            })
            .to_string(),
        ))
        .await
        .expect("snapshot upload response");
    assert_eq!(snapshot_upload.status(), StatusCode::OK);

    let finalize = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/snapshot/finalize",
            json!({
                "host_id": "laptop",
                "snapshot_id": "snapshot-b"
            })
            .to_string(),
        ))
        .await
        .expect("finalize response");
    assert_eq!(finalize.status(), StatusCode::OK);
    let finalize_body: serde_json::Value = read_json(finalize).await;
    assert_eq!(finalize_body["deleted"], json!(1));

    let pull = app
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=claude,codex",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    assert_eq!(pull.status(), StatusCode::OK);
    let body: PullResponse = read_json(pull).await;
    let rows = body
        .records
        .iter()
        .map(|record| {
            (
                record.record.vendor.as_str(),
                record.record.dedup_key.as_str(),
                record.record.input_tokens,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            ("claude", "active", 10),
            ("claude", "changed", 99),
            ("codex", "missing", 40),
        ]
    );
}

#[tokio::test]
async fn upload_replaces_omp_v220_message_key_with_stable_key() {
    let app = app("omp-v220-message-key").await;
    let mut old_record = record("laptop", "omp", "placeholder", 10);
    old_record.model = "gpt-5.5".to_string();
    old_record.effort = Some("openai-codex".to_string());
    old_record.dedup_key = omp_v220_key("msg-a", "resp-a", "openai-codex/gpt-5.5", &old_record);
    let first = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[old_record])))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: UploadResponse = read_json(first).await;
    assert_eq!(first_body.accepted, 1);
    assert_eq!(first_body.max_seq, 1);

    let mut new_record = record("laptop", "omp", "omp:message:msg-a:response:resp-a", 10);
    new_record.model = "gpt-5.5".to_string();
    new_record.effort = Some("openai-codex".to_string());
    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[new_record])))
        .await
        .expect("second response");

    assert_eq!(second.status(), StatusCode::OK);
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 1);
    assert_eq!(second_body.ignored, 0);
    assert_eq!(second_body.max_seq, 2);

    let pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=omp",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 1);
    assert_eq!(
        body.records[0].record.dedup_key,
        "omp:message:msg-a:response:resp-a"
    );
}

#[tokio::test]
async fn upload_refreshes_existing_omp_stable_metadata() {
    let app = app("omp-stable-metadata-refresh").await;
    let mut old_record = record("laptop", "omp", "omp:message:msg-a:response:resp-a", 10);
    old_record.model = "claude-sonnet-4-5-20250929".to_string();
    let first = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&[old_record.clone()]),
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: UploadResponse = read_json(first).await;
    assert_eq!(first_body.accepted, 1);
    assert_eq!(first_body.max_seq, 1);

    let mut refreshed = old_record;
    refreshed.effort = Some("anthropic".to_string());
    refreshed.cost_input = Some(0.01);
    refreshed.cost_output = Some(0.02);
    refreshed.cost_cache_read = Some(0.03);
    refreshed.cost_cache_creation = Some(0.04);
    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[refreshed])))
        .await
        .expect("second response");

    assert_eq!(second.status(), StatusCode::OK);
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 1);
    assert_eq!(second_body.ignored, 0);
    assert!(second_body.max_seq > 1);

    let pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=omp",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 1);
    assert_eq!(body.records[0].seq, second_body.max_seq);
    let record = &body.records[0].record;
    assert_eq!(record.effort.as_deref(), Some("anthropic"));
    assert_eq!(record.cost_input, Some(0.01));
    assert_eq!(record.cost_output, Some(0.02));
    assert_eq!(record.cost_cache_read, Some(0.03));
    assert_eq!(record.cost_cache_creation, Some(0.04));
}

#[tokio::test]
async fn upload_treats_omp_stable_message_key_as_duplicate_for_v220_key() {
    let app = app("omp-stable-message-key").await;
    let mut stable_record = record("laptop", "omp", "omp:message:msg-a:response:resp-a", 10);
    stable_record.model = "gpt-5.5".to_string();
    stable_record.effort = Some("openai-codex".to_string());
    let first = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&[stable_record.clone()]),
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: UploadResponse = read_json(first).await;
    assert_eq!(first_body.accepted, 1);
    assert_eq!(first_body.max_seq, 1);

    let mut old_record = stable_record;
    old_record.dedup_key = omp_v220_key("msg-a", "resp-a", "openai-codex/gpt-5.5", &old_record);
    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[old_record])))
        .await
        .expect("second response");

    assert_eq!(second.status(), StatusCode::OK);
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 0);
    assert_eq!(second_body.ignored, 1);
    assert_eq!(second_body.max_seq, 1);
}

#[tokio::test]
async fn upload_replaces_omp_v220_file_key_with_stable_file_keys() {
    let app = app("omp-v220-file-key").await;
    let mut old_record = record("laptop", "omp", "placeholder", 10);
    old_record.dedup_key = omp_v220_key("", "", "test-model", &old_record);
    let first = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[old_record])))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);

    let new_records = vec![
        record("laptop", "omp", "omp:file:/tmp/omp.jsonl:0", 10),
        record("laptop", "omp", "omp:file:/tmp/omp.jsonl:1", 10),
    ];
    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&new_records)))
        .await
        .expect("second response");

    assert_eq!(second.status(), StatusCode::OK);
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 2);
    assert_eq!(second_body.ignored, 0);
    assert_eq!(second_body.max_seq, 3);

    let pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=omp",
            Body::empty(),
        ))
        .await
        .expect("pull response");
    let body: PullResponse = read_json(pull).await;
    let keys = body
        .records
        .iter()
        .map(|record| record.record.dedup_key.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec!["omp:file:/tmp/omp.jsonl:0", "omp:file:/tmp/omp.jsonl:1"]
    );
}

#[tokio::test]
async fn upload_consumes_one_omp_stable_file_key_duplicate_for_v220_key() {
    let app = app("omp-stable-file-key").await;
    let stable_records = vec![
        record("laptop", "omp", "omp:file:/tmp/omp.jsonl:0", 10),
        record("laptop", "omp", "omp:file:/tmp/omp.jsonl:1", 10),
    ];
    let first = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/upload",
            ndjson(&stable_records),
        ))
        .await
        .expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body: UploadResponse = read_json(first).await;
    assert_eq!(first_body.accepted, 2);
    assert_eq!(first_body.max_seq, 2);

    let mut old_record = record("laptop", "omp", "placeholder", 10);
    old_record.dedup_key = omp_v220_key("", "", "test-model", &old_record);
    let second = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[old_record])))
        .await
        .expect("second response");

    assert_eq!(second.status(), StatusCode::OK);
    let second_body: UploadResponse = read_json(second).await;
    assert_eq!(second_body.accepted, 0);
    assert_eq!(second_body.ignored, 1);
    assert_eq!(second_body.max_seq, 2);
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
async fn pull_defaults_to_legacy_vendors_when_supported_vendors_are_absent() {
    let app = app("pull-legacy-vendors").await;
    let records = vec![
        record("laptop", "claude", "claude-a", 10),
        record("laptop", "omp", "omp-a", 20),
    ];
    let upload = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&records)))
        .await
        .expect("upload response");
    assert_eq!(upload.status(), StatusCode::OK);

    let legacy_pull = app
        .clone()
        .oneshot(authed_request("GET", "/v1/pull?after_seq=0", Body::empty()))
        .await
        .expect("legacy pull response");

    assert_eq!(legacy_pull.status(), StatusCode::OK);
    let legacy_body: PullResponse = read_json(legacy_pull).await;
    assert_eq!(legacy_body.records.len(), 1);
    assert_eq!(legacy_body.records[0].record.vendor, "claude");
    assert_eq!(legacy_body.max_seq, 1);

    let late_legacy_pull = app
        .clone()
        .oneshot(authed_request("GET", "/v1/pull?after_seq=2", Body::empty()))
        .await
        .expect("late legacy pull response");

    assert_eq!(late_legacy_pull.status(), StatusCode::OK);
    let late_legacy_body: PullResponse = read_json(late_legacy_pull).await;
    assert!(late_legacy_body.records.is_empty());
    assert_eq!(late_legacy_body.max_seq, 2);

    let current_pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=claude,codex,gemini,omp",
            Body::empty(),
        ))
        .await
        .expect("current pull response");

    assert_eq!(current_pull.status(), StatusCode::OK);
    let current_body: PullResponse = read_json(current_pull).await;
    assert_eq!(current_body.records.len(), 2);
    assert_eq!(current_body.records[0].record.vendor, "claude");
    assert_eq!(current_body.records[1].record.vendor, "omp");
    assert_eq!(current_body.max_seq, 2);
}

#[tokio::test]
async fn upload_and_pull_preserve_embedded_costs() {
    let app = app("costs").await;
    let mut costed = record("laptop", "omp", "costed", 10);
    costed.cost_input = Some(0.01);
    costed.cost_output = Some(0.02);
    costed.cost_cache_read = Some(0.03);
    costed.cost_cache_creation = Some(0.04);

    let upload = app
        .clone()
        .oneshot(authed_request("POST", "/v1/upload", ndjson(&[costed])))
        .await
        .expect("upload response");
    assert_eq!(upload.status(), StatusCode::OK);

    let pull = app
        .clone()
        .oneshot(authed_request(
            "GET",
            "/v1/pull?after_seq=0&supported_vendors=claude,codex,gemini,omp",
            Body::empty(),
        ))
        .await
        .expect("pull response");

    let body: PullResponse = read_json(pull).await;
    assert_eq!(body.records.len(), 1);
    let record = &body.records[0].record;
    assert_eq!(record.cost_input, Some(0.01));
    assert_eq!(record.cost_output, Some(0.02));
    assert_eq!(record.cost_cache_read, Some(0.03));
    assert_eq!(record.cost_cache_creation, Some(0.04));
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
async fn integrity_report_round_trips_through_server() {
    let app = app("integrity-report").await;
    let report = IntegrityReport {
        host_id: "laptop".to_string(),
        algorithm: INTEGRITY_ALGORITHM.to_string(),
        range_end_utc: "2026-06-01T00:00:00Z".to_string(),
        record_count: 2,
        digest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            .to_string(),
        computed_at: "2026-06-01T12:00:00Z".to_string(),
    };

    let submit = app
        .clone()
        .oneshot(authed_request(
            "POST",
            "/v1/integrity/report",
            serde_json::to_string(&report).expect("serialize report"),
        ))
        .await
        .expect("submit response");
    assert_eq!(submit.status(), StatusCode::OK);
    let submit_body: IntegritySubmitResponse = read_json(submit).await;
    assert!(submit_body.accepted);

    let list = app
        .oneshot(authed_request(
            "GET",
            "/v1/integrity/reports",
            Body::empty(),
        ))
        .await
        .expect("list response");
    assert_eq!(list.status(), StatusCode::OK);
    let body: IntegrityReportList = read_json(list).await;

    assert_eq!(body.reports, vec![report]);
}

#[tokio::test]
async fn integrity_report_rejects_disallowed_hosts() {
    let mut cfg = config("integrity-allowed-hosts");
    cfg.allowed_hosts = Some(HashSet::from(["workstation".to_string()]));
    let app = build_app(AppState::new(cfg).expect("app state"));
    let report = IntegrityReport {
        host_id: "laptop".to_string(),
        algorithm: INTEGRITY_ALGORITHM.to_string(),
        range_end_utc: "2026-06-01T00:00:00Z".to_string(),
        record_count: 2,
        digest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            .to_string(),
        computed_at: "2026-06-01T12:00:00Z".to_string(),
    };

    let response = app
        .oneshot(authed_request(
            "POST",
            "/v1/integrity/report",
            serde_json::to_string(&report).expect("serialize report"),
        ))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn pull_does_not_skip_records_committed_concurrently() {
    use tokio::sync::Barrier;

    const ITERATIONS: usize = 200;
    const PARALLEL_UPLOADS: usize = 8;

    for iteration in 0..ITERATIONS {
        let cfg = ServerConfig {
            max_batch_records: 1000,
            ..config(&format!("pull-race-{iteration}"))
        };
        let app = build_app(AppState::new(cfg).expect("app state"));
        let barrier = std::sync::Arc::new(Barrier::new(PARALLEL_UPLOADS + 1));

        let pull_app = app.clone();
        let pull_barrier = std::sync::Arc::clone(&barrier);
        let pull_handle = tokio::spawn(async move {
            pull_barrier.wait().await;
            pull_app
                .oneshot(authed_request(
                    "GET",
                    "/v1/pull?after_seq=0&exclude_host=host-b",
                    Body::empty(),
                ))
                .await
                .expect("pull response")
        });

        let upload_handles: Vec<_> = (0..PARALLEL_UPLOADS)
            .map(|slot| {
                let upload_app = app.clone();
                let upload_barrier = std::sync::Arc::clone(&barrier);
                let dedup_key = format!("race-{iteration}-{slot}");
                tokio::spawn(async move {
                    upload_barrier.wait().await;
                    upload_app
                        .oneshot(authed_request(
                            "POST",
                            "/v1/upload",
                            ndjson(&[record("host-a", "claude", &dedup_key, slot as i64 + 1)]),
                        ))
                        .await
                        .expect("upload response")
                })
            })
            .collect();

        let pull = pull_handle.await.expect("pull join");
        assert_eq!(pull.status(), StatusCode::OK, "iteration {iteration}");
        for handle in upload_handles {
            let response = handle.await.expect("upload join");
            assert_eq!(response.status(), StatusCode::OK, "iteration {iteration}");
        }
        let first_pull: PullResponse = read_json(pull).await;
        let last_returned_seq = first_pull.records.last().map(|r| r.seq).unwrap_or(0);

        let mut total_records = first_pull.records.len();
        let mut cursor = first_pull.max_seq;
        loop {
            let response = app
                .clone()
                .oneshot(authed_request(
                    "GET",
                    &format!("/v1/pull?after_seq={cursor}&exclude_host=host-b"),
                    Body::empty(),
                ))
                .await
                .expect("follow-up pull");
            assert_eq!(response.status(), StatusCode::OK, "iteration {iteration}");
            let body: PullResponse = read_json(response).await;
            total_records += body.records.len();
            if body.max_seq == cursor && !body.truncated {
                break;
            }
            cursor = body.max_seq;
        }

        assert_eq!(
            total_records,
            PARALLEL_UPLOADS,
            "iteration {iteration}: host-b lost records. first pull returned {} records (last seq={}), advertised max_seq={}, follow-up drained to seq={}",
            first_pull.records.len(),
            last_returned_seq,
            first_pull.max_seq,
            cursor,
        );
    }
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
