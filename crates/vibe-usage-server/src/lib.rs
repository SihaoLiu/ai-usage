use axum::extract::{DefaultBodyLimit, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::types::Value;
use rusqlite::{OptionalExtension, TransactionBehavior, params, params_from_iter};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;
use vibe_usage_proto::{
    HealthResponse, MachineInfo, MachineList, PullResponse, SCHEMA_VERSION, SequencedWireRecord,
    UploadResponse, WireRecord, is_valid_vendor,
};

type DbPool = Pool<SqliteConnectionManager>;
type BoxError = Box<dyn std::error::Error + Send + Sync>;
const RATE_LIMIT_REFILL_PER_SECOND: f64 = 1.0;
const RATE_LIMIT_BURST: f64 = 30.0;
const LEGACY_PULL_VENDORS: [&str; 3] = ["claude", "codex", "gemini"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub listen: String,
    pub db_path: PathBuf,
    pub shared_token: String,
    pub allowed_hosts: Option<HashSet<String>>,
    pub max_body_bytes: usize,
    pub max_batch_records: usize,
    pub log_level: String,
}

#[derive(Debug, Deserialize)]
struct RawServerConfig {
    listen: Option<String>,
    db_path: PathBuf,
    shared_token: String,
    allowed_hosts: Option<Vec<String>>,
    max_body_bytes: Option<usize>,
    max_batch_records: Option<usize>,
    log_level: Option<String>,
}

#[derive(Clone)]
pub struct AppState {
    config: Arc<ServerConfig>,
    pool: DbPool,
    started_at: Instant,
    rate_limiter: Arc<Mutex<HashMap<String, BucketState>>>,
}

#[derive(Debug, Clone)]
struct BucketState {
    tokens: f64,
    last_refill: Instant,
}

impl ServerConfig {
    pub fn load_from_path(path: &Path) -> Result<Self, BoxError> {
        let content = std::fs::read_to_string(path)?;
        let raw: RawServerConfig = serde_yaml::from_str(&content)?;
        if raw.shared_token.chars().count() < 32 {
            return Err("shared_token must be at least 32 characters".into());
        }
        Ok(Self {
            listen: raw.listen.unwrap_or_else(|| "127.0.0.1:8787".to_string()),
            db_path: raw.db_path,
            shared_token: raw.shared_token,
            allowed_hosts: raw.allowed_hosts.map(|hosts| hosts.into_iter().collect()),
            max_body_bytes: raw.max_body_bytes.unwrap_or(1_048_576),
            max_batch_records: raw.max_batch_records.unwrap_or(1000),
            log_level: raw.log_level.unwrap_or_else(|| "info".to_string()),
        })
    }
}

impl AppState {
    pub fn new(config: ServerConfig) -> Result<Self, BoxError> {
        if config.shared_token.chars().count() < 32 {
            return Err("shared_token must be at least 32 characters".into());
        }
        if let Some(parent) = config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let manager = SqliteConnectionManager::file(&config.db_path).with_init(|conn| {
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            conn.pragma_update(None, "busy_timeout", 5000)?;
            conn.pragma_update(None, "foreign_keys", "ON")?;
            conn.pragma_update(None, "journal_size_limit", 67_108_864)?;
            Ok(())
        });
        let pool = Pool::builder()
            .max_size(4)
            .connection_timeout(Duration::from_secs(5))
            .build(manager)?;
        {
            let conn = pool.get()?;
            conn.execute_batch(include_str!("../migrations/0001_init.sql"))?;
            ensure_fast_tier_column(&conn)?;
            ensure_cost_columns(&conn)?;
        }
        Ok(Self {
            config: Arc::new(config),
            pool,
            started_at: Instant::now(),
            rate_limiter: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

pub fn build_app(state: AppState) -> Router {
    let max_body_bytes = state.config.max_body_bytes;
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/upload", post(upload))
        .route("/v1/pull", get(pull))
        .route("/v1/machines", get(machines))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: Option<String>,
}

impl AppError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: Some(message.into()),
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: None,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self.message {
            Some(message) => (self.status, message).into_response(),
            None => self.status.into_response(),
        }
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(err: rusqlite::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

impl From<r2d2::Error> for AppError {
    fn from(err: r2d2::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        schema_version: SCHEMA_VERSION,
        uptime_seconds: state.started_at.elapsed().as_secs(),
    })
}

async fn upload(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Result<Json<UploadResponse>, AppError> {
    authorize(&state, &headers)?;
    let mut records = Vec::new();
    for (idx, line) in body
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        if records.len() + 1 > state.config.max_batch_records {
            return Err(AppError::new(
                StatusCode::BAD_REQUEST,
                "batch exceeds max_batch_records",
            ));
        }
        let record: WireRecord = serde_json::from_str(line).map_err(|err| {
            AppError::new(
                StatusCode::BAD_REQUEST,
                format!("line {}: invalid JSON: {err}", idx + 1),
            )
        })?;
        record.validate().map_err(|err| {
            AppError::new(StatusCode::BAD_REQUEST, format!("line {}: {err}", idx + 1))
        })?;
        if state
            .config
            .allowed_hosts
            .as_ref()
            .is_some_and(|hosts| !hosts.contains(&record.host_id))
        {
            return Err(AppError::new(
                StatusCode::FORBIDDEN,
                "host_id is not allowed",
            ));
        }
        records.push(record);
    }

    let mut conn = state.pool.get()?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let uploaded_at = Utc::now().to_rfc3339();
    let mut accepted = 0usize;
    let mut ignored = 0usize;
    let mut touched_hosts = BTreeSet::new();
    let mut consumed_omp_v220_keys = BTreeSet::new();

    for record in &records {
        if uploaded_with_omp_alias(&tx, record, &mut consumed_omp_v220_keys)? {
            ignored += 1;
            touched_hosts.insert(record.host_id.clone());
            continue;
        }
        let changed = tx.execute(
            "INSERT OR IGNORE INTO records (
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
        )?;
        if changed == 1 {
            accepted += 1;
        } else {
            ignored += 1;
        }
        touched_hosts.insert(record.host_id.clone());
    }

    for host_id in touched_hosts {
        tx.execute(
            "INSERT INTO machines (host_id, last_seen, record_count)
             VALUES (?1, ?2, (SELECT COUNT(*) FROM records WHERE host_id = ?1))
             ON CONFLICT(host_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                record_count = (SELECT COUNT(*) FROM records WHERE host_id = ?1)",
            params![host_id, uploaded_at],
        )?;
    }

    let max_seq = max_seq_in_tx(&tx)?;
    tx.commit()?;

    Ok(Json(UploadResponse {
        accepted,
        ignored,
        max_seq,
    }))
}

#[derive(Debug, Deserialize)]
struct PullQuery {
    after_seq: u64,
    #[serde(default)]
    exclude_host: Vec<String>,
    limit: Option<usize>,
    supported_vendors: Vec<String>,
}

async fn pull(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
) -> Result<Json<PullResponse>, AppError> {
    authorize(&state, &headers)?;
    let query = parse_pull_query(raw_query.as_deref())?;
    let limit = query.limit.unwrap_or(5000).clamp(1, 20_000);
    let fetch_limit = limit + 1;
    let mut conn = state.pool.get()?;
    // The page query and the global max_seq query must observe the same
    // database snapshot. Without an enclosing transaction, SQLite WAL would
    // begin a fresh read snapshot for each SELECT, letting writes from other
    // connections appear between them; the response would then advertise a
    // max_seq that includes records the page query never saw, causing the
    // client to skip them on the next pull.
    let tx = conn.transaction_with_behavior(TransactionBehavior::Deferred)?;
    let mut sql = String::from(
        "SELECT seq, host_id, vendor, dedup_key, schema_version, timestamp_utc,
            session_start, session_end, model, effort, fast_tier, input_tokens, output_tokens,
            cache_read, cache_creation, reasoning_out, cost_input, cost_output, cost_cache_read,
            cost_cache_creation, project_hash, uploaded_at
         FROM records
         WHERE seq > ?",
    );
    sql.push_str(" AND vendor IN (");
    append_placeholders(&mut sql, query.supported_vendors.len());
    sql.push(')');
    for _ in &query.exclude_host {
        sql.push_str(" AND host_id != ?");
    }
    sql.push_str(" ORDER BY seq ASC LIMIT ?");

    let mut values = Vec::new();
    values.push(Value::Integer(query.after_seq as i64));
    for vendor in &query.supported_vendors {
        values.push(Value::Text(vendor.clone()));
    }
    for host in &query.exclude_host {
        values.push(Value::Text(host.clone()));
    }
    values.push(Value::Integer(fetch_limit as i64));

    let mut records: Vec<SequencedWireRecord> = {
        let mut stmt = tx.prepare(&sql)?;
        stmt.query_map(params_from_iter(values.iter()), row_to_sequenced_record)?
            .collect::<Result<_, _>>()?
    };
    let truncated = records.len() > limit;
    if truncated {
        records.truncate(limit);
    }
    let snapshot_max_seq = max_seq_in_tx_for_vendors(&tx, &query.supported_vendors)?;
    tx.commit()?;
    let response_max_seq = if truncated {
        records
            .last()
            .map(|record| record.seq)
            .unwrap_or(query.after_seq)
    } else {
        snapshot_max_seq.max(query.after_seq)
    };

    Ok(Json(PullResponse {
        records,
        max_seq: response_max_seq,
        truncated,
    }))
}

fn parse_pull_query(raw_query: Option<&str>) -> Result<PullQuery, AppError> {
    let mut after_seq = None;
    let mut exclude_host = Vec::new();
    let mut limit = None;
    let mut supported_vendors = None::<Vec<String>>;

    for part in raw_query
        .unwrap_or("")
        .split('&')
        .filter(|part| !part.is_empty())
    {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        match key {
            "after_seq" => {
                after_seq = Some(value.parse::<u64>().map_err(|_| {
                    AppError::new(StatusCode::BAD_REQUEST, "after_seq must be an integer")
                })?);
            }
            "exclude_host" => exclude_host.push(value.to_string()),
            "supported_vendors" => {
                let vendors = supported_vendors.get_or_insert_with(Vec::new);
                for vendor in value.split(',').filter(|vendor| !vendor.is_empty()) {
                    if !is_valid_vendor(vendor) {
                        return Err(AppError::new(
                            StatusCode::BAD_REQUEST,
                            "supported_vendors contains invalid vendor",
                        ));
                    }
                    if !vendors.iter().any(|seen| seen == vendor) {
                        vendors.push(vendor.to_string());
                    }
                }
            }
            "limit" => {
                limit = Some(value.parse::<usize>().map_err(|_| {
                    AppError::new(StatusCode::BAD_REQUEST, "limit must be an integer")
                })?);
            }
            _ => {}
        }
    }

    let Some(after_seq) = after_seq else {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "after_seq is required",
        ));
    };

    let supported_vendors = supported_vendors.unwrap_or_else(|| {
        LEGACY_PULL_VENDORS
            .iter()
            .map(|vendor| (*vendor).to_string())
            .collect()
    });
    if supported_vendors.is_empty() {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "supported_vendors must not be empty",
        ));
    }

    Ok(PullQuery {
        after_seq,
        exclude_host,
        limit,
        supported_vendors,
    })
}

async fn machines(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<MachineList>, AppError> {
    authorize(&state, &headers)?;
    let conn = state.pool.get()?;
    let mut stmt =
        conn.prepare("SELECT host_id, last_seen, record_count FROM machines ORDER BY host_id ASC")?;
    let machines = stmt
        .query_map([], |row| {
            Ok(MachineInfo {
                host_id: row.get(0)?,
                last_seen: row.get(1)?,
                record_count: row.get::<_, i64>(2)? as u64,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Json(MachineList { machines }))
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), AppError> {
    let Some(value) = headers.get(axum::http::header::AUTHORIZATION) else {
        return Err(AppError::unauthorized());
    };
    let Ok(value) = value.to_str() else {
        return Err(AppError::unauthorized());
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return Err(AppError::unauthorized());
    };
    let expected = state.config.shared_token.as_bytes();
    let provided = token.as_bytes();
    if provided.len() == expected.len() && provided.ct_eq(expected).into() {
        check_rate_limit(state, token)
    } else {
        Err(AppError::unauthorized())
    }
}

fn check_rate_limit(state: &AppState, token: &str) -> Result<(), AppError> {
    let now = Instant::now();
    let mut buckets = state
        .rate_limiter
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let bucket = buckets.entry(token.to_string()).or_insert(BucketState {
        tokens: RATE_LIMIT_BURST,
        last_refill: now,
    });

    let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
    bucket.tokens = (bucket.tokens + elapsed * RATE_LIMIT_REFILL_PER_SECOND).min(RATE_LIMIT_BURST);
    bucket.last_refill = now;

    if bucket.tokens >= 1.0 {
        bucket.tokens -= 1.0;
        Ok(())
    } else {
        Err(AppError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate limit exceeded",
        ))
    }
}

fn row_to_sequenced_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<SequencedWireRecord> {
    Ok(SequencedWireRecord {
        seq: row.get::<_, i64>(0)? as u64,
        record: WireRecord {
            host_id: row.get(1)?,
            vendor: row.get(2)?,
            dedup_key: row.get(3)?,
            schema_version: row.get::<_, i64>(4)? as u32,
            timestamp: row.get(5)?,
            session_start_time: row.get(6)?,
            session_end_time: row.get(7)?,
            model: row.get(8)?,
            effort: row.get(9)?,
            fast_tier: row.get(10)?,
            input_tokens: row.get(11)?,
            output_tokens: row.get(12)?,
            cache_read_input_tokens: row.get(13)?,
            cache_creation_input_tokens: row.get(14)?,
            reasoning_output_tokens: row.get(15)?,
            cost_input: row.get(16)?,
            cost_output: row.get(17)?,
            cost_cache_read: row.get(18)?,
            cost_cache_creation: row.get(19)?,
            project_path_sha256: row.get(20)?,
        },
        uploaded_at: row.get(21)?,
    })
}

fn ensure_fast_tier_column(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(records)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !columns.iter().any(|column| column == "fast_tier") {
        conn.execute(
            "ALTER TABLE records ADD COLUMN fast_tier INTEGER NOT NULL DEFAULT -1",
            [],
        )?;
    }
    Ok(())
}

fn ensure_cost_columns(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(records)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    for (name, sql_type) in [
        ("cost_input", "REAL"),
        ("cost_output", "REAL"),
        ("cost_cache_read", "REAL"),
        ("cost_cache_creation", "REAL"),
    ] {
        if !columns.iter().any(|column| column == name) {
            conn.execute(
                &format!("ALTER TABLE records ADD COLUMN {name} {sql_type}"),
                [],
            )?;
        }
    }
    Ok(())
}

fn max_seq_in_tx(tx: &rusqlite::Transaction<'_>) -> rusqlite::Result<u64> {
    tx.query_row("SELECT COALESCE(MAX(seq), 0) FROM records", [], |row| {
        Ok(row.get::<_, i64>(0)? as u64)
    })
    .optional()
    .map(|value| value.unwrap_or(0))
}

fn max_seq_in_tx_for_vendors(
    tx: &rusqlite::Transaction<'_>,
    supported_vendors: &[String],
) -> rusqlite::Result<u64> {
    let mut sql = String::from("SELECT COALESCE(MAX(seq), 0) FROM records WHERE vendor IN (");
    append_placeholders(&mut sql, supported_vendors.len());
    sql.push(')');
    let values = supported_vendors
        .iter()
        .map(|vendor| Value::Text(vendor.clone()))
        .collect::<Vec<_>>();
    tx.query_row(&sql, params_from_iter(values.iter()), |row| {
        Ok(row.get::<_, i64>(0)? as u64)
    })
    .optional()
    .map(|value| value.unwrap_or(0))
}

fn append_placeholders(sql: &mut String, count: usize) {
    for idx in 0..count {
        if idx > 0 {
            sql.push_str(", ");
        }
        sql.push('?');
    }
}

fn uploaded_with_omp_alias(
    tx: &rusqlite::Transaction<'_>,
    record: &WireRecord,
    consumed_keys: &mut BTreeSet<String>,
) -> rusqlite::Result<bool> {
    if record.vendor != "omp" {
        return Ok(false);
    }
    if let Some(key) = parse_omp_v220_key(&record.dedup_key) {
        if !omp_v220_key_matches_record(&key, record) {
            return Ok(false);
        }
        if let Some(stable_key) = omp_stable_key_from_v220_key(&key) {
            return omp_stable_key_exists(tx, &record.host_id, &stable_key);
        }
        return omp_stable_file_key_exists(tx, record, &key);
    }
    let consume_once = record.dedup_key.starts_with("omp:file:");
    for legacy_key in omp_v220_key_candidates(record) {
        let scoped_key = format!("{}:{legacy_key}", record.host_id);
        let exists = tx
            .query_row(
                "SELECT 1 FROM records WHERE host_id = ?1 AND vendor = 'omp' AND dedup_key = ?2 LIMIT 1",
                params![record.host_id, legacy_key],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if exists && (!consume_once || consumed_keys.insert(scoped_key)) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[derive(Debug, Deserialize)]
struct OmpV220Key {
    #[serde(rename = "message")]
    message_id: String,
    #[serde(rename = "response")]
    response_id: String,
    model: String,
    #[serde(rename = "input")]
    input_tokens: i64,
    #[serde(rename = "output")]
    output_tokens: i64,
    #[serde(rename = "cache_read")]
    cache_read_input_tokens: i64,
    #[serde(rename = "cache_write")]
    cache_creation_input_tokens: i64,
}

fn parse_omp_v220_key(dedup_key: &str) -> Option<OmpV220Key> {
    serde_json::from_str(dedup_key).ok()
}

fn omp_v220_key_matches_record(key: &OmpV220Key, record: &WireRecord) -> bool {
    key.input_tokens == record.input_tokens
        && key.output_tokens == record.output_tokens
        && key.cache_read_input_tokens == record.cache_read_input_tokens
        && key.cache_creation_input_tokens == record.cache_creation_input_tokens
        && omp_model_candidates(record)
            .into_iter()
            .any(|model| model == key.model)
}

fn omp_stable_key_from_v220_key(key: &OmpV220Key) -> Option<String> {
    match (key.message_id.is_empty(), key.response_id.is_empty()) {
        (false, false) => Some(format!(
            "omp:message:{}:response:{}",
            key.message_id, key.response_id
        )),
        (false, true) => Some(format!("omp:message:{}", key.message_id)),
        (true, false) => Some(format!("omp:response:{}", key.response_id)),
        (true, true) => None,
    }
}

fn omp_stable_key_exists(
    tx: &rusqlite::Transaction<'_>,
    host_id: &str,
    stable_key: &str,
) -> rusqlite::Result<bool> {
    tx.query_row(
        "SELECT 1 FROM records WHERE host_id = ?1 AND vendor = 'omp' AND dedup_key = ?2 LIMIT 1",
        params![host_id, stable_key],
        |_| Ok(()),
    )
    .optional()
    .map(|value| value.is_some())
}

fn omp_stable_file_key_exists(
    tx: &rusqlite::Transaction<'_>,
    record: &WireRecord,
    key: &OmpV220Key,
) -> rusqlite::Result<bool> {
    tx.query_row(
        "SELECT 1 FROM records
         WHERE host_id = ?1
           AND vendor = 'omp'
           AND dedup_key LIKE 'omp:file:%'
           AND model = ?2
           AND input_tokens = ?3
           AND output_tokens = ?4
           AND cache_read = ?5
           AND cache_creation = ?6
         LIMIT 1",
        params![
            record.host_id,
            omp_normalized_model(&key.model),
            record.input_tokens,
            record.output_tokens,
            record.cache_read_input_tokens,
            record.cache_creation_input_tokens,
        ],
        |_| Ok(()),
    )
    .optional()
    .map(|value| value.is_some())
}

fn omp_v220_key_candidates(record: &WireRecord) -> Vec<String> {
    let (message_id, response_id) = omp_ids_from_dedup_key(&record.dedup_key);
    omp_model_candidates(record)
        .into_iter()
        .map(|model| {
            serde_json::json!({
                "message": message_id,
                "response": response_id,
                "model": model,
                "input": record.input_tokens,
                "output": record.output_tokens,
                "cache_read": record.cache_read_input_tokens,
                "cache_write": record.cache_creation_input_tokens,
            })
            .to_string()
        })
        .collect()
}

fn omp_model_candidates(record: &WireRecord) -> Vec<String> {
    let mut models = vec![record.model.clone()];
    if let Some(provider) = record.effort.as_deref().filter(|value| !value.is_empty()) {
        models.push(format!("{provider}/{}", record.model));
    }
    models.sort();
    models.dedup();
    models
}

fn omp_normalized_model(raw_model: &str) -> &str {
    raw_model
        .split_once('/')
        .and_then(|(_, model)| (!model.is_empty()).then_some(model))
        .unwrap_or(raw_model)
}

fn omp_ids_from_dedup_key(dedup_key: &str) -> (String, String) {
    if let Some(rest) = dedup_key.strip_prefix("omp:message:") {
        if let Some((message_id, response_id)) = rest.split_once(":response:") {
            return (message_id.to_string(), response_id.to_string());
        }
        return (rest.to_string(), String::new());
    }
    if let Some(response_id) = dedup_key.strip_prefix("omp:response:") {
        return (String::new(), response_id.to_string());
    }
    (String::new(), String::new())
}
