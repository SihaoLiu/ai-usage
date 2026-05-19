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
    UploadResponse, WireRecord,
};

type DbPool = Pool<SqliteConnectionManager>;
type BoxError = Box<dyn std::error::Error + Send + Sync>;
const RATE_LIMIT_REFILL_PER_SECOND: f64 = 1.0;
const RATE_LIMIT_BURST: f64 = 30.0;

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

    for record in &records {
        let changed = tx.execute(
            "INSERT OR IGNORE INTO records (
                host_id, vendor, dedup_key, schema_version, timestamp_utc,
                session_start, session_end, model, effort, fast_tier, input_tokens,
                output_tokens, cache_read, cache_creation, reasoning_out,
                project_hash, uploaded_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
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
    let conn = state.pool.get()?;
    let mut sql = String::from(
        "SELECT seq, host_id, vendor, dedup_key, schema_version, timestamp_utc,
            session_start, session_end, model, effort, fast_tier, input_tokens, output_tokens,
            cache_read, cache_creation, reasoning_out, project_hash, uploaded_at
         FROM records
         WHERE seq > ?1",
    );
    for _ in &query.exclude_host {
        sql.push_str(" AND host_id != ?");
    }
    sql.push_str(" ORDER BY seq ASC LIMIT ?");

    let mut values = Vec::new();
    values.push(Value::Integer(query.after_seq as i64));
    for host in &query.exclude_host {
        values.push(Value::Text(host.clone()));
    }
    values.push(Value::Integer(fetch_limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let mut records: Vec<SequencedWireRecord> = stmt
        .query_map(params_from_iter(values.iter()), row_to_sequenced_record)?
        .collect::<Result<_, _>>()?;
    let truncated = records.len() > limit;
    if truncated {
        records.truncate(limit);
    }
    let global_max_seq = max_seq(&conn)?;
    let response_max_seq = if truncated {
        records
            .last()
            .map(|record| record.seq)
            .unwrap_or(query.after_seq)
    } else {
        global_max_seq
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

    Ok(PullQuery {
        after_seq,
        exclude_host,
        limit,
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
            project_path_sha256: row.get(16)?,
        },
        uploaded_at: row.get(17)?,
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

fn max_seq(conn: &rusqlite::Connection) -> rusqlite::Result<u64> {
    conn.query_row("SELECT COALESCE(MAX(seq), 0) FROM records", [], |row| {
        Ok(row.get::<_, i64>(0)? as u64)
    })
    .optional()
    .map(|value| value.unwrap_or(0))
}

fn max_seq_in_tx(tx: &rusqlite::Transaction<'_>) -> rusqlite::Result<u64> {
    tx.query_row("SELECT COALESCE(MAX(seq), 0) FROM records", [], |row| {
        Ok(row.get::<_, i64>(0)? as u64)
    })
    .optional()
    .map(|value| value.unwrap_or(0))
}
