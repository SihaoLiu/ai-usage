use crate::sync::config::EnabledSyncConfig;
use crate::sync::engine::{SUPPORTED_PULL_VENDORS, SyncError, SyncTransport};
use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::time::Duration;
use vibe_usage_proto::{MachineList, PullResponse, UploadResponse, WireRecord};

const MAX_RATE_LIMIT_RETRIES: usize = 5;
const DEFAULT_RATE_LIMIT_RETRY_DELAY: Duration = Duration::from_millis(1100);
const MAX_RATE_LIMIT_RETRY_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpProgress {
    RateLimited {
        attempt: usize,
        retry_after: Duration,
    },
}

type HttpProgressCallback = Arc<dyn Fn(&HttpProgress) + Send + Sync>;

#[derive(Clone)]
pub struct SyncHttpClient {
    agent: ureq::Agent,
    server_url: String,
    token: String,
    progress: Option<HttpProgressCallback>,
}

impl SyncHttpClient {
    pub fn new(config: EnabledSyncConfig) -> Self {
        Self::new_inner(config, None)
    }

    pub fn new_with_progress<F>(config: EnabledSyncConfig, progress: F) -> Self
    where
        F: Fn(&HttpProgress) + Send + Sync + 'static,
    {
        Self::new_inner(config, Some(Arc::new(progress)))
    }

    fn new_inner(config: EnabledSyncConfig, progress: Option<HttpProgressCallback>) -> Self {
        let timeout = Duration::from_secs(config.request_timeout_seconds);
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .http_status_as_error(false)
            .build()
            .new_agent();
        Self {
            agent,
            server_url: config.server_url.trim_end_matches('/').to_string(),
            token: config.token,
            progress,
        }
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.server_url, path)
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.token)
    }

    pub fn machines(&self) -> Result<MachineList, SyncError> {
        let response = self.call_with_rate_limit_retry(|| {
            self.agent
                .get(&self.endpoint("/v1/machines"))
                .header("Authorization", self.auth_header())
                .call()
        })?;
        read_json_response(response)
    }

    fn call_with_rate_limit_retry<F>(
        &self,
        mut send: F,
    ) -> Result<ureq::http::Response<ureq::Body>, SyncError>
    where
        F: FnMut() -> Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    {
        for attempt in 0..=MAX_RATE_LIMIT_RETRIES {
            let response = send().map_err(transport_error)?;
            if response.status().as_u16() == 429 && attempt < MAX_RATE_LIMIT_RETRIES {
                let retry_after = rate_limit_retry_delay(&response);
                self.emit_progress(HttpProgress::RateLimited {
                    attempt: attempt + 1,
                    retry_after,
                });
                std::thread::sleep(retry_after);
                continue;
            }
            return Ok(response);
        }
        unreachable!("rate limit retry loop always returns")
    }

    fn emit_progress(&self, event: HttpProgress) {
        if let Some(progress) = self.progress.as_ref() {
            progress(&event);
        }
    }
}

impl SyncTransport for SyncHttpClient {
    fn upload(&self, records: &[WireRecord]) -> Result<UploadResponse, SyncError> {
        let body = records
            .iter()
            .map(|record| {
                serde_json::to_string(record).map_err(|err| SyncError::new(err.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");
        let response = self.call_with_rate_limit_retry(|| {
            self.agent
                .post(&self.endpoint("/v1/upload"))
                .header("Authorization", self.auth_header())
                .header("Content-Type", "application/x-ndjson")
                .send(body.clone())
        })?;
        read_json_response(response)
    }

    fn pull(
        &self,
        after_seq: u64,
        exclude_host: &str,
        limit: usize,
    ) -> Result<PullResponse, SyncError> {
        let supported_vendors = SUPPORTED_PULL_VENDORS.join(",");
        let path = format!(
            "/v1/pull?after_seq={after_seq}&exclude_host={exclude_host}&limit={limit}&supported_vendors={supported_vendors}"
        );
        let response = self.call_with_rate_limit_retry(|| {
            self.agent
                .get(&self.endpoint(&path))
                .header("Authorization", self.auth_header())
                .call()
        })?;
        read_json_response(response)
    }
}

fn read_json_response<T: DeserializeOwned>(
    mut response: ureq::http::Response<ureq::Body>,
) -> Result<T, SyncError> {
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.body_mut().read_to_string().unwrap_or_default();
        let trimmed = body.trim();
        if !trimmed.is_empty() {
            return Err(SyncError::new(format!("http status: {status}: {trimmed}")));
        }
        return Err(SyncError::new(format!("http status: {}", status)));
    }
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|err| SyncError::new(err.to_string()))?;
    serde_json::from_str(&body).map_err(|err| SyncError::new(err.to_string()))
}

fn rate_limit_retry_delay(response: &ureq::http::Response<ureq::Body>) -> Duration {
    response
        .headers()
        .get("Retry-After")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_RATE_LIMIT_RETRY_DELAY)
        .min(MAX_RATE_LIMIT_RETRY_DELAY)
}

fn transport_error(err: ureq::Error) -> SyncError {
    SyncError::new(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::config::EnabledSyncConfig;
    use crate::sync::engine::SyncTransport;
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use axum::{Json, Router};
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use vibe_usage_proto::{SCHEMA_VERSION, WireRecord};
    use vibe_usage_server::{AppState, ServerConfig, build_app};

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    fn unique_db_path(name: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("vibe-usage-client-test-{name}-{stamp}.db"))
    }

    fn server_config(name: &str) -> ServerConfig {
        ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            db_path: unique_db_path(name),
            shared_token: TOKEN.to_string(),
            allowed_hosts: Some(HashSet::from([
                "laptop".to_string(),
                "workstation".to_string(),
            ])),
            max_body_bytes: 1024 * 1024,
            max_batch_records: 1000,
            log_level: "info".to_string(),
        }
    }

    fn client_config(server_url: String) -> EnabledSyncConfig {
        EnabledSyncConfig {
            server_url,
            token: TOKEN.to_string(),
            machine_id: "workstation".to_string(),
            upload_project_hash: false,
            request_timeout_seconds: 15,
        }
    }

    fn record(host_id: &str, dedup_key: &str) -> WireRecord {
        record_with_vendor(host_id, "claude", dedup_key)
    }

    fn record_with_vendor(host_id: &str, vendor: &str, dedup_key: &str) -> WireRecord {
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
            input_tokens: 1,
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_client_uploads_and_pulls_records() {
        let state = AppState::new(server_config("transport")).expect("server state");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let server = tokio::spawn(async move {
            axum::serve(listener, build_app(state))
                .await
                .expect("server");
        });
        let client = SyncHttpClient::new(client_config(format!("http://{addr}")));

        let upload_client = client.clone();
        let upload = tokio::task::spawn_blocking(move || {
            upload_client.upload(&[record("laptop", "remote-a")])
        })
        .await
        .expect("upload join")
        .expect("upload response");
        assert_eq!(upload.accepted, 1);
        assert_eq!(upload.ignored, 0);

        let pull_client = client.clone();
        let pull = tokio::task::spawn_blocking(move || pull_client.pull(0, "workstation", 100))
            .await
            .expect("pull join")
            .expect("pull response");

        assert_eq!(pull.records.len(), 1);
        assert_eq!(pull.records[0].record.host_id, "laptop");
        assert_eq!(pull.records[0].record.dedup_key, "remote-a");
        assert_eq!(pull.records[0].record.fast_tier, 1);

        let machines_client = client.clone();
        let machines = tokio::task::spawn_blocking(move || machines_client.machines())
            .await
            .expect("machines join")
            .expect("machines response");
        assert_eq!(machines.machines.len(), 1);
        assert_eq!(machines.machines[0].host_id, "laptop");
        assert_eq!(machines.machines[0].record_count, 1);
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_client_pull_requests_omp_records() {
        let state = AppState::new(server_config("transport-omp")).expect("server state");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let server = tokio::spawn(async move {
            axum::serve(listener, build_app(state))
                .await
                .expect("server");
        });
        let client = SyncHttpClient::new(client_config(format!("http://{addr}")));

        let upload_client = client.clone();
        tokio::task::spawn_blocking(move || {
            upload_client.upload(&[record_with_vendor("laptop", "omp", "remote-omp-a")])
        })
        .await
        .expect("upload join")
        .expect("upload response");

        let pull_client = client.clone();
        let pull = tokio::task::spawn_blocking(move || pull_client.pull(0, "workstation", 100))
            .await
            .expect("pull join")
            .expect("pull response");

        assert_eq!(pull.records.len(), 1);
        assert_eq!(pull.records[0].record.vendor, "omp");
        assert_eq!(pull.records[0].record.dedup_key, "remote-omp-a");
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upload_retries_once_after_rate_limit() {
        async fn upload(State(attempts): State<Arc<AtomicUsize>>) -> Response {
            let attempt = attempts.fetch_add(1, Ordering::Relaxed);
            if attempt == 0 {
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("Retry-After", "0")],
                    "rate limit exceeded",
                )
                    .into_response()
            } else {
                Json(UploadResponse {
                    accepted: 1,
                    ignored: 0,
                    max_seq: 1,
                })
                .into_response()
            }
        }

        let attempts = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let app = Router::new()
            .route("/v1/upload", post(upload))
            .with_state(attempts.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("server");
        });
        let client = SyncHttpClient::new(client_config(format!("http://{addr}")));

        let upload_client = client.clone();
        let upload = tokio::task::spawn_blocking(move || {
            upload_client.upload(&[record("laptop", "rate-limited")])
        })
        .await
        .expect("upload join")
        .expect("upload response");

        assert_eq!(upload.accepted, 1);
        assert_eq!(upload.ignored, 0);
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upload_reports_rate_limit_retry_progress() {
        async fn upload(State(attempts): State<Arc<AtomicUsize>>) -> Response {
            let attempt = attempts.fetch_add(1, Ordering::Relaxed);
            if attempt == 0 {
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("Retry-After", "0")],
                    "rate limit exceeded",
                )
                    .into_response()
            } else {
                Json(UploadResponse {
                    accepted: 1,
                    ignored: 0,
                    max_seq: 1,
                })
                .into_response()
            }
        }

        let attempts = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let app = Router::new()
            .route("/v1/upload", post(upload))
            .with_state(attempts);
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("server");
        });
        let retries = Arc::new(AtomicUsize::new(0));
        let retry_counter = Arc::clone(&retries);
        let client = SyncHttpClient::new_with_progress(
            client_config(format!("http://{addr}")),
            move |event| match event {
                HttpProgress::RateLimited {
                    attempt,
                    retry_after,
                } => {
                    assert_eq!(*attempt, 1);
                    assert_eq!(*retry_after, Duration::from_secs(0));
                    retry_counter.fetch_add(1, Ordering::Relaxed);
                }
            },
        );

        let upload_client = client.clone();
        let upload = tokio::task::spawn_blocking(move || {
            upload_client.upload(&[record("laptop", "rate-limited")])
        })
        .await
        .expect("upload join")
        .expect("upload response");

        assert_eq!(upload.accepted, 1);
        assert_eq!(retries.load(Ordering::Relaxed), 1);
        server.abort();
    }
}
