use crate::sync::config::EnabledSyncConfig;
use crate::sync::engine::{SyncError, SyncTransport};
use serde::de::DeserializeOwned;
use std::time::Duration;
use vibe_usage_proto::{MachineList, PullResponse, UploadResponse, WireRecord};

#[derive(Clone)]
pub struct SyncHttpClient {
    agent: ureq::Agent,
    server_url: String,
    token: String,
}

impl SyncHttpClient {
    pub fn new(config: EnabledSyncConfig) -> Self {
        let timeout = Duration::from_secs(config.request_timeout_seconds);
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build()
            .new_agent();
        Self {
            agent,
            server_url: config.server_url.trim_end_matches('/').to_string(),
            token: config.token,
        }
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.server_url, path)
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.token)
    }

    pub fn machines(&self) -> Result<MachineList, SyncError> {
        let response = self
            .agent
            .get(&self.endpoint("/v1/machines"))
            .header("Authorization", self.auth_header())
            .call()
            .map_err(transport_error)?;
        read_json_response(response)
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
        let response = self
            .agent
            .post(&self.endpoint("/v1/upload"))
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/x-ndjson")
            .send(body)
            .map_err(transport_error)?;
        read_json_response(response)
    }

    fn pull(
        &self,
        after_seq: u64,
        exclude_host: &str,
        limit: usize,
    ) -> Result<PullResponse, SyncError> {
        let path =
            format!("/v1/pull?after_seq={after_seq}&exclude_host={exclude_host}&limit={limit}");
        let response = self
            .agent
            .get(&self.endpoint(&path))
            .header("Authorization", self.auth_header())
            .call()
            .map_err(transport_error)?;
        read_json_response(response)
    }
}

fn read_json_response<T: DeserializeOwned>(
    mut response: ureq::http::Response<ureq::Body>,
) -> Result<T, SyncError> {
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|err| SyncError::new(err.to_string()))?;
    serde_json::from_str(&body).map_err(|err| SyncError::new(err.to_string()))
}

fn transport_error(err: ureq::Error) -> SyncError {
    SyncError::new(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::config::EnabledSyncConfig;
    use crate::sync::engine::SyncTransport;
    use std::collections::HashSet;
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
        WireRecord {
            schema_version: SCHEMA_VERSION,
            host_id: host_id.to_string(),
            vendor: "claude".to_string(),
            dedup_key: dedup_key.to_string(),
            timestamp: "2026-05-18T12:00:00Z".to_string(),
            session_start_time: "2026-05-18T12:00:00Z".to_string(),
            session_end_time: "2026-05-18T12:05:00Z".to_string(),
            model: "test-model".to_string(),
            effort: None,
            input_tokens: 1,
            output_tokens: 2,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 4,
            reasoning_output_tokens: 5,
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
}
