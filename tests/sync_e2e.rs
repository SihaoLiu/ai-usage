use ai_usage_server::{AppState, AutoUpdateConfig, ServerConfig, build_app};
use chrono::{Duration, Utc};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

fn unique_temp_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ai-usage-e2e-{name}-{stamp}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn server_config(db_path: PathBuf) -> ServerConfig {
    ServerConfig {
        listen: "127.0.0.1:0".to_string(),
        db_path,
        shared_token: TOKEN.to_string(),
        allowed_hosts: Some(HashSet::from(["host-a".to_string(), "host-b".to_string()])),
        max_body_bytes: 1024 * 1024,
        max_batch_records: 1000,
        log_level: "info".to_string(),
        auto_update: AutoUpdateConfig::default(),
    }
}

fn write_client_config(home: &Path, server_url: &str, machine_id: &str) {
    fs::write(
        home.join(".fee.env"),
        "CLAUDE_MONTHLY_FEE=0\nCODEX_MONTHLY_FEE=0\nGEMINI_MONTHLY_FEE=0\nKIMI_MONTHLY_FEE=0\n",
    )
    .expect("write fee config");
    let path = home.join(".secrets").join("ai-usage.yaml");
    fs::create_dir_all(path.parent().expect("secrets parent")).expect("create secrets dir");
    fs::write(
        &path,
        format!(
            "sync:\n  enabled: true\n  server_url: \"{server_url}\"\n  token: \"{TOKEN}\"\n  machine_id: \"{machine_id}\"\n  upload_project_hash: false\n  request_timeout_seconds: 5\n"
        ),
    )
    .expect("write client config");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .expect("set client config permissions");
    }
}

fn write_claude_usage(home: &Path, model: &str, timestamp: &str) {
    let dir = home
        .join(".config")
        .join("claude")
        .join("projects")
        .join("e2e");
    fs::create_dir_all(&dir).expect("create usage dir");
    let line = format!(
        r#"{{"timestamp":"{timestamp}","requestId":"req-a","message":{{"id":"msg-a","model":"{model}","usage":{{"input_tokens":123,"output_tokens":45,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}}}}"#
    );
    fs::write(dir.join("session.jsonl"), format!("{line}\n")).expect("write usage file");
}

fn write_claude_usage_without_ids(home: &Path) {
    let dir = home
        .join(".config")
        .join("claude")
        .join("projects")
        .join("e2e-empty-ids");
    fs::create_dir_all(&dir).expect("create usage dir");
    let first_timestamp = (Utc::now() - Duration::days(2)).to_rfc3339();
    let second_timestamp = (Utc::now() - Duration::days(2) + Duration::minutes(1)).to_rfc3339();
    let first = format!(
        r#"{{"timestamp":"{first_timestamp}","message":{{"model":"empty-id-model","usage":{{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}}}}"#
    );
    let second = format!(
        r#"{{"timestamp":"{second_timestamp}","message":{{"model":"empty-id-model","usage":{{"input_tokens":20,"output_tokens":2,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}}}}"#
    );
    fs::write(dir.join("session.jsonl"), format!("{first}\n{second}\n")).expect("write usage file");
}

fn ensure_claude_dir(home: &Path) {
    fs::create_dir_all(home.join(".config").join("claude").join("projects"))
        .expect("create claude dir");
}

fn run_cli(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ai-usage"))
        .args(args)
        .current_dir(home)
        .env("HOME", home)
        .env("AI_USAGE_CACHE_DIR", home.join(".cache").join("ai-usage"))
        .env(
            "AI_USAGE_SECRETS",
            home.join(".secrets").join("ai-usage.yaml"),
        )
        .env("AI_USAGE_ALLOW_INSECURE_HTTP_FOR_TESTS", "1")
        .env("COLUMNS", "140")
        .env("LINES", "50")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .expect("run ai-usage")
}

fn assert_success(output: Output, label: &str) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "{label} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    stdout
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_homes_exchange_usage_through_local_server() {
    let root = unique_temp_dir("exchange");
    let home_a = root.join("home-a");
    let home_b = root.join("home-b");
    fs::create_dir_all(&home_a).expect("create home a");
    fs::create_dir_all(&home_b).expect("create home b");

    let state = AppState::new(server_config(root.join("server.db"))).expect("server state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("server addr");
    let server = tokio::spawn(async move {
        axum::serve(listener, build_app(state))
            .await
            .expect("server");
    });
    let server_url = format!("http://{addr}");

    write_client_config(&home_a, &server_url, "host-a");
    write_client_config(&home_b, &server_url, "host-b");
    ensure_claude_dir(&home_b);
    write_claude_usage(&home_a, "e2e-model", &Utc::now().to_rfc3339());

    assert_success(run_cli(&home_a, &["sync", "push"]), "host-a push");
    assert_success(run_cli(&home_b, &["sync", "pull"]), "host-b pull");
    let output = assert_success(
        run_cli(&home_b, &["--once", "--tool", "claude", "--host", "host-a"]),
        "host-b display",
    );

    // Unknown model ids render as derived title-case labels.
    assert!(output.contains("E2e Model"), "{output}");
    assert!(
        home_b
            .join(".cache")
            .join("ai-usage")
            .join("remote")
            .join("host-a.bin")
            .exists()
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_clean_refetches_records_from_server_after_local_wipe() {
    let root = unique_temp_dir("clean");
    let home_a = root.join("home-a");
    let home_b = root.join("home-b");
    fs::create_dir_all(&home_a).expect("create home a");
    fs::create_dir_all(&home_b).expect("create home b");

    let state = AppState::new(server_config(root.join("server.db"))).expect("server state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("server addr");
    let server = tokio::spawn(async move {
        axum::serve(listener, build_app(state))
            .await
            .expect("server");
    });
    let server_url = format!("http://{addr}");

    write_client_config(&home_a, &server_url, "host-a");
    write_client_config(&home_b, &server_url, "host-b");
    ensure_claude_dir(&home_b);
    write_claude_usage(&home_a, "clean-model", &Utc::now().to_rfc3339());

    assert_success(run_cli(&home_a, &["sync", "push"]), "host-a push");
    assert_success(run_cli(&home_b, &["sync", "pull"]), "host-b pull");
    let remote_path = home_b
        .join(".cache")
        .join("ai-usage")
        .join("remote")
        .join("host-a.bin");
    let state_path = home_b
        .join(".cache")
        .join("ai-usage")
        .join("sync_state.json");
    assert!(
        remote_path.exists(),
        "first pull must populate remote cache"
    );
    assert!(state_path.exists(), "first pull must persist cursor");

    fs::write(&remote_path, b"corrupted contents").expect("corrupt remote cache");

    let clean_output = assert_success(run_cli(&home_b, &["sync", "clean"]), "host-b clean");
    assert!(
        clean_output.contains("sync clean complete"),
        "clean stdout: {clean_output}"
    );
    assert!(
        remote_path.exists(),
        "clean must repopulate remote cache from server"
    );

    let after_clean = assert_success(
        run_cli(&home_b, &["--once", "--tool", "claude", "--host", "host-a"]),
        "host-b display after clean",
    );
    // Unknown model ids render as derived title-case labels.
    assert!(after_clean.contains("Clean Model"), "{after_clean}");

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn integrity_passes_for_claude_records_without_source_ids() {
    let root = unique_temp_dir("empty-source-ids");
    let home_a = root.join("home-a");
    let home_b = root.join("home-b");
    fs::create_dir_all(&home_a).expect("create home a");
    fs::create_dir_all(&home_b).expect("create home b");

    let state = AppState::new(server_config(root.join("server.db"))).expect("server state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("server addr");
    let server = tokio::spawn(async move {
        axum::serve(listener, build_app(state))
            .await
            .expect("server");
    });
    let server_url = format!("http://{addr}");

    write_client_config(&home_a, &server_url, "host-a");
    write_client_config(&home_b, &server_url, "host-b");
    ensure_claude_dir(&home_b);
    write_claude_usage_without_ids(&home_a);

    assert_success(run_cli(&home_a, &["sync", "push"]), "host-a push");
    assert_success(run_cli(&home_b, &["sync", "pull"]), "host-b pull");

    let transcript_path = home_b
        .join(".cache")
        .join("ai-usage")
        .join("integrity")
        .join("remote-host-a.jsonl");
    let transcript = fs::read_to_string(&transcript_path).expect("read remote transcript");
    let lines = transcript
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("parse transcript"))
        .collect::<Vec<_>>();

    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["status"], "checked");
    assert_eq!(lines[0]["expected_record_count"], 2);
    assert_eq!(lines[0]["actual_record_count"], 2);

    server.abort();
}
