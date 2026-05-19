use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const DEFAULT_TIMEOUT_SECONDS: u64 = 15;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncConfig {
    Disabled,
    Enabled(EnabledSyncConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnabledSyncConfig {
    pub server_url: String,
    pub token: String,
    pub machine_id: String,
    pub upload_project_hash: bool,
    pub request_timeout_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigLoadResult {
    pub config: SyncConfig,
    pub warning: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SecretsFile {
    sync: Option<RawSyncConfig>,
}

#[derive(Debug, Deserialize)]
struct RawSyncConfig {
    server_url: Option<String>,
    token: Option<String>,
    machine_id: Option<String>,
    enabled: Option<bool>,
    upload_project_hash: Option<bool>,
    request_timeout_seconds: Option<u64>,
}

pub fn load_sync_config(debug_sync: bool) -> SyncConfig {
    let path = default_secrets_path();
    let fallback_hostname = default_hostname();
    let result = load_sync_config_from_path(&path, &fallback_hostname);
    if let Some(warning) = result.warning {
        eprintln!("{warning}");
    } else if debug_sync && matches!(result.config, SyncConfig::Disabled) {
        eprintln!(
            "vibe-usage: sync config not found at {}; sync disabled",
            path.display()
        );
    }
    result.config
}

pub fn sync_config_path() -> PathBuf {
    default_secrets_path()
}

pub fn init_sync_config(force: bool) -> io::Result<PathBuf> {
    let path = sync_config_path();
    let machine_id = sanitize_hostname(&default_hostname());
    write_sync_config_template(&path, &machine_id, force)?;
    Ok(path)
}

fn load_sync_config_from_path(path: &Path, fallback_hostname: &str) -> ConfigLoadResult {
    if !path.exists() {
        return ConfigLoadResult {
            config: SyncConfig::Disabled,
            warning: None,
        };
    }

    #[cfg(unix)]
    if let Some(warning) = permission_warning(path) {
        return ConfigLoadResult {
            config: SyncConfig::Disabled,
            warning: Some(warning),
        };
    }

    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => return invalid(path, format!("cannot read file: {err}")),
    };

    let parsed: SecretsFile = match serde_yaml::from_str(&content) {
        Ok(parsed) => parsed,
        Err(err) => return invalid(path, format!("invalid YAML: {err}")),
    };

    let Some(raw) = parsed.sync else {
        return invalid(path, "missing sync section");
    };

    if raw.enabled == Some(false) {
        return ConfigLoadResult {
            config: SyncConfig::Disabled,
            warning: None,
        };
    }

    let Some(server_url) = raw.server_url else {
        return invalid(path, "server_url is required");
    };
    if !server_url.starts_with("https://") {
        return invalid(path, "server_url must start with https://");
    }

    let Some(token) = raw.token else {
        return invalid(path, "token is required");
    };
    if token.chars().count() < 32 {
        return invalid(path, "token must be at least 32 characters");
    }

    let machine_id = raw
        .machine_id
        .unwrap_or_else(|| sanitize_hostname(fallback_hostname));
    if !vibe_usage_proto::is_valid_host_id(&machine_id) {
        return invalid(path, "machine_id must match [a-z0-9_-]{1,64}");
    }

    let request_timeout_seconds = raw
        .request_timeout_seconds
        .unwrap_or(DEFAULT_TIMEOUT_SECONDS);
    if request_timeout_seconds == 0 {
        return invalid(path, "request_timeout_seconds must be greater than 0");
    }

    ConfigLoadResult {
        config: SyncConfig::Enabled(EnabledSyncConfig {
            server_url,
            token,
            machine_id,
            upload_project_hash: raw.upload_project_hash.unwrap_or(true),
            request_timeout_seconds,
        }),
        warning: None,
    }
}

fn write_sync_config_template(path: &Path, machine_id: &str, force: bool) -> io::Result<()> {
    if path.exists() && !force {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "sync config already exists",
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true);
    if force {
        options.truncate(true);
    } else {
        options.create_new(true);
    }
    let mut file = options.open(path)?;
    file.write_all(sync_config_template(machine_id).as_bytes())?;
    file.sync_all()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn sync_config_template(machine_id: &str) -> String {
    format!(
        r#"sync:
  enabled: false
  server_url: "https://replace-with-your-sync-host.example"
  token: "replace-me"
  machine_id: "{machine_id}"
  upload_project_hash: true
  request_timeout_seconds: 15
"#
    )
}

fn invalid(path: &Path, reason: impl AsRef<str>) -> ConfigLoadResult {
    ConfigLoadResult {
        config: SyncConfig::Disabled,
        warning: Some(format!(
            "vibe-usage: invalid sync config at {}: {}",
            path.display(),
            reason.as_ref()
        )),
    }
}

fn resolve_secrets_path(home: &Path, env_override: Option<&Path>) -> PathBuf {
    env_override
        .map(Path::to_path_buf)
        .unwrap_or_else(|| home.join(".secrets").join("ai-usage.yaml"))
}

fn default_secrets_path() -> PathBuf {
    if let Ok(path) = std::env::var("VIBE_USAGE_SECRETS") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    resolve_secrets_path(&home, None)
}

fn default_hostname() -> String {
    if let Ok(hostname) = std::env::var("HOSTNAME") {
        if !hostname.trim().is_empty() {
            return hostname;
        }
    }
    fs::read_to_string("/etc/hostname")
        .ok()
        .map(|hostname| hostname.trim().to_string())
        .filter(|hostname| !hostname.is_empty())
        .unwrap_or_else(|| "machine".to_string())
}

fn sanitize_hostname(raw: &str) -> String {
    let mut result = String::new();
    let mut last_was_separator = false;
    for ch in raw.chars() {
        let next = if ch.is_ascii_alphanumeric() {
            Some(ch.to_ascii_lowercase())
        } else if ch == '-' || ch == '_' {
            Some(ch)
        } else {
            Some('-')
        };
        let Some(ch) = next else {
            continue;
        };
        if ch == '-' {
            if result.is_empty() || last_was_separator {
                continue;
            }
            last_was_separator = true;
        } else {
            last_was_separator = false;
        }
        result.push(ch);
        if result.len() == 64 {
            break;
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    if result.is_empty() {
        "machine".to_string()
    } else {
        result
    }
}

#[cfg(unix)]
fn permission_warning(path: &Path) -> Option<String> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = fs::metadata(path).ok()?;
    let mode = metadata.permissions().mode() & 0o777;
    if mode & 0o077 == 0 {
        None
    } else {
        Some(format!(
            "vibe-usage: refusing to read {}: permissions must be 0600",
            path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vibe-usage-config-test-{name}-{stamp}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_config(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, content).expect("write config");
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).expect("set permissions");
    }

    fn valid_yaml() -> String {
        format!(
            "sync:\n  server_url: \"https://usage.example.com\"\n  token: \"{}\"\n  machine_id: \"workstation-home\"\n  enabled: true\n  upload_project_hash: true\n  request_timeout_seconds: 15\n",
            "x".repeat(32)
        )
    }

    #[test]
    fn missing_config_disables_sync_without_warning() {
        let path = unique_temp_dir("missing").join("ai-usage.yaml");

        let result = load_sync_config_from_path(&path, "fallback-host");

        assert_eq!(result.config, SyncConfig::Disabled);
        assert_eq!(result.warning, None);
    }

    #[test]
    fn enabled_false_disables_sync_without_warning() {
        let dir = unique_temp_dir("disabled");
        let path = dir.join("ai-usage.yaml");
        write_config(
            &path,
            &format!(
                "sync:\n  server_url: \"https://usage.example.com\"\n  token: \"{}\"\n  enabled: false\n",
                "x".repeat(32)
            ),
        );

        let result = load_sync_config_from_path(&path, "fallback-host");

        assert_eq!(result.config, SyncConfig::Disabled);
        assert_eq!(result.warning, None);
    }

    #[test]
    fn valid_config_loads_enabled_settings() {
        let dir = unique_temp_dir("valid");
        let path = dir.join("ai-usage.yaml");
        write_config(&path, &valid_yaml());

        let result = load_sync_config_from_path(&path, "fallback-host");

        assert_eq!(result.warning, None);
        assert_eq!(
            result.config,
            SyncConfig::Enabled(EnabledSyncConfig {
                server_url: "https://usage.example.com".to_string(),
                token: "x".repeat(32),
                machine_id: "workstation-home".to_string(),
                upload_project_hash: true,
                request_timeout_seconds: 15,
            })
        );
    }

    #[test]
    fn malformed_yaml_disables_sync_with_warning() {
        let dir = unique_temp_dir("malformed");
        let path = dir.join("ai-usage.yaml");
        write_config(&path, "sync: [");

        let result = load_sync_config_from_path(&path, "fallback-host");

        assert_eq!(result.config, SyncConfig::Disabled);
        assert!(result.warning.expect("warning").contains("invalid YAML"));
    }

    #[cfg(unix)]
    #[test]
    fn group_or_world_readable_file_is_refused() {
        let dir = unique_temp_dir("permissions");
        let path = dir.join("ai-usage.yaml");
        write_config(&path, &valid_yaml());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).expect("set permissions");

        let result = load_sync_config_from_path(&path, "fallback-host");

        assert_eq!(result.config, SyncConfig::Disabled);
        assert_eq!(
            result.warning,
            Some(format!(
                "vibe-usage: refusing to read {}: permissions must be 0600",
                path.display()
            ))
        );
    }

    #[test]
    fn invalid_fields_disable_sync_with_warning() {
        for (name, yaml, expected) in [
            (
                "http-url",
                format!(
                    "sync:\n  server_url: \"http://usage.example.com\"\n  token: \"{}\"\n  machine_id: \"workstation-home\"\n",
                    "x".repeat(32)
                ),
                "server_url must start with https://",
            ),
            (
                "short-token",
                "sync:\n  server_url: \"https://usage.example.com\"\n  token: \"short\"\n  machine_id: \"workstation-home\"\n".to_string(),
                "token must be at least 32 characters",
            ),
            (
                "bad-machine-id",
                format!(
                    "sync:\n  server_url: \"https://usage.example.com\"\n  token: \"{}\"\n  machine_id: \"Workstation\"\n",
                    "x".repeat(32)
                ),
                "machine_id must match",
            ),
        ] {
            let dir = unique_temp_dir(name);
            let path = dir.join("ai-usage.yaml");
            write_config(&path, &yaml);

            let result = load_sync_config_from_path(&path, "fallback-host");

            assert_eq!(result.config, SyncConfig::Disabled, "{name}");
            assert!(
                result.warning.expect("warning").contains(expected),
                "{name}"
            );
        }
    }

    #[test]
    fn missing_machine_id_uses_sanitized_hostname() {
        let dir = unique_temp_dir("hostname");
        let path = dir.join("ai-usage.yaml");
        write_config(
            &path,
            &format!(
                "sync:\n  server_url: \"https://usage.example.com\"\n  token: \"{}\"\n",
                "x".repeat(32)
            ),
        );

        let result = load_sync_config_from_path(&path, "Work Station.local");

        let SyncConfig::Enabled(config) = result.config else {
            panic!("expected enabled config");
        };
        assert_eq!(config.machine_id, "work-station-local");
    }

    #[test]
    fn secrets_path_prefers_env_override() {
        let home = Path::new("/tmp/home-for-test");
        let override_path = Path::new("/tmp/override.yaml");

        assert_eq!(
            resolve_secrets_path(home, Some(override_path)),
            override_path.to_path_buf()
        );
        assert_eq!(
            resolve_secrets_path(home, None),
            home.join(".secrets").join("ai-usage.yaml")
        );
    }

    #[test]
    fn init_template_creates_disabled_config() {
        let dir = unique_temp_dir("init-template");
        let path = dir.join(".secrets").join("ai-usage.yaml");

        write_sync_config_template(&path, "workstation-home", false).expect("write template");

        let content = fs::read_to_string(&path).expect("read template");
        assert!(content.contains("enabled: false"));
        assert!(content.contains("machine_id: \"workstation-home\""));
        let result = load_sync_config_from_path(&path, "fallback-host");
        assert_eq!(result.config, SyncConfig::Disabled);
        assert_eq!(result.warning, None);
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn init_template_refuses_to_overwrite_without_force() {
        let dir = unique_temp_dir("init-existing");
        let path = dir.join("ai-usage.yaml");
        write_sync_config_template(&path, "first-host", false).expect("write initial template");

        let err = write_sync_config_template(&path, "second-host", false).expect_err("exists");

        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        let content = fs::read_to_string(&path).expect("read template");
        assert!(content.contains("machine_id: \"first-host\""));
    }

    #[test]
    fn init_template_force_replaces_existing_file() {
        let dir = unique_temp_dir("init-force");
        let path = dir.join("ai-usage.yaml");
        write_sync_config_template(&path, "first-host", false).expect("write initial template");

        write_sync_config_template(&path, "second-host", true).expect("replace template");

        let content = fs::read_to_string(&path).expect("read template");
        assert!(content.contains("machine_id: \"second-host\""));
    }
}
