//! Shared GitHub release updater for workspace binaries.

use std::ffi::{CString, OsString};
use std::fs;
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const DEFAULT_REPO: &str = "SihaoLiu/ai-usage";
pub const BUILD_TARGET: &str = env!("AI_USAGE_BUILD_TARGET");
pub const DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS: u64 = 3600;
pub const MIN_AUTO_UPDATE_INTERVAL_SECONDS: u64 = 60;

const HTTP_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateConfig {
    pub binary_name: String,
    pub current_version: String,
    pub build_target: String,
    pub repo: String,
}

impl UpdateConfig {
    pub fn current(binary_name: &str, current_version: &str) -> Self {
        Self {
            binary_name: binary_name.to_string(),
            current_version: current_version.to_string(),
            build_target: BUILD_TARGET.to_string(),
            repo: DEFAULT_REPO.to_string(),
        }
    }

    fn user_agent(&self) -> String {
        format!("{}/{}", self.binary_name, self.current_version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatestRelease {
    pub tag: String,
    pub version: String,
    pub asset_name: String,
    pub asset_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledUpdate {
    pub current: String,
    pub latest: String,
    pub tag: String,
    pub asset_name: String,
    pub executable: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallOutcome {
    AlreadyLatest { current: String, latest: String },
    Updated(InstalledUpdate),
}

pub fn expected_asset_name(binary_name: &str, target: &str) -> String {
    let cleaned: String = target
        .split('-')
        .filter(|seg| *seg != "unknown")
        .collect::<Vec<_>>()
        .join("-");
    format!("{binary_name}-{cleaned}")
}

pub fn normalize_auto_update_interval(seconds: u64) -> Duration {
    Duration::from_secs(seconds.max(MIN_AUTO_UPDATE_INTERVAL_SECONDS))
}

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(HTTP_TIMEOUT))
        .build()
        .new_agent()
}

pub fn fetch_latest_release(config: &UpdateConfig) -> Result<LatestRelease, String> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        config.repo
    );
    let agent = http_agent();

    let body: String = agent
        .get(&url)
        .header("User-Agent", config.user_agent())
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|err| format!("GitHub API request failed: {err}"))?
        .body_mut()
        .read_to_string()
        .map_err(|err| format!("failed to read GitHub API response: {err}"))?;

    parse_latest_release(config, &body)
}

pub fn parse_latest_release(config: &UpdateConfig, body: &str) -> Result<LatestRelease, String> {
    let asset_name = expected_asset_name(&config.binary_name, &config.build_target);
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|err| format!("failed to parse GitHub API response: {err}"))?;

    let tag = value
        .get("tag_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "GitHub release missing tag_name".to_string())?
        .to_string();
    let version = tag.strip_prefix('v').unwrap_or(&tag).to_string();

    let assets = value
        .get("assets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "GitHub release missing assets".to_string())?;

    let asset = assets
        .iter()
        .find(|a| a.get("name").and_then(|n| n.as_str()) == Some(asset_name.as_str()))
        .ok_or_else(|| {
            format!(
                "No asset named '{asset_name}' in release {tag}. Available assets: {}",
                assets
                    .iter()
                    .filter_map(|a| a.get("name").and_then(|n| n.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    let asset_url = asset
        .get("browser_download_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "asset missing browser_download_url".to_string())?
        .to_string();

    Ok(LatestRelease {
        tag,
        version,
        asset_name,
        asset_url,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StableVersion {
    major: u64,
    minor: u64,
    patch: u64,
}

fn parse_stable_version(value: &str) -> Option<StableVersion> {
    let value = value.strip_prefix('v').unwrap_or(value);
    let mut parts = value.split('.');
    let major = parts.next()?.parse::<u64>().ok()?;
    let minor = parts.next()?.parse::<u64>().ok()?;
    let patch = parts.next()?.parse::<u64>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(StableVersion {
        major,
        minor,
        patch,
    })
}

pub fn is_newer(latest: &str, current: &str) -> bool {
    let Some(latest) = parse_stable_version(latest) else {
        return false;
    };
    let Some(current) = parse_stable_version(current) else {
        return false;
    };
    latest > current
}

fn download_to(config: &UpdateConfig, dir: &Path, url: &str) -> Result<PathBuf, String> {
    let pid = std::process::id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let temp_path = dir.join(format!(".{}.download.{pid}.{now}", config.binary_name));

    let agent = http_agent();
    let bytes: Vec<u8> = agent
        .get(url)
        .header("User-Agent", config.user_agent())
        .header("Accept", "application/octet-stream")
        .call()
        .map_err(|err| format!("download failed: {err}"))?
        .body_mut()
        .with_config()
        .limit(512 * 1024 * 1024)
        .read_to_vec()
        .map_err(|err| format!("download read error: {err}"))?;

    if bytes.is_empty() {
        return Err("downloaded file is empty".to_string());
    }

    let mut file = fs::File::create(&temp_path)
        .map_err(|err| format!("cannot create temp file {}: {err}", temp_path.display()))?;
    file.write_all(&bytes)
        .map_err(|err| format!("write to temp file failed: {err}"))?;
    file.sync_all().ok();
    drop(file);
    Ok(temp_path)
}

fn install_over(current_exe: &Path, downloaded: &Path) -> Result<(), String> {
    fs::set_permissions(downloaded, fs::Permissions::from_mode(0o755))
        .map_err(|err| format!("chmod failed: {err}"))?;
    fs::rename(downloaded, current_exe).map_err(|err| {
        let _ = fs::remove_file(downloaded);
        format!(
            "failed to replace {}: {err}. You may need write permission on the install directory.",
            current_exe.display()
        )
    })
}

pub fn check_and_install<F: FnMut(&str)>(
    config: &UpdateConfig,
    mut log: F,
) -> Result<InstallOutcome, String> {
    log(&format!(
        "Checking for updates (current: v{}, target: {})...",
        config.current_version, config.build_target
    ));
    let release = fetch_latest_release(config)?;
    log(&format!(
        "Latest release: {} ({})",
        release.tag, release.asset_name
    ));

    if !is_newer(&release.version, &config.current_version) {
        return Ok(InstallOutcome::AlreadyLatest {
            current: config.current_version.clone(),
            latest: release.version,
        });
    }

    let current_exe = std::env::current_exe()
        .map_err(|err| format!("cannot resolve current executable: {err}"))?;
    let install_dir = current_exe
        .parent()
        .ok_or_else(|| "current executable has no parent directory".to_string())?
        .to_path_buf();

    log(&format!("Downloading {} ...", release.asset_url));
    let downloaded = download_to(config, &install_dir, &release.asset_url)?;

    log(&format!("Installing to {} ...", current_exe.display()));
    install_over(&current_exe, &downloaded)?;

    Ok(InstallOutcome::Updated(InstalledUpdate {
        current: config.current_version.clone(),
        latest: release.version,
        tag: release.tag,
        asset_name: release.asset_name,
        executable: current_exe,
    }))
}

pub fn exec_path(exe_path: &Path, args: &[OsString]) -> Result<std::convert::Infallible, String> {
    let c_exe = CString::new(exe_path.as_os_str().as_bytes())
        .map_err(|err| format!("path contains NUL byte: {err}"))?;
    let c_args: Vec<CString> = args
        .iter()
        .map(|arg| {
            CString::new(arg.as_bytes()).map_err(|err| format!("argv contains NUL byte: {err}"))
        })
        .collect::<Result<_, _>>()?;

    let mut argv_ptrs: Vec<*const libc::c_char> = c_args.iter().map(|arg| arg.as_ptr()).collect();
    argv_ptrs.push(std::ptr::null());

    unsafe {
        libc::execv(c_exe.as_ptr(), argv_ptrs.as_ptr());
    }
    Err(format!("execv failed: {}", io::Error::last_os_error()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(binary_name: &str) -> UpdateConfig {
        UpdateConfig {
            binary_name: binary_name.to_string(),
            current_version: "2.2.2".to_string(),
            build_target: "x86_64-unknown-linux-gnu".to_string(),
            repo: DEFAULT_REPO.to_string(),
        }
    }

    #[test]
    fn asset_name_strips_unknown_vendor() {
        assert_eq!(
            expected_asset_name("ai-usage", "x86_64-unknown-linux-musl"),
            "ai-usage-x86_64-linux-musl"
        );
        assert_eq!(
            expected_asset_name("ai-usage", "aarch64-unknown-linux-musl"),
            "ai-usage-aarch64-linux-musl"
        );
    }

    #[test]
    fn asset_name_keeps_apple_vendor() {
        assert_eq!(
            expected_asset_name("ai-usage", "aarch64-apple-darwin"),
            "ai-usage-aarch64-apple-darwin"
        );
    }

    #[test]
    fn asset_name_supports_server_binary() {
        assert_eq!(
            expected_asset_name("ai-usage-server", "x86_64-unknown-linux-gnu"),
            "ai-usage-server-x86_64-linux-gnu"
        );
    }

    #[test]
    fn parses_matching_latest_release_asset() {
        let body = r#"{
          "tag_name": "v2.3.0",
          "assets": [
            {
              "name": "ai-usage-x86_64-linux-gnu",
              "browser_download_url": "https://example.com/ai-usage"
            }
          ]
        }"#;

        let release = parse_latest_release(&config("ai-usage"), body).expect("release parses");

        assert_eq!(release.tag, "v2.3.0");
        assert_eq!(release.version, "2.3.0");
        assert_eq!(release.asset_name, "ai-usage-x86_64-linux-gnu");
        assert_eq!(release.asset_url, "https://example.com/ai-usage");
    }

    #[test]
    fn missing_asset_error_lists_available_assets() {
        let body = r#"{
          "tag_name": "v2.3.0",
          "assets": [
            {
              "name": "ai-usage-x86_64-linux-gnu",
              "browser_download_url": "https://example.com/ai-usage"
            }
          ]
        }"#;

        let err = parse_latest_release(&config("ai-usage-server"), body)
            .expect_err("server asset should be missing");

        assert!(err.contains("ai-usage-server-x86_64-linux-gnu"));
        assert!(err.contains("ai-usage-x86_64-linux-gnu"));
    }

    #[test]
    fn semver_compare_handles_basic_cases() {
        assert!(is_newer("1.5.9", "1.5.8"));
        assert!(is_newer("1.6.0", "1.5.99"));
        assert!(is_newer("2.0.0", "1.99.99"));
        assert!(!is_newer("1.5.8", "1.5.8"));
        assert!(!is_newer("1.5.7", "1.5.8"));
    }

    #[test]
    fn semver_compare_rejects_prerelease_and_malformed_versions() {
        assert!(!is_newer("2.3.0-rc.1", "2.2.2"));
        assert!(!is_newer("2.3", "2.2.2"));
        assert!(!is_newer("release-2.3.0", "2.2.2"));
    }

    #[test]
    fn auto_update_interval_has_minimum() {
        assert_eq!(
            normalize_auto_update_interval(0),
            Duration::from_secs(MIN_AUTO_UPDATE_INTERVAL_SECONDS)
        );
        assert_eq!(
            normalize_auto_update_interval(7200),
            Duration::from_secs(7200)
        );
    }
}
