//! In-app self-update: query GitHub Releases for the latest tag, download the
//! binary matching the build target, atomically replace the running executable,
//! and re-exec it with the original argv.

use std::ffi::{CString, OsString};
use std::fs;
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

const REPO: &str = "SihaoLiu/ai-usage";
const USER_AGENT: &str = concat!("vibe-usage/", env!("CARGO_PKG_VERSION"));
const HTTP_TIMEOUT: Duration = Duration::from_secs(60);

pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build-time target triple, e.g. `x86_64-unknown-linux-musl`.
pub const BUILD_TARGET: &str = env!("VIBE_USAGE_BUILD_TARGET");

#[derive(Debug)]
pub struct LatestRelease {
    pub tag: String,
    pub version: String,
    pub asset_name: String,
    pub asset_url: String,
}

#[derive(Debug)]
pub enum UpdateOutcome {
    AlreadyLatest { current: String, latest: String },
}

/// Asset name for this build, derived from the cargo target triple by
/// stripping the redundant `unknown-` vendor segment.
///
/// Examples:
/// - `x86_64-unknown-linux-musl`  -> `vibe-usage-x86_64-linux-musl`
/// - `aarch64-unknown-linux-musl` -> `vibe-usage-aarch64-linux-musl`
/// - `x86_64-apple-darwin`        -> `vibe-usage-x86_64-apple-darwin`
/// - `aarch64-apple-darwin`       -> `vibe-usage-aarch64-apple-darwin`
pub fn expected_asset_name(target: &str) -> String {
    let cleaned: String = target
        .split('-')
        .filter(|seg| *seg != "unknown")
        .collect::<Vec<_>>()
        .join("-");
    format!("vibe-usage-{cleaned}")
}

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(HTTP_TIMEOUT))
        .build()
        .new_agent()
}

/// Hit the GitHub API for the latest release and find the asset matching
/// this build target.
pub fn fetch_latest_release() -> Result<LatestRelease, String> {
    let asset_name = expected_asset_name(BUILD_TARGET);
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let agent = http_agent();

    let body: String = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("GitHub API request failed: {e}"))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("failed to read GitHub API response: {e}"))?;

    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("failed to parse GitHub API response: {e}"))?;

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
        .find(|a| {
            a.get("name").and_then(|n| n.as_str()) == Some(asset_name.as_str())
        })
        .ok_or_else(|| {
            format!(
                "No asset named '{asset_name}' in release {tag}. \
                 Available assets: {}",
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

fn is_newer(latest: &str, current: &str) -> bool {
    fn parts(s: &str) -> Vec<u64> {
        s.split('.')
            .map(|p| p.chars().take_while(|c| c.is_ascii_digit()).collect::<String>())
            .map(|p| p.parse::<u64>().unwrap_or(0))
            .collect()
    }
    let l = parts(latest);
    let c = parts(current);
    let n = l.len().max(c.len());
    for i in 0..n {
        let a = l.get(i).copied().unwrap_or(0);
        let b = c.get(i).copied().unwrap_or(0);
        if a != b {
            return a > b;
        }
    }
    false
}

/// Download `url` into a fresh temp file in `dir` and return its path.
fn download_to(dir: &Path, url: &str) -> Result<PathBuf, String> {
    let pid = std::process::id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let temp_path = dir.join(format!(".vibe-usage.download.{pid}.{now}"));

    let agent = http_agent();
    let bytes: Vec<u8> = agent
        .get(url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/octet-stream")
        .call()
        .map_err(|e| format!("download failed: {e}"))?
        .body_mut()
        .with_config()
        .limit(512 * 1024 * 1024)
        .read_to_vec()
        .map_err(|e| format!("download read error: {e}"))?;

    if bytes.is_empty() {
        return Err("downloaded file is empty".to_string());
    }

    let mut file = fs::File::create(&temp_path)
        .map_err(|e| format!("cannot create temp file {}: {e}", temp_path.display()))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write to temp file failed: {e}"))?;
    file.sync_all().ok();
    drop(file);
    Ok(temp_path)
}

/// chmod 755 + atomic rename over the current executable path.
fn install_over(current_exe: &Path, downloaded: &Path) -> Result<(), String> {
    fs::set_permissions(downloaded, fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("chmod failed: {e}"))?;
    fs::rename(downloaded, current_exe).map_err(|e| {
        let _ = fs::remove_file(downloaded);
        format!(
            "failed to replace {}: {e}. \
             You may need write permission on the install directory.",
            current_exe.display()
        )
    })
}

/// Replace the current process image with a fresh exec of `exe_path` using
/// `args`. Never returns on success.
fn exec_self(exe_path: &Path, args: &[OsString]) -> Result<std::convert::Infallible, String> {
    let c_exe = CString::new(exe_path.as_os_str().as_bytes())
        .map_err(|e| format!("path contains NUL byte: {e}"))?;
    let c_args: Vec<CString> = args
        .iter()
        .map(|a| {
            CString::new(a.as_bytes())
                .map_err(|e| format!("argv contains NUL byte: {e}"))
        })
        .collect::<Result<_, _>>()?;

    let mut argv_ptrs: Vec<*const libc::c_char> =
        c_args.iter().map(|c| c.as_ptr()).collect();
    argv_ptrs.push(std::ptr::null());

    // execv replaces this process; on success it does not return.
    unsafe {
        libc::execv(c_exe.as_ptr(), argv_ptrs.as_ptr());
    }
    Err(format!("execv failed: {}", io::Error::last_os_error()))
}

/// Drive the full update flow.
///
/// `log` is invoked for user-visible progress messages so callers in monitor
/// mode can flush them in raw-mode-safe form. Returns `Ok(AlreadyLatest)` when
/// no update was needed; on a successful update the process is re-exec'd and
/// this function never returns.
pub fn run_update<F: FnMut(&str)>(mut log: F) -> Result<UpdateOutcome, String> {
    log(&format!(
        "Checking for updates (current: v{CURRENT_VERSION}, target: {BUILD_TARGET})..."
    ));
    let release = fetch_latest_release()?;
    log(&format!(
        "Latest release: {} ({})",
        release.tag, release.asset_name
    ));

    if !is_newer(&release.version, CURRENT_VERSION) {
        return Ok(UpdateOutcome::AlreadyLatest {
            current: CURRENT_VERSION.to_string(),
            latest: release.version,
        });
    }

    let current_exe = std::env::current_exe()
        .map_err(|e| format!("cannot resolve current executable: {e}"))?;
    let install_dir = current_exe
        .parent()
        .ok_or_else(|| "current executable has no parent directory".to_string())?
        .to_path_buf();

    log(&format!("Downloading {} ...", release.asset_url));
    let downloaded = download_to(&install_dir, &release.asset_url)?;

    log(&format!("Installing to {} ...", current_exe.display()));
    install_over(&current_exe, &downloaded)?;

    let args: Vec<OsString> = std::env::args_os().collect();
    log(&format!(
        "Update applied. Restarting v{} ...",
        release.version
    ));
    exec_self(&current_exe, &args)?;
    unreachable!("exec_self returned without error");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_name_strips_unknown_vendor() {
        assert_eq!(
            expected_asset_name("x86_64-unknown-linux-musl"),
            "vibe-usage-x86_64-linux-musl"
        );
        assert_eq!(
            expected_asset_name("aarch64-unknown-linux-musl"),
            "vibe-usage-aarch64-linux-musl"
        );
    }

    #[test]
    fn asset_name_for_darwin_keeps_apple_vendor() {
        assert_eq!(
            expected_asset_name("x86_64-apple-darwin"),
            "vibe-usage-x86_64-apple-darwin"
        );
        assert_eq!(
            expected_asset_name("aarch64-apple-darwin"),
            "vibe-usage-aarch64-apple-darwin"
        );
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
    fn semver_compare_tolerates_v_prefix_stripping() {
        // Caller strips the leading 'v'; ensure plain numerics still work.
        assert!(is_newer("1.5.8", "1.5.7"));
    }
}
