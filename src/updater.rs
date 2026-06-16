//! Client self-update wrapper around the shared updater crate.

use std::ffi::OsString;

pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug)]
pub enum UpdateOutcome {
    AlreadyLatest { current: String, latest: String },
}

#[cfg(test)]
pub fn expected_asset_name(target: &str) -> String {
    expected_asset_name_for_binary("ai-usage", target)
}

#[cfg(test)]
pub fn expected_asset_name_for_binary(binary_name: &str, target: &str) -> String {
    ai_usage_updater::expected_asset_name(binary_name, target)
}

fn update_config() -> ai_usage_updater::UpdateConfig {
    ai_usage_updater::UpdateConfig::current("ai-usage", CURRENT_VERSION)
}

pub fn run_update<F: FnMut(&str)>(mut log: F) -> Result<UpdateOutcome, String> {
    let config = update_config();
    match ai_usage_updater::check_and_install(&config, |message| log(message))? {
        ai_usage_updater::InstallOutcome::AlreadyLatest { current, latest } => {
            Ok(UpdateOutcome::AlreadyLatest { current, latest })
        }
        ai_usage_updater::InstallOutcome::Updated(update) => {
            log(&format!(
                "Update applied. Restarting v{} ...",
                update.latest
            ));
            let args: Vec<OsString> = std::env::args_os().collect();
            ai_usage_updater::exec_path(&update.executable, &args)?;
            unreachable!("exec_path returned without error");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_name_strips_unknown_vendor() {
        assert_eq!(
            expected_asset_name("x86_64-unknown-linux-musl"),
            "ai-usage-x86_64-linux-musl"
        );
        assert_eq!(
            expected_asset_name("aarch64-unknown-linux-musl"),
            "ai-usage-aarch64-linux-musl"
        );
    }

    #[test]
    fn asset_name_for_darwin_keeps_apple_vendor() {
        assert_eq!(
            expected_asset_name("x86_64-apple-darwin"),
            "ai-usage-x86_64-apple-darwin"
        );
        assert_eq!(
            expected_asset_name("aarch64-apple-darwin"),
            "ai-usage-aarch64-apple-darwin"
        );
    }

    #[test]
    fn asset_name_can_target_server_binary() {
        assert_eq!(
            expected_asset_name_for_binary("ai-usage-server", "x86_64-unknown-linux-gnu"),
            "ai-usage-server-x86_64-linux-gnu"
        );
    }

    #[test]
    fn semver_compare_handles_basic_cases() {
        assert!(ai_usage_updater::is_newer("1.5.9", "1.5.8"));
        assert!(ai_usage_updater::is_newer("1.6.0", "1.5.99"));
        assert!(ai_usage_updater::is_newer("2.0.0", "1.99.99"));
        assert!(!ai_usage_updater::is_newer("1.5.8", "1.5.8"));
        assert!(!ai_usage_updater::is_newer("1.5.7", "1.5.8"));
    }

    #[test]
    fn semver_compare_tolerates_v_prefix_stripping() {
        assert!(ai_usage_updater::is_newer("1.5.8", "1.5.7"));
    }
}
