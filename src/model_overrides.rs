//! User-editable model overrides.
//!
//! Reads `~/.config/ai-usage/models.toml` (XDG-aware) so a user can fix a
//! display name or pin a price for any model id without recompiling. This is
//! the escape hatch for private vendors and the day-one window before LiteLLM
//! lists a freshly released model. Overrides win over both the algorithmic
//! label and the live LiteLLM pricing.
//!
//! ```toml
//! # Give any model a display name (covers private vendors / the day-one gap).
//! [display."anthropic/claude-opus-4-8"]
//! short = "Opus 4.8"
//!
//! # Pin pricing in $/MTok; cache_read/cache_write default to the input rate.
//! [pricing."some-private-model"]
//! input = 3.0
//! output = 15.0
//! cache_read = 0.30
//! cache_write = 3.75
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::constants::ModelPricing;

/// Parsed, ready-to-apply overrides.
#[derive(Debug, Default, Clone)]
pub struct ModelOverrides {
    /// model id -> short display label
    pub display: HashMap<String, String>,
    /// model id -> pricing (already in $/MTok)
    pub pricing: HashMap<String, ModelPricing>,
}

#[derive(Debug, Default, Deserialize)]
struct RawOverrides {
    #[serde(default)]
    display: HashMap<String, DisplayEntry>,
    #[serde(default)]
    pricing: HashMap<String, PricingEntry>,
}

#[derive(Debug, Deserialize)]
struct DisplayEntry {
    short: String,
}

#[derive(Debug, Deserialize)]
struct PricingEntry {
    input: f64,
    output: f64,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
    input_above_200k: Option<f64>,
    output_above_200k: Option<f64>,
    cache_read_above_200k: Option<f64>,
    cache_write_above_200k: Option<f64>,
}

impl PricingEntry {
    fn into_pricing(self) -> ModelPricing {
        ModelPricing {
            input: self.input,
            output: self.output,
            // Mirror the loader's convention: absent cache rates fall back to
            // the input rate rather than zero.
            cache_input: self.cache_read.unwrap_or(self.input),
            cache_output: self.cache_write.unwrap_or(self.input),
            input_above_200k: self.input_above_200k,
            output_above_200k: self.output_above_200k,
            cache_input_above_200k: self.cache_read_above_200k,
            cache_output_above_200k: self.cache_write_above_200k,
            _comment: Some("Source: user override (models.toml)".to_string()),
        }
    }
}

/// Parse overrides from a TOML string. Returns an error on malformed TOML so
/// the loader can decide whether to warn; in-process callers use [`load`].
pub fn parse_from_str(s: &str) -> Result<ModelOverrides, toml::de::Error> {
    let raw: RawOverrides = toml::from_str(s)?;
    Ok(ModelOverrides {
        display: raw.display.into_iter().map(|(k, v)| (k, v.short)).collect(),
        pricing: raw
            .pricing
            .into_iter()
            .map(|(k, v)| (k, v.into_pricing()))
            .collect(),
    })
}

/// `$XDG_CONFIG_HOME/ai-usage/models.toml`, falling back to
/// `~/.config/ai-usage/models.toml`.
fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("ai-usage").join("models.toml"))
}

/// Load and cache overrides for the process. Any failure (missing file, parse
/// error) degrades silently to empty overrides, matching the pricing loader's
/// best-effort layering.
pub fn load() -> &'static ModelOverrides {
    static CACHE: OnceLock<ModelOverrides> = OnceLock::new();
    CACHE.get_or_init(|| {
        let Some(path) = config_path() else {
            return ModelOverrides::default();
        };
        let Ok(content) = std::fs::read_to_string(&path) else {
            return ModelOverrides::default();
        };
        match parse_from_str(&content) {
            Ok(overrides) => overrides,
            Err(e) => {
                eprintln!("warning: ignoring {}: {}", path.display(), e);
                ModelOverrides::default()
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_display_and_pricing_sections() {
        let toml = r#"
[display."anthropic/claude-opus-4-8"]
short = "Opus 4.8"

[display."weird-private-model"]
short = "Weird"

[pricing."some-private-model"]
input = 3.0
output = 15.0
cache_read = 0.30
cache_write = 3.75
input_above_200k = 6.0
"#;
        let ov = parse_from_str(toml).expect("valid toml");

        assert_eq!(
            ov.display
                .get("anthropic/claude-opus-4-8")
                .map(String::as_str),
            Some("Opus 4.8")
        );
        assert_eq!(
            ov.display.get("weird-private-model").map(String::as_str),
            Some("Weird")
        );

        let p = ov
            .pricing
            .get("some-private-model")
            .expect("pricing present");
        assert!((p.input - 3.0).abs() < 1e-9);
        assert!((p.output - 15.0).abs() < 1e-9);
        assert!((p.cache_input - 0.30).abs() < 1e-9);
        assert!((p.cache_output - 3.75).abs() < 1e-9);
        assert_eq!(p.input_above_200k, Some(6.0));
    }

    #[test]
    fn cache_rates_default_to_input_when_absent() {
        let toml = r#"
[pricing."bare-model"]
input = 2.5
output = 10.0
"#;
        let ov = parse_from_str(toml).expect("valid toml");
        let p = ov.pricing.get("bare-model").expect("present");
        assert!((p.cache_input - 2.5).abs() < 1e-9);
        assert!((p.cache_output - 2.5).abs() < 1e-9);
    }

    #[test]
    fn empty_input_yields_empty_overrides() {
        let ov = parse_from_str("").expect("empty is valid");
        assert!(ov.display.is_empty());
        assert!(ov.pricing.is_empty());
    }

    #[test]
    fn malformed_toml_is_an_error() {
        assert!(parse_from_str("this is = = not toml").is_err());
    }
}
