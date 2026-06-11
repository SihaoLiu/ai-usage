use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::PathBuf;

use crate::model_id::{Provider, parse_model_identity};

/// Embedded pricing data from pricing.json
const PRICING_JSON: &str = include_str!("../pricing.json");

/// Token threshold for Claude 1M-context tiered pricing.
/// Mirrors ccusage's `DEFAULT_TIERED_THRESHOLD` and Anthropic's published 200k tier.
pub const TIER_THRESHOLD: i64 = 200_000;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelPricing {
    pub input: f64,
    pub output: f64,
    pub cache_input: f64,
    pub cache_output: f64,
    /// Optional per-MTok rate for input tokens above the 200k threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_above_200k: Option<f64>,
    /// Optional per-MTok rate for output tokens above the 200k threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_above_200k: Option<f64>,
    /// Optional per-MTok rate for cache-read tokens above the 200k threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_input_above_200k: Option<f64>,
    /// Optional per-MTok rate for cache-creation tokens above the 200k threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_output_above_200k: Option<f64>,
    #[serde(rename = "_comment", default, skip_serializing_if = "Option::is_none")]
    pub _comment: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct FastTierPricing {
    #[serde(default)]
    default: Option<f64>,
    #[serde(default)]
    models: HashMap<String, f64>,
    #[serde(rename = "_comment", default, skip_serializing_if = "Option::is_none")]
    _comment: Option<String>,
}

impl ModelPricing {
    /// Apply tiered pricing for a single token bucket within one request.
    ///
    /// When `tokens > TIER_THRESHOLD` and a tiered rate is configured, the portion
    /// up to the threshold is charged at `base_per_mtok` and the excess at
    /// `above_per_mtok` (matches ccusage's `calculateTieredCost` semantics, which
    /// is the prorata interpretation of Anthropic's 1M-context premium tier).
    pub fn tier_cost(tokens: i64, base_per_mtok: f64, above_per_mtok: Option<f64>) -> f64 {
        if tokens <= 0 {
            return 0.0;
        }
        if let Some(above) = above_per_mtok
            && tokens > TIER_THRESHOLD
        {
            let below = TIER_THRESHOLD as f64;
            let extra = (tokens - TIER_THRESHOLD) as f64;
            return below * base_per_mtok / 1_000_000.0 + extra * above / 1_000_000.0;
        }
        tokens as f64 * base_per_mtok / 1_000_000.0
    }

    /// Scale every rate (base + tier overrides) by a per-record factor.
    pub fn scaled_by(&self, multiplier: f64) -> ModelPricing {
        ModelPricing {
            input: self.input * multiplier,
            output: self.output * multiplier,
            cache_input: self.cache_input * multiplier,
            cache_output: self.cache_output * multiplier,
            input_above_200k: self.input_above_200k.map(|v| v * multiplier),
            output_above_200k: self.output_above_200k.map(|v| v * multiplier),
            cache_input_above_200k: self.cache_input_above_200k.map(|v| v * multiplier),
            cache_output_above_200k: self.cache_output_above_200k.map(|v| v * multiplier),
            _comment: self._comment.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct VendorPricing {
    models: HashMap<String, ModelPricing>,
    default: ModelPricing,
}

#[derive(Debug, Deserialize)]
struct PricingData {
    #[serde(rename = "_meta")]
    #[serde(default)]
    _meta: Option<serde_json::Value>,
    claude: VendorPricing,
    codex: VendorPricing,
    gemini: VendorPricing,
    #[serde(default)]
    fast_tiers: HashMap<String, HashMap<String, FastTierPricing>>,
}

/// All pricing tables for the three vendors.
pub struct AllPricing {
    pub claude_models: HashMap<String, ModelPricing>,
    pub claude_default: ModelPricing,
    pub codex_models: HashMap<String, ModelPricing>,
    pub codex_default: ModelPricing,
    pub gemini_models: HashMap<String, ModelPricing>,
    pub gemini_default: ModelPricing,
    /// User-supplied per-model price overrides (from `models.toml`), keyed by
    /// exact model id and consulted before any table, regardless of vendor.
    overrides_pricing: HashMap<String, ModelPricing>,
    fast_tiers: HashMap<String, HashMap<i8, FastTierPricing>>,
}

impl AllPricing {
    /// Load the embedded `pricing.json` without applying date-alias expansion.
    /// Intended as the baseline layer for the layered loader in `crate::pricing`.
    pub fn load_raw() -> Self {
        let data: PricingData =
            serde_json::from_str(PRICING_JSON).expect("Failed to parse embedded pricing.json");
        AllPricing {
            claude_models: data.claude.models,
            claude_default: data.claude.default,
            codex_models: data.codex.models,
            codex_default: data.codex.default,
            gemini_models: data.gemini.models,
            gemini_default: data.gemini.default,
            overrides_pricing: HashMap::new(),
            fast_tiers: normalize_fast_tiers(data.fast_tiers),
        }
    }

    /// Install user price overrides (highest priority, vendor-agnostic).
    pub fn set_pricing_overrides(&mut self, overrides: HashMap<String, ModelPricing>) {
        self.overrides_pricing = overrides;
    }

    /// Apply `-YYYYMMDD` date-alias expansion to every vendor table. Must be
    /// called once after all overlay layers have been merged in.
    pub fn finalize(mut self) -> Self {
        self.claude_models = expand_date_aliases(self.claude_models);
        self.codex_models = expand_date_aliases(self.codex_models);
        self.gemini_models = expand_date_aliases(self.gemini_models);
        self
    }

    /// Overlay another set of vendor tables into this one. Base rates from the
    /// incoming layer win (so live LiteLLM data refreshes prices), but optional
    /// tier-pricing fields fall back to the existing entry when the incoming
    /// one omits them.
    pub fn overlay(
        &mut self,
        claude: HashMap<String, ModelPricing>,
        codex: HashMap<String, ModelPricing>,
        gemini: HashMap<String, ModelPricing>,
    ) {
        overlay_table(&mut self.claude_models, claude);
        overlay_table(&mut self.codex_models, codex);
        overlay_table(&mut self.gemini_models, gemini);
    }

    pub fn get_pricing(&self, vendor: &str, model: &str) -> &ModelPricing {
        // User overrides win regardless of vendor routing (covers private
        // vendors and the day-one gap before LiteLLM lists a new model).
        if let Some(p) = self.overrides_pricing.get(model) {
            return p;
        }

        let (table, default) = match vendor {
            "codex" => (&self.codex_models, &self.codex_default),
            "gemini" => (&self.gemini_models, &self.gemini_default),
            _ => (&self.claude_models, &self.claude_default),
        };

        if let Some(p) = table.get(model) {
            return p;
        }

        // Fuzzy fallback: strip a trailing -YYYYMMDD date suffix and retry.
        // Mirrors ccusage's prefix-tolerant matching so e.g. a future
        // claude-sonnet-4-5-20251201 still resolves to claude-sonnet-4-5
        // pricing rather than the vendor default.
        if let Some(stripped) = strip_date_suffix(model)
            && let Some(p) = table.get(stripped)
        {
            return p;
        }

        // Class-aware same-family fallback: borrow the newest known model that
        // shares provider + family + size/modifier class. Keeps a brand-new
        // claude-opus-4-8 priced like opus instead of the vendor default, while
        // never letting a `-mini`/`-nano` variant inherit the base model's rate.
        if let Some(p) = same_class_fallback(table, model) {
            return p;
        }

        default
    }

    pub fn pricing_for_entry<'a>(
        &'a self,
        vendor: &str,
        model: &str,
        fast_tier: i8,
    ) -> Cow<'a, ModelPricing> {
        let base = self.get_pricing(vendor, model);
        let factor = self.fast_tier_factor(vendor, model, fast_tier);
        if (factor - 1.0).abs() < f64::EPSILON {
            Cow::Borrowed(base)
        } else {
            Cow::Owned(base.scaled_by(factor))
        }
    }

    fn fast_tier_factor(&self, vendor: &str, model: &str, fast_tier: i8) -> f64 {
        if fast_tier <= 0 {
            return 1.0;
        }
        let Some(vendor_tiers) = self.fast_tiers.get(vendor) else {
            return 1.0;
        };
        let Some(tier) = vendor_tiers.get(&fast_tier) else {
            return 1.0;
        };
        if let Some(mult) = tier.models.get(model) {
            return *mult;
        }
        if let Some(stripped) = strip_date_suffix(model)
            && let Some(mult) = tier.models.get(stripped)
        {
            return *mult;
        }
        tier.default.unwrap_or(1.0)
    }
}

fn normalize_fast_tiers(
    raw: HashMap<String, HashMap<String, FastTierPricing>>,
) -> HashMap<String, HashMap<i8, FastTierPricing>> {
    raw.into_iter()
        .map(|(vendor, tiers)| {
            let tiers = tiers
                .into_iter()
                .filter_map(|(key, tier)| key.parse::<i8>().ok().map(|parsed| (parsed, tier)))
                .collect();
            (vendor, tiers)
        })
        .collect()
}

fn overlay_table(target: &mut HashMap<String, ModelPricing>, src: HashMap<String, ModelPricing>) {
    for (key, mut new) in src {
        if let Some(existing) = target.get(&key) {
            if new.input_above_200k.is_none() {
                new.input_above_200k = existing.input_above_200k;
            }
            if new.output_above_200k.is_none() {
                new.output_above_200k = existing.output_above_200k;
            }
            if new.cache_input_above_200k.is_none() {
                new.cache_input_above_200k = existing.cache_input_above_200k;
            }
            if new.cache_output_above_200k.is_none() {
                new.cache_output_above_200k = existing.cache_output_above_200k;
            }
        }
        target.insert(key, new);
    }
}

/// Find the newest known model that shares the target's provider, family, and
/// size/modifier class. Returns `None` for unknown-provider ids (no safe peer)
/// so the caller drops to the vendor default.
fn same_class_fallback<'a>(
    table: &'a HashMap<String, ModelPricing>,
    model: &str,
) -> Option<&'a ModelPricing> {
    let target = parse_model_identity(model);
    if target.provider == Provider::Unknown {
        return None;
    }
    table
        .iter()
        .filter_map(|(key, pricing)| {
            let candidate = parse_model_identity(key);
            same_class(&target, &candidate).then_some((candidate.version_key, pricing))
        })
        .max_by_key(|(version_key, _)| *version_key)
        .map(|(_, pricing)| pricing)
}

/// Two ids are the same pricing class when they agree on provider, family, and
/// the exact set of size/modifier tokens (so `gpt-5.5-mini` never matches the
/// base `gpt-5.5`).
fn same_class(a: &crate::model_id::ModelIdentity, b: &crate::model_id::ModelIdentity) -> bool {
    if a.provider != b.provider || a.family != b.family {
        return false;
    }
    let mut am = a.modifiers.clone();
    let mut bm = b.modifiers.clone();
    am.sort();
    bm.sort();
    am == bm
}

/// Strip a trailing `-YYYYMMDD` (8 digits, hyphen-separated) from a model name.
/// Returns the prefix slice if the suffix is present, otherwise `None`.
fn strip_date_suffix(model: &str) -> Option<&str> {
    let (prefix, last) = model.rsplit_once('-')?;
    if last.len() == 8 && last.bytes().all(|b| b.is_ascii_digit()) {
        Some(prefix)
    } else {
        None
    }
}

/// Add a stripped-date alias entry for every dated model so a JSONL emitting
/// a different date suffix (or no suffix at all) still resolves to the same
/// pricing without requiring a release.
fn expand_date_aliases(map: HashMap<String, ModelPricing>) -> HashMap<String, ModelPricing> {
    let mut result = map;
    let aliases: Vec<(String, ModelPricing)> = result
        .iter()
        .filter_map(|(k, v)| strip_date_suffix(k).map(|stripped| (stripped.to_string(), v.clone())))
        .filter(|(stripped, _)| !result.contains_key(stripped))
        .collect();
    for (k, v) in aliases {
        result.insert(k, v);
    }
    result
}

/// Fee key mapping
const FEE_KEYS: &[(&str, &str)] = &[
    ("CLAUDE_MONTHLY_FEE", "claude"),
    ("CODEX_MONTHLY_FEE", "codex"),
    ("GEMINI_MONTHLY_FEE", "gemini"),
];

#[derive(Debug, Clone)]
pub struct SubscriptionFees {
    pub claude: f64,
    pub codex: f64,
    pub gemini: f64,
}

impl Default for SubscriptionFees {
    fn default() -> Self {
        SubscriptionFees {
            claude: 0.0,
            codex: 0.0,
            gemini: 0.0,
        }
    }
}

impl SubscriptionFees {
    pub fn get(&self, vendor: &str) -> f64 {
        match vendor {
            "claude" => self.claude,
            "codex" => self.codex,
            "gemini" => self.gemini,
            "all" => self.claude + self.codex + self.gemini,
            _ => 0.0,
        }
    }
}

/// Get the fee.env file path (next to the binary's working directory).
pub fn fee_env_path() -> PathBuf {
    // Look relative to the executable or current directory
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    // Try multiple locations
    let candidates = [
        // Next to the executable
        exe_dir.as_ref().map(|d| d.join(".fee.env")),
        // Current working directory
        Some(PathBuf::from(".fee.env")),
    ];

    for candidate in candidates.iter().flatten() {
        if candidate.exists() {
            return candidate.clone();
        }
    }

    // Default to current directory
    PathBuf::from(".fee.env")
}

/// Load subscription fees from .fee.env file.
pub fn load_subscription_fees() -> Option<SubscriptionFees> {
    let path = fee_env_path();
    let content = std::fs::read_to_string(&path).ok()?;

    let mut fees: HashMap<String, f64> = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            for &(fee_key, vendor) in FEE_KEYS {
                if key == fee_key
                    && let Ok(v) = value.parse::<f64>()
                {
                    fees.insert(vendor.to_string(), v);
                }
            }
        }
    }

    if fees.len() == 3 {
        Some(SubscriptionFees {
            claude: fees["claude"],
            codex: fees["codex"],
            gemini: fees["gemini"],
        })
    } else {
        None
    }
}

/// Save subscription fees to .fee.env file.
pub fn save_subscription_fees(fees: &SubscriptionFees) -> std::io::Result<()> {
    let path = fee_env_path();
    let content = format!(
        "CLAUDE_MONTHLY_FEE={}\nCODEX_MONTHLY_FEE={}\nGEMINI_MONTHLY_FEE={}\n",
        fees.claude, fees.codex, fees.gemini
    );
    std::fs::write(&path, content)
}

/// Interactively prompt for subscription fees.
pub fn prompt_subscription_fees() -> SubscriptionFees {
    use std::io::{self, BufRead, Write};

    if !std::io::stdin().is_terminal() {
        eprintln!("Error: .fee.env not found and stdin is not a terminal.");
        eprintln!("Create .fee.env manually with:");
        eprintln!("  CLAUDE_MONTHLY_FEE=200");
        eprintln!("  CODEX_MONTHLY_FEE=200");
        eprintln!("  GEMINI_MONTHLY_FEE=19.99");
        std::process::exit(1);
    }

    println!("Subscription fee configuration not found.");
    println!("Please enter your monthly subscription fees:\n");

    let prompts = [
        ("claude", "Claude Code (Max)  monthly fee", 200.0),
        ("codex", "OpenAI Codex (Pro) monthly fee", 200.0),
        ("gemini", "Gemini CLI         monthly fee", 19.99),
    ];

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut values = Vec::new();

    for (_, label, default) in &prompts {
        loop {
            print!("  {} [${:.2}]: ", label, default);
            stdout.flush().unwrap();
            let mut line = String::new();
            stdin.lock().read_line(&mut line).unwrap();
            let line = line.trim();
            let value = if line.is_empty() {
                *default
            } else {
                match line.parse::<f64>() {
                    Ok(v) => v,
                    Err(_) => {
                        println!("    Invalid number, please try again.");
                        continue;
                    }
                }
            };
            values.push(value);
            break;
        }
    }

    let fees = SubscriptionFees {
        claude: values[0],
        codex: values[1],
        gemini: values[2],
    };

    if let Err(e) = save_subscription_fees(&fees) {
        eprintln!("Warning: Could not save .fee.env: {}", e);
    } else {
        let path = fee_env_path();
        println!("\nSaved to {}", path.display());
        println!("(Make sure .fee.env is in your .gitignore)\n");
    }

    fees
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_codex_pricing() -> ModelPricing {
        ModelPricing {
            input: 1.25,
            output: 10.0,
            cache_input: 0.125,
            cache_output: 0.0,
            input_above_200k: None,
            output_above_200k: None,
            cache_input_above_200k: None,
            cache_output_above_200k: None,
            _comment: None,
        }
    }

    #[test]
    fn pricing_falls_back_to_newest_in_same_family() {
        let p = AllPricing::load_raw().finalize();
        // claude-opus-4-9 is not embedded; it must borrow the newest opus rate
        // (5/25), not the sonnet-priced vendor default (3/15).
        let opus_new = p.get_pricing("claude", "claude-opus-4-9");
        assert!((opus_new.input - 5.0).abs() < 1e-9);
        assert!((opus_new.output - 25.0).abs() < 1e-9);

        // A future fable release borrows the newest fable rate (10/50).
        let fable_new = p.get_pricing("claude", "claude-fable-5-1");
        assert!((fable_new.input - 10.0).abs() < 1e-9);
        assert!((fable_new.output - 50.0).abs() < 1e-9);

        // gemini-3.2-pro borrows the newest pro rate (2.00), not the flash-priced
        // gemini default (0.50).
        let gem_new = p.get_pricing("gemini", "gemini-3.2-pro-preview");
        assert!((gem_new.input - 2.0).abs() < 1e-9);
    }

    #[test]
    fn pricing_embedded_covers_current_claude_models() {
        let p = AllPricing::load_raw().finalize();
        let fable = p.get_pricing("claude", "claude-fable-5");
        assert!((fable.input - 10.0).abs() < 1e-9);
        assert!((fable.output - 50.0).abs() < 1e-9);
        assert!((fable.cache_input - 1.0).abs() < 1e-9);
        assert!((fable.cache_output - 12.5).abs() < 1e-9);
        assert!(fable.input_above_200k.is_none());

        let mythos = p.get_pricing("claude", "claude-mythos-5");
        assert!((mythos.input - 10.0).abs() < 1e-9);

        let opus8 = p.get_pricing("claude", "claude-opus-4-8");
        assert!((opus8.input - 5.0).abs() < 1e-9);
        assert!((opus8.output - 25.0).abs() < 1e-9);
        assert!(opus8.input_above_200k.is_none());
    }

    #[test]
    fn overlay_remote_rates_win_over_embedded_baseline() {
        // The layered loader applies remote (cache/LiteLLM) tables on top of
        // the embedded baseline: remote base rates must win per-model, while
        // the embedded entry only fills in tier fields the remote layer omits
        // and keeps covering models the remote layer does not know about.
        let mut p = AllPricing::load_raw();
        let mut remote = HashMap::new();
        remote.insert(
            "claude-opus-4-7".to_string(),
            ModelPricing {
                input: 4.0,
                output: 20.0,
                cache_input: 0.4,
                cache_output: 5.0,
                input_above_200k: None,
                output_above_200k: None,
                cache_input_above_200k: None,
                cache_output_above_200k: None,
                _comment: None,
            },
        );
        p.overlay(remote, HashMap::new(), HashMap::new());
        let p = p.finalize();

        let opus7 = p.get_pricing("claude", "claude-opus-4-7");
        assert!((opus7.input - 4.0).abs() < 1e-9);
        assert!((opus7.output - 20.0).abs() < 1e-9);
        // Tier fields omitted by the remote layer survive from the baseline.
        assert_eq!(opus7.input_above_200k, Some(10.0));

        // Models absent from the remote layer keep their embedded rates.
        let fable = p.get_pricing("claude", "claude-fable-5");
        assert!((fable.input - 10.0).abs() < 1e-9);
    }

    #[test]
    fn pricing_fallback_respects_size_class() {
        let p = AllPricing::load_raw().finalize();
        // A future `-mini` must borrow a mini rate (newest = gpt-5.4-mini 0.75),
        // never the much pricier base gpt rate.
        let mini = p.get_pricing("codex", "gpt-5.9-mini");
        assert!((mini.input - 0.75).abs() < 1e-9);
        assert!(mini.input < 1.0, "must not inherit a base-class rate");
    }

    #[test]
    fn pricing_unknown_model_uses_vendor_default() {
        let p = AllPricing::load_raw().finalize();
        let unknown = p.get_pricing("claude", "totally-mystery-thing");
        assert!((unknown.input - 3.0).abs() < 1e-9);
    }

    #[test]
    fn pricing_override_wins_over_table_and_fallback() {
        let mut p = AllPricing::load_raw().finalize();
        let mut ov = HashMap::new();
        ov.insert(
            "claude-opus-4-7".to_string(),
            ModelPricing {
                input: 99.0,
                output: 199.0,
                cache_input: 9.0,
                cache_output: 9.0,
                input_above_200k: None,
                output_above_200k: None,
                cache_input_above_200k: None,
                cache_output_above_200k: None,
                _comment: None,
            },
        );
        p.set_pricing_overrides(ov);
        let got = p.get_pricing("claude", "claude-opus-4-7");
        assert!((got.input - 99.0).abs() < 1e-9);
    }

    #[test]
    fn pricing_for_entry_scales_codex_fast_from_record_tier() {
        let mut pricing = AllPricing::load_raw().finalize();
        pricing
            .codex_models
            .insert("gpt-5.5".to_string(), sample_codex_pricing());

        let p = pricing.pricing_for_entry("codex", "gpt-5.5", -1);
        assert!(matches!(p, Cow::Borrowed(_)));
        assert!((p.input - 1.25).abs() < f64::EPSILON);

        let p = pricing.pricing_for_entry("codex", "gpt-5.5", 0);
        assert!(matches!(p, Cow::Borrowed(_)));
        assert!((p.output - 10.0).abs() < f64::EPSILON);

        let p = pricing.pricing_for_entry("codex", "gpt-5.5", 1);
        assert!(matches!(p, Cow::Owned(_)));
        assert!((p.input - 1.25 * 2.5).abs() < 1e-9);
        assert!((p.output - 10.0 * 2.5).abs() < 1e-9);
        assert!((p.cache_input - 0.125 * 2.5).abs() < 1e-9);
    }

    #[test]
    fn pricing_for_entry_scales_claude_fast_model_specific_rate() {
        let pricing = AllPricing::load_raw().finalize();

        let standard = pricing.pricing_for_entry("claude", "claude-opus-4-7", 0);
        let fast = pricing.pricing_for_entry("claude", "claude-opus-4-7", 1);

        assert!(matches!(standard, Cow::Borrowed(_)));
        assert!(matches!(fast, Cow::Owned(_)));
        assert!((standard.input - 5.0).abs() < 1e-9);
        assert!((fast.input - 30.0).abs() < 1e-9);
        assert!((fast.output - 150.0).abs() < 1e-9);
    }

    #[test]
    fn pricing_for_entry_uses_standard_rate_when_factor_is_missing() {
        let pricing = AllPricing::load_raw().finalize();

        let p = pricing.pricing_for_entry("codex", "gpt-5.3-codex", 1);
        assert!(matches!(p, Cow::Borrowed(_)));

        let p = pricing.pricing_for_entry("gemini", "gemini-3-pro-preview", 1);
        assert!(matches!(p, Cow::Borrowed(_)));

        let p = pricing.pricing_for_entry("claude", "claude-opus-4-7", 2);
        assert!(matches!(p, Cow::Borrowed(_)));
    }
}
