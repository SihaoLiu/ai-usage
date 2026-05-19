use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::PathBuf;

/// Embedded pricing data from pricing.json
const PRICING_JSON: &str = include_str!("../pricing.json");

/// Token threshold for Claude 1M-context tiered pricing.
/// Mirrors ccusage's `DEFAULT_TIERED_THRESHOLD` and Anthropic's published 200k tier.
pub const TIER_THRESHOLD: i64 = 200_000;

/// OpenAI Codex API service tier. Set globally in `~/.codex/config.toml` via
/// `service_tier = "fast"|"flex"`. Codex rollout JSONL files do not record
/// this per-turn, so the effective tier is resolved once from the config (or
/// a CLI override) and applied uniformly to every Codex usage entry.
///
/// Multipliers come from the official Codex speed docs
/// (https://developers.openai.com/codex/speed):
///   - Fast/Priority: gpt-5.5 = 2.5x, gpt-5.4 family = 2.0x, others = 1.0x
///   - Flex:           ~0.5x of standard for supported models
///   - Default/Auto:   1.0x (standard)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodexServiceTier {
    #[default]
    Default,
    Fast,
    Flex,
}

impl CodexServiceTier {
    /// Parse a string value from `config.toml` / CLI flag into a tier.
    /// Returns `None` for unknown values so the caller can decide how to fall
    /// back. "priority" is treated as a synonym of "fast" because OpenAI's
    /// general Responses API exposes priority processing under that name.
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" | "standard" | "auto" => Some(Self::Default),
            "fast" | "priority" => Some(Self::Fast),
            "flex" => Some(Self::Flex),
            _ => None,
        }
    }

    /// Per-model cost multiplier vs the standard API rate.
    pub fn cost_multiplier(self, model: &str) -> f64 {
        match self {
            CodexServiceTier::Default => 1.0,
            CodexServiceTier::Fast => fast_tier_multiplier(model),
            CodexServiceTier::Flex => flex_tier_multiplier(model),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            CodexServiceTier::Default => "default",
            CodexServiceTier::Fast => "fast",
            CodexServiceTier::Flex => "flex",
        }
    }
}

/// Fast-tier multiplier per Codex pricing docs. Models that don't support
/// fast mode fall back to 1.0x because OpenAI bills them at standard rate
/// even when the project default is fast.
fn fast_tier_multiplier(model: &str) -> f64 {
    let m = model.to_ascii_lowercase();
    if m.starts_with("gpt-5.5") {
        2.5
    } else if m.starts_with("gpt-5.4") {
        2.0
    } else {
        1.0
    }
}

/// Flex-tier discount. OpenAI describes flex as roughly 50% of standard for
/// the supported gpt-5 family; we apply 0.5x conservatively to the same model
/// set that supports fast, and leave older/unsupported SKUs at 1.0x.
fn flex_tier_multiplier(model: &str) -> f64 {
    let m = model.to_ascii_lowercase();
    if m.starts_with("gpt-5") { 0.5 } else { 1.0 }
}

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

    /// Scale every rate (base + tier overrides) by the Codex service-tier
    /// multiplier for `model`. Returns `self` unchanged when the multiplier
    /// is 1.0 so non-fast usage stays free of clones.
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
}

/// All pricing tables for the three vendors.
pub struct AllPricing {
    pub claude_models: HashMap<String, ModelPricing>,
    pub claude_default: ModelPricing,
    pub codex_models: HashMap<String, ModelPricing>,
    pub codex_default: ModelPricing,
    pub gemini_models: HashMap<String, ModelPricing>,
    pub gemini_default: ModelPricing,
    /// Effective Codex API service tier (resolved from config.toml or CLI).
    /// Drives the per-model `fast`/`flex` multipliers applied in
    /// `pricing_for_entry`.
    pub codex_service_tier: CodexServiceTier,
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
            codex_service_tier: CodexServiceTier::default(),
        }
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
    /// one omits them — LiteLLM currently doesn't carry `*_above_200k_tokens`
    /// for several 1M-context Claude SKUs, and we don't want overlaying flat
    /// LiteLLM data to silently erase the tier knowledge in the embedded
    /// baseline.
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

        default
    }

    /// Pricing for a single usage entry with the Codex service-tier
    /// multiplier already applied. For Claude/Gemini, and for Codex when the
    /// effective tier is `Default`, returns a borrow of the raw row so the
    /// hot aggregation path stays allocation-free. For Codex `Fast`/`Flex`,
    /// returns an owned, scaled copy.
    pub fn pricing_for_entry<'a>(&'a self, vendor: &str, model: &str) -> Cow<'a, ModelPricing> {
        let base = self.get_pricing(vendor, model);
        if vendor != "codex" {
            return Cow::Borrowed(base);
        }
        let mult = self.codex_service_tier.cost_multiplier(model);
        if (mult - 1.0).abs() < f64::EPSILON {
            Cow::Borrowed(base)
        } else {
            Cow::Owned(base.scaled_by(mult))
        }
    }
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

    #[test]
    fn service_tier_from_str_recognises_codex_and_priority_synonyms() {
        assert_eq!(
            CodexServiceTier::from_str("fast"),
            Some(CodexServiceTier::Fast)
        );
        assert_eq!(
            CodexServiceTier::from_str("PRIORITY"),
            Some(CodexServiceTier::Fast)
        );
        assert_eq!(
            CodexServiceTier::from_str(" flex "),
            Some(CodexServiceTier::Flex)
        );
        assert_eq!(
            CodexServiceTier::from_str("default"),
            Some(CodexServiceTier::Default)
        );
        assert_eq!(
            CodexServiceTier::from_str("standard"),
            Some(CodexServiceTier::Default)
        );
        assert_eq!(CodexServiceTier::from_str("turbo"), None);
    }

    #[test]
    fn fast_tier_uses_documented_multipliers() {
        let fast = CodexServiceTier::Fast;
        assert!((fast.cost_multiplier("gpt-5.5") - 2.5).abs() < f64::EPSILON);
        assert!((fast.cost_multiplier("gpt-5.5-codex") - 2.5).abs() < f64::EPSILON);
        assert!((fast.cost_multiplier("gpt-5.4") - 2.0).abs() < f64::EPSILON);
        assert!((fast.cost_multiplier("gpt-5.4-codex") - 2.0).abs() < f64::EPSILON);
        // Unsupported / older models keep standard pricing.
        assert!((fast.cost_multiplier("gpt-5.3-codex") - 1.0).abs() < f64::EPSILON);
        assert!((fast.cost_multiplier("gpt-5") - 1.0).abs() < f64::EPSILON);
        // Default tier never moves the rate.
        assert!((CodexServiceTier::Default.cost_multiplier("gpt-5.5") - 1.0).abs() < f64::EPSILON);
    }

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
    fn pricing_for_entry_scales_codex_fast_and_borrows_others() {
        let mut pricing = AllPricing::load_raw().finalize();
        pricing
            .codex_models
            .insert("gpt-5.5".to_string(), sample_codex_pricing());

        // Default tier: borrow the row as-is, no scaling.
        pricing.codex_service_tier = CodexServiceTier::Default;
        let p = pricing.pricing_for_entry("codex", "gpt-5.5");
        assert!(matches!(p, Cow::Borrowed(_)));
        assert!((p.input - 1.25).abs() < f64::EPSILON);

        // Fast tier: scale by the gpt-5.5 multiplier (2.5x).
        pricing.codex_service_tier = CodexServiceTier::Fast;
        let p = pricing.pricing_for_entry("codex", "gpt-5.5");
        assert!(matches!(p, Cow::Owned(_)));
        assert!((p.input - 1.25 * 2.5).abs() < 1e-9);
        assert!((p.output - 10.0 * 2.5).abs() < 1e-9);
        assert!((p.cache_input - 0.125 * 2.5).abs() < 1e-9);

        // Fast tier on an unsupported codex model leaves the rate untouched.
        let p = pricing.pricing_for_entry("codex", "gpt-5.3-codex");
        assert!(matches!(p, Cow::Borrowed(_)));

        // Fast tier never affects non-codex vendors.
        let p = pricing.pricing_for_entry("claude", "claude-opus-4-7");
        assert!(matches!(p, Cow::Borrowed(_)));
    }
}
