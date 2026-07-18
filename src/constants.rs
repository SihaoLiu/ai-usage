use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

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

/// Per-vendor model pricing tables, one field per tracked vendor. Used both
/// as an overlay payload for [`AllPricing::overlay`] and as the on-disk /
/// LiteLLM parse shape in `crate::pricing`.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct VendorTables {
    #[serde(default)]
    pub claude: HashMap<String, ModelPricing>,
    #[serde(default)]
    pub codex: HashMap<String, ModelPricing>,
    #[serde(default)]
    pub gemini: HashMap<String, ModelPricing>,
    #[serde(default)]
    pub kimi: HashMap<String, ModelPricing>,
}

#[derive(Debug, Deserialize)]
struct PricingData {
    #[serde(rename = "_meta")]
    #[serde(default)]
    _meta: Option<serde_json::Value>,
    claude: VendorPricing,
    codex: VendorPricing,
    gemini: VendorPricing,
    kimi: VendorPricing,
    #[serde(default)]
    fast_tiers: HashMap<String, HashMap<String, FastTierPricing>>,
}

/// All pricing tables for the tracked vendors.
pub struct AllPricing {
    pub claude_models: HashMap<String, ModelPricing>,
    pub claude_default: ModelPricing,
    pub codex_models: HashMap<String, ModelPricing>,
    pub codex_default: ModelPricing,
    pub gemini_models: HashMap<String, ModelPricing>,
    pub gemini_default: ModelPricing,
    pub kimi_models: HashMap<String, ModelPricing>,
    pub kimi_default: ModelPricing,
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
            kimi_models: data.kimi.models,
            kimi_default: data.kimi.default,
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
        self.kimi_models = expand_date_aliases(self.kimi_models);
        self
    }

    /// Overlay another set of vendor tables into this one. Base rates from the
    /// incoming layer win (so live LiteLLM data refreshes prices), but optional
    /// tier-pricing fields fall back to the existing entry when the incoming
    /// one omits them.
    pub fn overlay(&mut self, tables: VendorTables) {
        overlay_table(&mut self.claude_models, tables.claude);
        overlay_table(&mut self.codex_models, tables.codex);
        overlay_table(&mut self.gemini_models, tables.gemini);
        overlay_table(&mut self.kimi_models, tables.kimi);
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
            "kimi" => (&self.kimi_models, &self.kimi_default),
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
    ("KIMI_MONTHLY_FEE", "kimi"),
];

#[derive(Debug, Clone)]
pub struct SubscriptionFees {
    pub claude: f64,
    pub codex: f64,
    pub gemini: f64,
    pub kimi: f64,
}

impl Default for SubscriptionFees {
    fn default() -> Self {
        SubscriptionFees {
            claude: 0.0,
            codex: 0.0,
            gemini: 0.0,
            kimi: 0.0,
        }
    }
}

impl SubscriptionFees {
    pub fn get(&self, vendor: &str) -> f64 {
        match vendor {
            "claude" => self.claude,
            "codex" => self.codex,
            "gemini" => self.gemini,
            "kimi" => self.kimi,
            "all" => self.claude + self.codex + self.gemini + self.kimi,
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

/// Vendors whose fee keys every .fee.env has always contained. A file
/// missing one of these is incomplete user input and defers to the
/// interactive prompt, exactly as before newer vendors existed.
const LEGACY_FEE_VENDORS: [&str; 3] = ["claude", "codex", "gemini"];

/// Parse fee lines into vendor -> value. A recognized key whose value fails
/// to parse maps to `None` (present but malformed), which is distinct from
/// the key being absent from the file entirely.
fn parse_fee_lines(content: &str) -> HashMap<String, Option<f64>> {
    let mut fees: HashMap<String, Option<f64>> = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            for &(fee_key, vendor) in FEE_KEYS {
                if key == fee_key {
                    fees.insert(vendor.to_string(), value.parse::<f64>().ok());
                }
            }
        }
    }
    fees
}

/// Interpret the parsed fee keys. `None` defers to the interactive prompt:
/// either a legacy key is absent (incomplete user input) or any recognized
/// key has a malformed value (silently zeroing it would discard the user's
/// intent). Otherwise returns the fees plus the env keys of newer vendors
/// the file does not name yet, which default to 0 until the user sets them.
fn interpret_fee_keys(
    fees: &HashMap<String, Option<f64>>,
) -> Option<(SubscriptionFees, Vec<&'static str>)> {
    if fees.values().any(|value| value.is_none()) {
        return None;
    }
    if LEGACY_FEE_VENDORS
        .iter()
        .any(|vendor| !fees.contains_key(*vendor))
    {
        return None;
    }
    let get = |vendor: &str| fees.get(vendor).copied().flatten().unwrap_or(0.0);
    let missing: Vec<&'static str> = FEE_KEYS
        .iter()
        .filter(|(_, vendor)| !fees.contains_key(*vendor))
        .map(|(key, _)| *key)
        .collect();
    Some((
        SubscriptionFees {
            claude: get("claude"),
            codex: get("codex"),
            gemini: get("gemini"),
            kimi: get("kimi"),
        },
        missing,
    ))
}

/// Append `KEY=0` lines for fee keys the file predates, preserving every
/// existing line and comment. Keeps the file complete so the note below
/// appears only once per upgrade.
fn append_missing_fee_keys(path: &Path, keys: &[&str]) {
    use std::io::Write;

    let needs_newline = std::fs::read_to_string(path)
        .map(|content| !content.is_empty() && !content.ends_with('\n'))
        .unwrap_or(false);
    let mut lines = String::new();
    if needs_newline {
        lines.push('\n');
    }
    for key in keys {
        lines.push_str(key);
        lines.push_str("=0\n");
    }
    let appended = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(lines.as_bytes()));
    if appended.is_ok() {
        eprintln!(
            "Note: added {} = 0 to {}; edit the file if you pay for that subscription.",
            keys.join(", "),
            path.display()
        );
    }
}

/// Load subscription fees from the .fee.env file.
pub fn load_subscription_fees() -> Option<SubscriptionFees> {
    let path = fee_env_path();
    let content = std::fs::read_to_string(&path).ok()?;
    let parsed = parse_fee_lines(&content);
    for &(key, vendor) in FEE_KEYS {
        if parsed.get(vendor) == Some(&None) {
            eprintln!(
                "Error: invalid value for {key} in {}; fix its value (a plain number)",
                path.display()
            );
        }
    }
    let (fees, missing_keys) = interpret_fee_keys(&parsed)?;
    if !missing_keys.is_empty() {
        append_missing_fee_keys(&path, &missing_keys);
    }
    Some(fees)
}

/// Save subscription fees to .fee.env file.
pub fn save_subscription_fees(fees: &SubscriptionFees) -> std::io::Result<()> {
    let path = fee_env_path();
    let content = format!(
        "CLAUDE_MONTHLY_FEE={}\nCODEX_MONTHLY_FEE={}\nGEMINI_MONTHLY_FEE={}\nKIMI_MONTHLY_FEE={}\n",
        fees.claude, fees.codex, fees.gemini, fees.kimi
    );
    std::fs::write(&path, content)
}

/// Interactively prompt for subscription fees.
pub fn prompt_subscription_fees() -> SubscriptionFees {
    use std::io::{self, BufRead, Write};

    if !std::io::stdin().is_terminal() {
        eprintln!("Error: no usable .fee.env and stdin is not a terminal.");
        eprintln!("Create or fix .fee.env manually with:");
        eprintln!("  CLAUDE_MONTHLY_FEE=200");
        eprintln!("  CODEX_MONTHLY_FEE=200");
        eprintln!("  GEMINI_MONTHLY_FEE=19.99");
        eprintln!("  KIMI_MONTHLY_FEE=40");
        std::process::exit(1);
    }

    println!("Subscription fee configuration is missing or incomplete.");
    println!("Please enter your monthly subscription fees:\n");

    let prompts = [
        ("claude", "Claude Code (Max)  monthly fee", 200.0),
        ("codex", "OpenAI Codex (Pro) monthly fee", 200.0),
        ("gemini", "Gemini CLI         monthly fee", 19.99),
        ("kimi", "Kimi Code          monthly fee", 40.0),
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
        kimi: values[3],
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
        p.overlay(VendorTables {
            claude: remote,
            ..VendorTables::default()
        });
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
    fn pricing_embedded_covers_kimi_models() {
        let p = AllPricing::load_raw().finalize();
        let k3 = p.get_pricing("kimi", "k3");
        assert!((k3.input - 3.0).abs() < 1e-9);
        assert!((k3.output - 15.0).abs() < 1e-9);
        assert!((k3.cache_input - 0.30).abs() < 1e-9);
        assert!(k3.input_above_200k.is_none());

        let coding = p.get_pricing("kimi", "kimi-for-coding");
        assert!((coding.input - 0.95).abs() < 1e-9);
        assert!((coding.output - 4.0).abs() < 1e-9);
        assert!((coding.cache_input - 0.19).abs() < 1e-9);
    }

    #[test]
    fn pricing_kimi_fallback_and_default() {
        let p = AllPricing::load_raw().finalize();
        // A future flagship borrows the newest same-class rate (k3), not the
        // coding-tier rate.
        let k4 = p.get_pricing("kimi", "k4");
        assert!((k4.input - 3.0).abs() < 1e-9);
        assert!((k4.output - 15.0).abs() < 1e-9);
        // Unknown-provider ids drop to the kimi vendor default.
        let unknown = p.get_pricing("kimi", "totally-mystery-thing");
        assert!((unknown.input - 3.0).abs() < 1e-9);
        assert!((unknown.output - 15.0).abs() < 1e-9);
    }

    #[test]
    fn fee_lines_parse_keeps_malformed_values_distinct_from_absent_keys() {
        let parsed = parse_fee_lines(
            "# switched to Max plan in June\nCLAUDE_MONTHLY_FEE=$200\nCODEX_MONTHLY_FEE=200\nGEMINI_MONTHLY_FEE=19.99\nUNRELATED=5\n",
        );
        assert_eq!(parsed.len(), 3);
        // A recognized key with an unparsable value is present-but-malformed,
        // not absent: it must never be treated as a missing newer vendor.
        assert_eq!(parsed["claude"], None);
        assert_eq!(parsed["codex"], Some(200.0));
        assert_eq!(parsed["gemini"], Some(19.99));
        assert!(!parsed.contains_key("kimi"));
    }

    #[test]
    fn fee_keys_missing_a_legacy_vendor_or_malformed_defer_to_the_prompt() {
        let mut fees = HashMap::new();
        fees.insert("codex".to_string(), Some(200.0));
        fees.insert("gemini".to_string(), Some(19.99));
        assert!(interpret_fee_keys(&fees).is_none());
        assert!(interpret_fee_keys(&HashMap::new()).is_none());

        // Any malformed value defers to the prompt, kimi included: silently
        // pinning it to 0 would discard the user's intended fee.
        let mut malformed_kimi = HashMap::new();
        malformed_kimi.insert("claude".to_string(), Some(100.0));
        malformed_kimi.insert("codex".to_string(), Some(200.0));
        malformed_kimi.insert("gemini".to_string(), Some(19.99));
        malformed_kimi.insert("kimi".to_string(), None);
        assert!(interpret_fee_keys(&malformed_kimi).is_none());
    }

    #[test]
    fn fee_keys_missing_only_newer_vendors_load_with_zero() {
        let mut fees = HashMap::new();
        fees.insert("claude".to_string(), Some(100.0));
        fees.insert("codex".to_string(), Some(200.0));
        fees.insert("gemini".to_string(), Some(19.99));

        let (loaded, missing) = interpret_fee_keys(&fees).expect("legacy-complete file loads");
        assert!((loaded.claude - 100.0).abs() < f64::EPSILON);
        assert!((loaded.kimi - 0.0).abs() < f64::EPSILON);
        assert_eq!(missing, vec!["KIMI_MONTHLY_FEE"]);

        fees.insert("kimi".to_string(), Some(40.0));
        let (loaded, missing) = interpret_fee_keys(&fees).expect("complete file loads");
        assert!((loaded.kimi - 40.0).abs() < f64::EPSILON);
        assert!(missing.is_empty());
    }

    #[test]
    fn appending_missing_fee_keys_preserves_existing_lines() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ai-usage-fee-test-{stamp}.env"));
        std::fs::write(&path, "# my comment\nCLAUDE_MONTHLY_FEE=100").expect("write fee file");

        append_missing_fee_keys(&path, &["KIMI_MONTHLY_FEE"]);
        let content = std::fs::read_to_string(&path).expect("read fee file");
        std::fs::remove_file(&path).ok();

        assert_eq!(
            content,
            "# my comment\nCLAUDE_MONTHLY_FEE=100\nKIMI_MONTHLY_FEE=0\n"
        );
    }

    #[test]
    fn subscription_fees_include_kimi_in_all() {
        let fees = SubscriptionFees {
            claude: 1.0,
            codex: 2.0,
            gemini: 3.0,
            kimi: 4.0,
        };
        assert!((fees.get("kimi") - 4.0).abs() < f64::EPSILON);
        assert!((fees.get("all") - 10.0).abs() < f64::EPSILON);
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
