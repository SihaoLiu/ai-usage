use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::PathBuf;

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
        if let Some(above) = above_per_mtok {
            if tokens > TIER_THRESHOLD {
                let below = TIER_THRESHOLD as f64;
                let extra = (tokens - TIER_THRESHOLD) as f64;
                return below * base_per_mtok / 1_000_000.0
                    + extra * above / 1_000_000.0;
            }
        }
        tokens as f64 * base_per_mtok / 1_000_000.0
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
        if let Some(stripped) = strip_date_suffix(model) {
            if let Some(p) = table.get(stripped) {
                return p;
            }
        }

        default
    }
}

fn overlay_table(
    target: &mut HashMap<String, ModelPricing>,
    src: HashMap<String, ModelPricing>,
) {
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
        .filter_map(|(k, v)| {
            strip_date_suffix(k).map(|stripped| (stripped.to_string(), v.clone()))
        })
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
                if key == fee_key {
                    if let Ok(v) = value.parse::<f64>() {
                        fees.insert(vendor.to_string(), v);
                    }
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
