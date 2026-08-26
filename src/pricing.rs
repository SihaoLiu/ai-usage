//! Layered pricing loader.
//!
//! Resolution order:
//!   1. Embedded `pricing.json` release-verified rates.
//!   2. A cached or live remote snapshot that only fills missing models.
//!   3. User overrides, applied by the caller as the final authority.
//!
//! Any layer that fails (network timeout, parse error, missing file, missing
//! `$HOME`) is silently skipped — the next layer takes over. The embedded
//! baseline guarantees we always have *something*.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::constants::{AllPricing, ModelPricing, VendorTables, replace_file};
use crate::model_id::{canonical_model_leaf, infer_vendor_with_provider};

const LITELLM_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

const CACHE_TTL: Duration = Duration::from_secs(24 * 3600);
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_SCHEMA_VERSION: u32 = 2;

/// Load embedded authority plus a cached or live supplement. Always returns a
/// valid `AllPricing`; degrades gracefully when offline.
pub fn load_layered() -> AllPricing {
    let mut combined = AllPricing::load_raw();

    let cached = read_cache_file();
    let supplemental = match cached {
        Some((tables, true)) => Some(tables),
        Some((tables, false)) => match fetch_live(FETCH_TIMEOUT) {
            Some(live) => {
                let _ = write_cache_file(&live);
                Some(live)
            }
            None => Some(tables),
        },
        None => fetch_live(FETCH_TIMEOUT).inspect(|live| {
            let _ = write_cache_file(live);
        }),
    };
    if let Some(tables) = supplemental {
        combined.overlay(tables);
    }

    let mut finalized = combined.finalize();
    // User overrides are the top layer, applied after date-alias expansion so
    // they win over the embedded baseline, the cache, and live LiteLLM data.
    finalized.set_pricing_overrides(crate::model_overrides::load().pricing.clone());
    finalized
}

// ---- cache file -----------------------------------------------------------

fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("ai-usage"))
}

fn cache_path() -> Option<PathBuf> {
    Some(cache_dir()?.join("pricing-cache.json"))
}

#[derive(Debug, Deserialize, Serialize)]
struct PricingCache {
    schema_version: u32,
    fetched_at: u64,
    tables: VendorTables,
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn decode_cache(content: &str) -> Option<PricingCache> {
    let cache: PricingCache = serde_json::from_str(content).ok()?;
    (cache.schema_version == CACHE_SCHEMA_VERSION).then_some(cache)
}

fn cache_is_fresh(cache: &PricingCache, now: u64) -> bool {
    now.checked_sub(cache.fetched_at)
        .is_some_and(|age| age <= CACHE_TTL.as_secs())
}

#[cfg(test)]
fn parse_cache(content: &str, now: u64) -> Option<PricingCache> {
    let cache = decode_cache(content)?;
    cache_is_fresh(&cache, now).then_some(cache)
}

fn read_cache_file() -> Option<(VendorTables, bool)> {
    let path = cache_path()?;
    let content = fs::read_to_string(&path).ok()?;
    let cache = decode_cache(&content)?;
    let fresh = cache_is_fresh(&cache, unix_timestamp());
    Some((cache.tables, fresh))
}

fn write_cache_file(tables: &VendorTables) -> std::io::Result<()> {
    let Some(path) = cache_path() else {
        return Ok(());
    };
    write_cache_file_at(&path, tables, unix_timestamp())
}

fn write_cache_file_at(path: &Path, tables: &VendorTables, fetched_at: u64) -> std::io::Result<()> {
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory)?;
    let cache = PricingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        fetched_at,
        tables: tables.clone(),
    };
    let json = serde_json::to_vec_pretty(&cache)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = directory.join(format!(".pricing-cache-{}-{nonce}.tmp", std::process::id()));
    fs::write(&temp_path, json)?;
    if let Err(error) = replace_file(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    Ok(())
}

// ---- live fetch -----------------------------------------------------------

fn fetch_live(timeout: Duration) -> Option<VendorTables> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .new_agent();
    let body: String = agent
        .get(LITELLM_URL)
        .call()
        .ok()?
        .body_mut()
        .read_to_string()
        .ok()?;
    parse_litellm_payload(&body)
}

// ---- LiteLLM JSON → ModelPricing -----------------------------------------

#[derive(Debug, Deserialize)]
struct LiteLLMEntry {
    #[serde(default)]
    input_cost_per_token: Option<f64>,
    #[serde(default)]
    output_cost_per_token: Option<f64>,
    #[serde(default)]
    cache_creation_input_token_cost: Option<f64>,
    #[serde(default)]
    cache_read_input_token_cost: Option<f64>,
    #[serde(default)]
    input_cost_per_token_above_200k_tokens: Option<f64>,
    #[serde(default)]
    output_cost_per_token_above_200k_tokens: Option<f64>,
    #[serde(default)]
    cache_creation_input_token_cost_above_200k_tokens: Option<f64>,
    #[serde(default)]
    cache_read_input_token_cost_above_200k_tokens: Option<f64>,
    #[serde(default)]
    litellm_provider: Option<String>,
    #[serde(default)]
    mode: Option<String>,
}

fn parse_litellm_payload(s: &str) -> Option<VendorTables> {
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_str(s).ok()?;

    let mut tables = VendorTables::default();
    for (raw_name, value) in raw {
        // The first entry in the LiteLLM JSON is a documentation stub.
        if raw_name == "sample_spec" {
            continue;
        }
        let Ok(entry) = serde_json::from_value::<LiteLLMEntry>(value) else {
            continue;
        };

        // Filter to chat-like modes when the field is present; LiteLLM uses
        // it to mark embeddings / image / audio models we don't want.
        if let Some(mode) = entry.mode.as_deref()
            && !matches!(mode, "chat" | "responses" | "completion")
        {
            continue;
        }

        let (Some(input), Some(output)) = (entry.input_cost_per_token, entry.output_cost_per_token)
        else {
            continue;
        };
        if output <= 0.0 {
            continue;
        }

        // LiteLLM omits cache_* costs for models that don't support caching.
        // Default them to the input rate so cost math doesn't divide by zero
        // and matches Anthropic's "cache_read falls back to input" convention.
        let cache_read = entry.cache_read_input_token_cost.unwrap_or(input);
        let cache_creation = entry.cache_creation_input_token_cost.unwrap_or(input);

        let pricing = ModelPricing {
            input: input * 1_000_000.0,
            output: output * 1_000_000.0,
            cache_input: cache_read * 1_000_000.0,
            cache_output: cache_creation * 1_000_000.0,
            // LiteLLM currently exposes one cache-creation rate. Treat it as
            // the fallback for both retention durations until it publishes
            // duration-specific fields.
            cache_output_1h: None,
            input_above_200k: entry
                .input_cost_per_token_above_200k_tokens
                .map(|v| v * 1_000_000.0),
            output_above_200k: entry
                .output_cost_per_token_above_200k_tokens
                .map(|v| v * 1_000_000.0),
            cache_input_above_200k: entry
                .cache_read_input_token_cost_above_200k_tokens
                .map(|v| v * 1_000_000.0),
            cache_output_above_200k: entry
                .cache_creation_input_token_cost_above_200k_tokens
                .map(|v| v * 1_000_000.0),
            cache_output_1h_above_200k: None,
            _comment: Some(format!("Source: LiteLLM ({})", raw_name)),
        };

        let vendor = infer_vendor_with_provider(&raw_name, entry.litellm_provider.as_deref());
        let Some(pricing_key) = vendor.pricing_key() else {
            continue;
        };
        let full_key = raw_name.trim().to_ascii_lowercase();
        let leaf_key = canonical_key(&raw_name);
        let first_party = entry
            .litellm_provider
            .as_deref()
            .map_or(full_key == leaf_key, |provider| {
                vendor.is_first_party_provider(provider)
            });
        let storage_key = if first_party {
            leaf_key
        } else if full_key != leaf_key || full_key.contains('/') {
            full_key.clone()
        } else {
            format!(
                "{}/{}",
                entry.litellm_provider.as_deref().unwrap_or("remote"),
                full_key
            )
        };
        tables.insert_model(pricing_key, storage_key, pricing);
    }

    Some(tables)
}

/// Derive the unqualified model key used for first-party aliases.
fn canonical_key(name: &str) -> String {
    canonical_model_leaf(name)
}

/// Expose registry classification to focused parser tests.
#[cfg(test)]
fn classify_vendor(name: &str, provider: Option<&str>) -> Option<&'static str> {
    infer_vendor_with_provider(name, provider).pricing_key()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_handles_provider_hints_and_names() {
        assert_eq!(
            classify_vendor("claude-3-5-sonnet-20241022", Some("anthropic")),
            Some("claude")
        );
        assert_eq!(classify_vendor("gpt-5", Some("openai")), Some("codex"));
        assert_eq!(
            classify_vendor("gemini-2.5-pro", Some("vertex_ai")),
            Some("gemini")
        );
        assert_eq!(classify_vendor("o1", None), Some("codex"));
        assert_eq!(classify_vendor("claude-opus-4-7", None), Some("claude"));
        assert_eq!(
            classify_vendor("mistral-large", Some("mistral")),
            Some("mistral")
        );
    }

    #[test]
    fn classify_routes_kimi_and_moonshot_to_kimi() {
        assert_eq!(classify_vendor("kimi-k2.5", Some("moonshot")), Some("kimi"));
        assert_eq!(classify_vendor("k3", Some("moonshot")), Some("kimi"));
        assert_eq!(classify_vendor("kimi-k2", None), Some("kimi"));
        assert_eq!(classify_vendor("kimi-for-coding", None), Some("kimi"));
    }

    #[test]
    fn litellm_moonshot_entries_land_in_kimi_table() {
        let payload = r#"{
            "moonshot/kimi-k2.5": {
                "input_cost_per_token": 0.0000006,
                "output_cost_per_token": 0.000003,
                "cache_read_input_token_cost": 0.0000001,
                "litellm_provider": "moonshot",
                "mode": "chat"
            }
        }"#;
        let tables = parse_litellm_payload(payload).expect("parse");
        let entry = tables
            .models("kimi")
            .and_then(|models| models.get("kimi-k2.5"))
            .expect("kimi entry present");
        assert!((entry.input - 0.6).abs() < 1e-9);
        assert!((entry.output - 3.0).abs() < 1e-9);
        assert!((entry.cache_input - 0.1).abs() < 1e-9);
    }

    #[test]
    fn canonical_key_strips_provider_prefixes() {
        assert_eq!(canonical_key("claude-3-5-sonnet"), "claude-3-5-sonnet");
        assert_eq!(
            canonical_key("anthropic/claude-3-5-sonnet"),
            "claude-3-5-sonnet"
        );
        assert_eq!(canonical_key("anthropic.claude-3"), "claude-3");
        assert_eq!(canonical_key("vertex_ai/gemini-2.5-pro"), "gemini-2.5-pro");
    }

    #[test]
    fn remote_tables_keep_provider_keys_and_only_alias_first_party_models() {
        let payload = r#"{
            "gpt-5.6-sol": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000030,
                "litellm_provider": "openai",
                "mode": "responses"
            },
            "azure/eu/gpt-5.6-sol": {
                "input_cost_per_token": 0.0000055,
                "output_cost_per_token": 0.000033,
                "litellm_provider": "azure",
                "mode": "responses"
            },
            "deepseek/deepseek-v4-pro": {
                "input_cost_per_token": 0.000000435,
                "output_cost_per_token": 0.00000087,
                "litellm_provider": "deepseek",
                "mode": "chat"
            },
            "bedrock/anthropic.claude-sonnet-4-6": {
                "input_cost_per_token": 0.0000033,
                "output_cost_per_token": 0.0000165,
                "mode": "chat"
            },
            "google/gemini-4-flash": {
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000010,
                "litellm_provider": "google",
                "mode": "chat"
            },
            "vertex_ai.gemini-4-hosted": {
                "input_cost_per_token": 0.000009,
                "output_cost_per_token": 0.000090,
                "mode": "chat"
            }
        }"#;

        let tables = parse_litellm_payload(payload).expect("parse");
        let openai = tables.models("codex").expect("openai table");
        assert!((openai["gpt-5.6-sol"].input - 5.0).abs() < 1e-9);
        assert!((openai["azure/eu/gpt-5.6-sol"].input - 5.5).abs() < 1e-9);
        let deepseek = tables.models("deepseek").expect("deepseek table");
        assert!((deepseek["deepseek-v4-pro"].input - 0.435).abs() < 1e-9);
        let anthropic = tables.models("claude").expect("anthropic table");
        assert!(anthropic.contains_key("bedrock/anthropic.claude-sonnet-4-6"));
        assert!(!anthropic.contains_key("claude-sonnet-4-6"));
        let google = tables.models("gemini").expect("google table");
        assert!(google.contains_key("gemini-4-flash"));
        assert!(google.contains_key("vertex_ai.gemini-4-hosted"));
        assert!(!google.contains_key("gemini-4-hosted"));
    }

    #[test]
    fn embedded_rates_win_over_first_party_qualified_and_derived_remote_keys() {
        let payload = r#"{
            "deepseek/deepseek-v4-pro": {
                "input_cost_per_token": 0.000099,
                "output_cost_per_token": 0.000199,
                "litellm_provider": "deepseek",
                "mode": "chat"
            },
            "claude-sonnet-4-5": {
                "input_cost_per_token": 0.000099,
                "output_cost_per_token": 0.000199,
                "litellm_provider": "anthropic",
                "mode": "chat"
            }
        }"#;
        let tables = parse_litellm_payload(payload).expect("parse");
        let mut pricing = AllPricing::load_raw();
        pricing.overlay(tables);
        let pricing = pricing.finalize();

        let deepseek = pricing.get_pricing("deepseek", "deepseek/deepseek-v4-pro");
        assert!((deepseek.input - 0.435).abs() < 1e-9);
        assert!((deepseek.output - 0.87).abs() < 1e-9);

        let sonnet = pricing.get_pricing("claude", "claude-sonnet-4-5");
        assert!((sonnet.input - 3.0).abs() < 1e-9);
        assert!((sonnet.output - 15.0).abs() < 1e-9);
    }

    #[test]
    fn cache_requires_current_schema_and_fresh_timestamp() {
        let tables = VendorTables::default();
        let cache = PricingCache {
            schema_version: CACHE_SCHEMA_VERSION,
            fetched_at: 1_000,
            tables,
        };
        let json = serde_json::to_string(&cache).expect("serialize");

        assert!(parse_cache(&json, 1_000 + CACHE_TTL.as_secs()).is_some());
        assert!(parse_cache(&json, 1_001 + CACHE_TTL.as_secs()).is_none());
        assert!(parse_cache("{not-json", 1_000).is_none());
        assert!(parse_cache(r#"{"claude": {}}"#, 1_000).is_none());
    }

    #[test]
    fn cache_write_replaces_an_existing_snapshot() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("temp")
            .join(format!(
                "pricing-cache-replace-{}-{nonce}",
                std::process::id()
            ));
        let path = directory.join("pricing-cache.json");

        let mut first = VendorTables::default();
        first.insert_model("codex", "first-model".to_string(), sample_pricing(1.0));
        write_cache_file_at(&path, &first, 1_000).expect("write first cache");

        let mut second = VendorTables::default();
        second.insert_model("deepseek", "second-model".to_string(), sample_pricing(2.0));
        write_cache_file_at(&path, &second, 2_000).expect("replace cache");

        let content = fs::read_to_string(&path).expect("read replaced cache");
        let cache = decode_cache(&content).expect("decode replaced cache");
        assert_eq!(cache.fetched_at, 2_000);
        assert!(cache.tables.models("codex").is_none());
        assert!(
            cache
                .tables
                .models("deepseek")
                .is_some_and(|models| models.contains_key("second-model"))
        );

        fs::remove_dir_all(directory).expect("remove cache test directory");
    }

    fn sample_pricing(input: f64) -> ModelPricing {
        ModelPricing {
            input,
            output: input,
            cache_input: input,
            cache_output: input,
            cache_output_1h: None,
            input_above_200k: None,
            output_above_200k: None,
            cache_input_above_200k: None,
            cache_output_above_200k: None,
            cache_output_1h_above_200k: None,
            _comment: None,
        }
    }

    #[test]
    fn litellm_per_token_converts_to_per_mtok() {
        let payload = r#"{
            "sample_spec": {"foo": "bar"},
            "claude-3-5-sonnet-20241022": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
                "input_cost_per_token_above_200k_tokens": 0.000006,
                "litellm_provider": "anthropic",
                "mode": "chat"
            },
            "text-embedding-3-large": {
                "input_cost_per_token": 0.00000013,
                "output_cost_per_token": 0,
                "litellm_provider": "openai",
                "mode": "embedding"
            }
        }"#;
        let tables = parse_litellm_payload(payload).expect("parse");
        let entry = tables
            .models("claude")
            .expect("claude table")
            .get("claude-3-5-sonnet-20241022")
            .expect("claude sonnet present");
        assert!((entry.input - 3.0).abs() < 1e-9);
        assert!((entry.output - 15.0).abs() < 1e-9);
        assert!((entry.cache_input - 0.3).abs() < 1e-9);
        assert!((entry.cache_output - 3.75).abs() < 1e-9);
        assert_eq!(entry.input_above_200k, Some(6.0));
        // Embedding model must be filtered out (mode != chat).
        assert!(tables.models("codex").is_none());
        assert!(tables.models("gemini").is_none());
    }
}
