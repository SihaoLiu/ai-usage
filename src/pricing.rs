//! Layered pricing loader.
//!
//! Resolution order (later layers win per-model, never drop earlier layers):
//!   1. Embedded `pricing.json` baseline (compile-time) — covers
//!      project-specific or future model names that LiteLLM may not carry.
//!   2. `~/.cache/ai-usage/pricing-cache.json` — last good LiteLLM snapshot.
//!   3. Live LiteLLM remote (5s timeout, refreshed at most every 24h).
//!
//! Any layer that fails (network timeout, parse error, missing file, missing
//! `$HOME`) is silently skipped — the next layer takes over. The embedded
//! baseline guarantees we always have *something*.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::constants::{AllPricing, ModelPricing};

const LITELLM_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

const CACHE_TTL: Duration = Duration::from_secs(24 * 3600);
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Load pricing with embedded → cache → live overlay. Always returns a valid
/// `AllPricing`; degrades gracefully when offline.
pub fn load_layered() -> AllPricing {
    let mut combined = AllPricing::load_raw();

    if let Some(cached) = read_cache_file() {
        combined.overlay(cached.claude, cached.codex, cached.gemini);
    }

    if cache_is_stale() {
        if let Some(live) = fetch_live(FETCH_TIMEOUT) {
            let _ = write_cache_file(&live);
            combined.overlay(live.claude, live.codex, live.gemini);
        }
    }

    combined.finalize()
}

/// Per-vendor model tables. Shape used both for the on-disk cache file and as
/// the parser output for the live LiteLLM response.
#[derive(Debug, Default, Deserialize, Serialize)]
struct VendorTables {
    #[serde(default)]
    claude: HashMap<String, ModelPricing>,
    #[serde(default)]
    codex: HashMap<String, ModelPricing>,
    #[serde(default)]
    gemini: HashMap<String, ModelPricing>,
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

fn cache_is_stale() -> bool {
    let Some(path) = cache_path() else {
        return true;
    };
    let Ok(meta) = fs::metadata(&path) else {
        return true;
    };
    let Ok(mtime) = meta.modified() else {
        return true;
    };
    SystemTime::now()
        .duration_since(mtime)
        .map(|age| age > CACHE_TTL)
        .unwrap_or(true)
}

fn read_cache_file() -> Option<VendorTables> {
    let path = cache_path()?;
    let content = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn write_cache_file(tables: &VendorTables) -> std::io::Result<()> {
    let Some(dir) = cache_dir() else {
        return Ok(());
    };
    fs::create_dir_all(&dir)?;
    let Some(path) = cache_path() else {
        return Ok(());
    };
    let json = serde_json::to_string_pretty(tables)
        .unwrap_or_else(|_| "{}".to_string());
    fs::write(&path, json)
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
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(s).ok()?;

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
        if let Some(mode) = entry.mode.as_deref() {
            if !matches!(mode, "chat" | "responses" | "completion") {
                continue;
            }
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
            _comment: Some(format!("Source: LiteLLM ({})", raw_name)),
        };

        let canonical = canonical_key(&raw_name);
        let Some(vendor) = classify_vendor(&canonical, entry.litellm_provider.as_deref()) else {
            continue;
        };
        let bucket = match vendor {
            "claude" => &mut tables.claude,
            "codex" => &mut tables.codex,
            "gemini" => &mut tables.gemini,
            _ => continue,
        };
        bucket.insert(canonical, pricing);
    }

    Some(tables)
}

/// Strip LiteLLM's leading provider segments so the key matches ai-usage's
/// flat naming (e.g. `anthropic/claude-...` and `vertex_ai/gemini-...`).
fn canonical_key(name: &str) -> String {
    let after_slash = name.rsplit_once('/').map_or(name, |(_, s)| s);
    after_slash
        .strip_prefix("anthropic.")
        .unwrap_or(after_slash)
        .to_string()
}

/// Map a model name (and optional `litellm_provider`) to one of the three
/// vendor buckets ai-usage tracks. Returns `None` for anything irrelevant
/// (Mistral, Cohere, image models, etc.) so we don't waste cache space.
fn classify_vendor(name: &str, provider: Option<&str>) -> Option<&'static str> {
    if let Some(p) = provider {
        let pl = p.to_ascii_lowercase();
        if pl.contains("anthropic") {
            return Some("claude");
        }
        if pl.contains("openai") {
            return Some("codex");
        }
        if pl.contains("gemini") || pl.contains("vertex_ai") || pl.contains("google") {
            // Filter further by name to skip non-chat google models like Imagen.
            if name.starts_with("gemini") || name.contains("gemini") {
                return Some("gemini");
            }
        }
    }

    let lower = name.to_ascii_lowercase();
    if lower.starts_with("claude-") || lower.contains("claude") {
        return Some("claude");
    }
    if lower.starts_with("gemini") || lower.starts_with("models/gemini") {
        return Some("gemini");
    }
    if lower.starts_with("gpt-")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4-")
        || lower.starts_with("codex-")
        || lower == "o1"
        || lower == "o3"
    {
        return Some("codex");
    }
    None
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
        assert_eq!(classify_vendor("gemini-2.5-pro", Some("vertex_ai")), Some("gemini"));
        assert_eq!(classify_vendor("o1", None), Some("codex"));
        assert_eq!(classify_vendor("claude-opus-4-7", None), Some("claude"));
        assert_eq!(classify_vendor("mistral-large", Some("mistral")), None);
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
            .claude
            .get("claude-3-5-sonnet-20241022")
            .expect("claude sonnet present");
        assert!((entry.input - 3.0).abs() < 1e-9);
        assert!((entry.output - 15.0).abs() < 1e-9);
        assert!((entry.cache_input - 0.3).abs() < 1e-9);
        assert!((entry.cache_output - 3.75).abs() < 1e-9);
        assert_eq!(entry.input_above_200k, Some(6.0));
        // Embedding model must be filtered out (mode != chat).
        assert!(tables.codex.is_empty());
        assert!(tables.gemini.is_empty());
    }
}
