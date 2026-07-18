pub mod claude;
pub mod codex;
pub mod gemini;
pub mod kimi;
pub mod omp;

use std::collections::HashMap;

use chrono::{DateTime, Local};

use crate::constants::{AllPricing, ModelPricing};
use crate::data::UsageEntry;
use crate::model_id::{Provider, normalize_reasoning_effort, parse_model_identity};
use crate::time_utils::{
    TokenFractions, distribute_tokens_to_intervals, parse_timestamp, to_interval,
};

pub(crate) fn pricing_provider_for_entry<'a>(tool: &'a str, entry: &'a UsageEntry) -> &'a str {
    if tool != "omp" {
        return tool;
    }

    match parse_model_identity(&entry.model).provider {
        Provider::Claude => "claude",
        Provider::Google => "gemini",
        Provider::Kimi => "kimi",
        Provider::Openai | Provider::Unknown => "codex",
    }
}

fn model_key_for_entry(entry: &UsageEntry, combine_effort: bool) -> String {
    if !combine_effort {
        return entry.model.clone();
    }

    entry
        .effort
        .as_deref()
        .and_then(normalize_reasoning_effort)
        .map(|effort| format!("{} ({effort})", entry.model))
        .unwrap_or_else(|| entry.model.clone())
}

/// Model breakdown row (shared across all tools).
/// Cost component fields are populated during aggregation by applying tiered
/// pricing on each individual entry, then summing. This is the only correct
/// way to handle Claude's 1M-context >200k-tier pricing — applying the tier
/// post-aggregation overstates cost when many entries are below the threshold.
#[derive(Debug, Clone)]
pub struct ModelBreakdownRow {
    pub model: String,
    pub tool: String,
    pub count: i64,
    pub input: i64,
    pub output: i64,
    pub cache_creation: i64,
    pub cache_read: i64,
    pub reasoning: i64,
    pub thinking: i64,
    pub total: i64,
    pub total_with_cache: i64,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read_cost: f64,
    pub cache_creation_cost: f64,
}

/// Token breakdown for a time interval (per model).
#[derive(Debug, Clone, Default)]
pub struct IntervalTokenBreakdown {
    pub input: f64,
    pub output: f64,
    pub cache_creation: f64,
    pub cache_read: f64,
}

/// Time series: interval -> model -> token breakdown
pub type ModelTimeSeries = HashMap<DateTime<Local>, HashMap<String, IntervalTokenBreakdown>>;

/// Time series: interval -> tool -> total tokens
pub type ToolTimeSeries = HashMap<DateTime<Local>, HashMap<String, f64>>;

/// Calculate the per-model breakdown across all entries.
///
/// Rows are aggregated per model (optionally split by effort), `<synthetic>`
/// models are dropped, and the result is sorted by message count descending.
/// Cost is computed per-entry using `ModelPricing::tier_cost` so that the
/// 200k-tier premium for Claude 1M-context models is applied correctly.
pub(crate) fn calculate_model_breakdown_generic(
    usage_data: &[UsageEntry],
    tool: &str,
    combine_effort: bool,
    pricing: &AllPricing,
) -> Vec<ModelBreakdownRow> {
    let mut model_stats: HashMap<String, ModelBreakdownRow> = HashMap::new();

    for entry in usage_data {
        let model_key = model_key_for_entry(entry, combine_effort);

        let row = model_stats
            .entry(model_key.clone())
            .or_insert_with(|| ModelBreakdownRow {
                model: model_key,
                tool: tool.to_string(),
                count: 0,
                input: 0,
                output: 0,
                cache_creation: 0,
                cache_read: 0,
                reasoning: 0,
                thinking: 0,
                total: 0,
                total_with_cache: 0,
                input_cost: 0.0,
                output_cost: 0.0,
                cache_read_cost: 0.0,
                cache_creation_cost: 0.0,
            });

        row.count += 1;
        row.input += entry.usage.input_tokens;
        row.output += entry.usage.output_tokens;
        row.cache_read += entry.usage.cache_read_input_tokens;

        match tool {
            "codex" => {
                row.reasoning += entry.usage.reasoning_output_tokens;
            }
            "gemini" => {
                row.thinking += entry.usage.cache_creation_input_tokens;
                row.cache_creation += entry.usage.cache_creation_input_tokens;
            }
            _ => {
                row.cache_creation += entry.usage.cache_creation_input_tokens;
            }
        }
        if tool != "omp"
            && let Some(costs) = entry.costs
        {
            row.input_cost += costs.input;
            row.output_cost += costs.output;
            row.cache_read_cost += costs.cache_read;
            row.cache_creation_cost += costs.cache_creation;
            continue;
        }

        let pricing_provider = pricing_provider_for_entry(tool, entry);
        let p = pricing.pricing_for_entry(pricing_provider, &entry.model, entry.fast_tier);
        row.input_cost +=
            ModelPricing::tier_cost(entry.usage.input_tokens, p.input, p.input_above_200k);
        row.output_cost +=
            ModelPricing::tier_cost(entry.usage.output_tokens, p.output, p.output_above_200k);
        row.cache_read_cost += ModelPricing::tier_cost(
            entry.usage.cache_read_input_tokens,
            p.cache_input,
            p.cache_input_above_200k,
        );
        row.cache_creation_cost += match (tool, pricing_provider) {
            ("omp", _) => ModelPricing::tier_cost(
                entry.usage.cache_creation_input_tokens,
                p.cache_output,
                p.cache_output_above_200k,
            ),
            (_, "codex") => ModelPricing::tier_cost(
                entry.usage.reasoning_output_tokens,
                p.output,
                p.output_above_200k,
            ),
            (_, "gemini") => ModelPricing::tier_cost(
                entry.usage.cache_creation_input_tokens,
                p.output,
                p.output_above_200k,
            ),
            _ => ModelPricing::tier_cost(
                entry.usage.cache_creation_input_tokens,
                p.cache_output,
                p.cache_output_above_200k,
            ),
        };
    }

    let mut result: Vec<ModelBreakdownRow> = model_stats
        .into_values()
        .filter(|r| !r.model.contains("<synthetic>"))
        .map(|mut r| {
            r.total = r.input + r.output;
            r.total_with_cache = match tool {
                "codex" => r.input + r.output + r.cache_read + r.reasoning,
                "gemini" => r.input + r.output + r.cache_read + r.thinking,
                _ => r.input + r.output + r.cache_creation + r.cache_read,
            };
            r
        })
        .collect();

    result.sort_by(|a, b| b.count.cmp(&a.count));
    result
}

/// Calculate model token breakdown time series with interval distribution.
pub(crate) fn calculate_model_token_breakdown_time_series_generic(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
    combine_effort: bool,
    tool: &str,
) -> ModelTimeSeries {
    let mut time_series: ModelTimeSeries = HashMap::new();

    for entry in usage_data {
        if entry.timestamp.is_empty() {
            continue;
        }

        let model_key = model_key_for_entry(entry, combine_effort);

        let tokens = TokenFractions {
            input: entry.usage.input_tokens as f64,
            output: entry.usage.output_tokens as f64,
            cache_creation: match tool {
                "codex" => entry.usage.reasoning_output_tokens as f64,
                _ => entry.usage.cache_creation_input_tokens as f64,
            },
            cache_read: entry.usage.cache_read_input_tokens as f64,
        };

        let distributed = distribute_tokens_to_intervals(
            &entry.session_start_time,
            &entry.session_end_time,
            &tokens,
            interval_minutes,
        );

        if !distributed.is_empty() {
            for (interval_time, fraction) in distributed {
                let model_map = time_series.entry(interval_time).or_default();
                let breakdown = model_map.entry(model_key.clone()).or_default();
                breakdown.input += fraction.input;
                breakdown.output += fraction.output;
                breakdown.cache_creation += fraction.cache_creation;
                breakdown.cache_read += fraction.cache_read;
            }
        } else {
            // Fallback: use timestamp-based bucketing
            let timestamp_local = entry
                .parsed_timestamp
                .or_else(|| parse_timestamp(&entry.timestamp));

            if let Some(ts) = timestamp_local {
                let interval_time = to_interval(&ts, interval_minutes);
                let model_map = time_series.entry(interval_time).or_default();
                let breakdown = model_map.entry(model_key.clone()).or_default();
                breakdown.input += tokens.input;
                breakdown.output += tokens.output;
                breakdown.cache_creation += tokens.cache_creation;
                breakdown.cache_read += tokens.cache_read;
            }
        }
    }

    time_series
}

// Public wrappers for each tool
pub use claude::{
    calculate_model_breakdown as calculate_claude_model_breakdown,
    calculate_model_token_breakdown_time_series as calculate_claude_model_token_breakdown_time_series,
};
pub use codex::{
    calculate_codex_model_breakdown, calculate_codex_model_token_breakdown_time_series,
};
pub use gemini::{
    calculate_gemini_model_breakdown, calculate_gemini_model_token_breakdown_time_series,
};
pub use kimi::{calculate_kimi_model_breakdown, calculate_kimi_model_token_breakdown_time_series};
pub use omp::{calculate_omp_model_breakdown, calculate_omp_model_token_breakdown_time_series};

// Re-export the generic functions for use by tool modules
pub(crate) use self::calculate_model_breakdown_generic as _calc_breakdown;
pub(crate) use self::calculate_model_token_breakdown_time_series_generic as _calc_time_series;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{TokenUsage, UNKNOWN_FAST_TIER, UsageCost};

    fn usage_entry(model: &str, effort: Option<&str>, costs: Option<UsageCost>) -> UsageEntry {
        UsageEntry {
            host_id: None,
            timestamp: "2026-06-15T12:00:00Z".to_string(),
            parsed_timestamp: None,
            session_start_time: String::new(),
            session_end_time: String::new(),
            model: model.to_string(),
            effort: effort.map(str::to_string),
            fast_tier: UNKNOWN_FAST_TIER,
            usage: TokenUsage {
                input_tokens: 1_000_000,
                output_tokens: 100_000,
                cache_read_input_tokens: 500_000,
                cache_creation_input_tokens: 20_000,
                reasoning_output_tokens: 0,
            },
            costs,
        }
    }

    #[test]
    fn omp_breakdown_ignores_endpoint_provider_in_effort() {
        let pricing = AllPricing::load_raw().finalize();
        let entry = usage_entry("gpt-5", Some("rust-cat"), Some(UsageCost::default()));

        let rows = calculate_model_breakdown_generic(&[entry], "omp", true, &pricing);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "gpt-5");
        assert!(rows[0].input_cost > 0.0);
        assert!(rows[0].output_cost > 0.0);
    }

    #[test]
    fn codex_breakdown_groups_same_model_across_efforts() {
        let pricing = AllPricing::load_raw().finalize();
        let high = usage_entry("gpt-5", Some("high"), None);
        let max = usage_entry("gpt-5", Some("max"), None);

        let rows = codex::calculate_codex_model_breakdown(&[high, max], &pricing);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "gpt-5");
        assert_eq!(rows[0].count, 2);
    }

    #[test]
    fn omp_breakdown_groups_same_model_across_efforts() {
        let pricing = AllPricing::load_raw().finalize();
        let xhigh = usage_entry("gpt-5", Some("xhigh"), None);
        let max = usage_entry("gpt-5", Some("max"), None);

        let rows = omp::calculate_omp_model_breakdown(&[xhigh, max], &pricing);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "gpt-5");
        assert_eq!(rows[0].count, 2);
    }

    #[test]
    fn model_time_series_groups_same_model_across_efforts() {
        let high = usage_entry("gpt-5", Some("high"), None);
        let max = usage_entry("gpt-5", Some("max"), None);

        let series = codex::calculate_codex_model_token_breakdown_time_series(&[high, max], 60);
        let models: Vec<&String> = series.values().flat_map(|models| models.keys()).collect();

        assert!(models.iter().any(|model| model.as_str() == "gpt-5"));
        assert!(!models.iter().any(|model| model.contains("high")));
        assert!(!models.iter().any(|model| model.contains("max")));
    }

    #[test]
    fn omp_pricing_provider_comes_from_model_id() {
        let claude = usage_entry("claude-sonnet-4-5-20250929", Some("rust-cat"), None);
        let google = usage_entry("gemini-2.5-pro", Some("rust-cat"), None);
        let open = usage_entry("gpt-5", Some("rust-cat"), None);
        let kimi = usage_entry("kimi-k2.5", Some("rust-cat"), None);

        assert_eq!(pricing_provider_for_entry("omp", &claude), "claude");
        assert_eq!(pricing_provider_for_entry("omp", &google), "gemini");
        assert_eq!(pricing_provider_for_entry("omp", &open), "codex");
        assert_eq!(pricing_provider_for_entry("omp", &kimi), "kimi");
    }
}
