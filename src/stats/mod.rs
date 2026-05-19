pub mod claude;
pub mod codex;
pub mod gemini;

use std::collections::HashMap;

use chrono::{DateTime, Local};

use crate::constants::{AllPricing, ModelPricing};
use crate::data::UsageEntry;
use crate::time_utils::{
    TokenFractions, distribute_tokens_to_intervals, parse_timestamp, to_interval,
};

/// Model breakdown row (shared across all vendors).
///
/// Cost component fields are populated during aggregation by applying tiered
/// pricing on each individual entry, then summing. This is the only correct
/// way to handle Claude's 1M-context >200k-tier pricing — applying the tier
/// post-aggregation overstates cost when many entries are below the threshold.
#[derive(Debug, Clone)]
pub struct ModelBreakdownRow {
    pub model: String,
    pub vendor: String,
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

/// Time series: interval -> vendor -> total tokens
pub type VendorTimeSeries = HashMap<DateTime<Local>, HashMap<String, f64>>;

/// Calculate model breakdown with 1% threshold filtering.
///
/// Cost is computed per-entry using `ModelPricing::tier_cost` so that the
/// 200k-tier premium for Claude 1M-context models is applied correctly.
pub(crate) fn calculate_model_breakdown_generic(
    usage_data: &[UsageEntry],
    vendor: &str,
    combine_effort: bool,
    pricing: &AllPricing,
) -> Vec<ModelBreakdownRow> {
    let mut model_stats: HashMap<String, ModelBreakdownRow> = HashMap::new();

    for entry in usage_data {
        let model_key = if combine_effort {
            if let Some(ref effort) = entry.effort {
                format!("{} ({})", entry.model, effort)
            } else {
                entry.model.clone()
            }
        } else {
            entry.model.clone()
        };

        let row = model_stats
            .entry(model_key.clone())
            .or_insert_with(|| ModelBreakdownRow {
                model: model_key,
                vendor: vendor.to_string(),
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

        match vendor {
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

        // Apply tiered pricing per-entry. Use entry.model (without the effort
        // suffix) so the lookup matches pricing.json keys and the stored
        // fast tier snapshot can be applied per record.
        let p = pricing.pricing_for_entry(vendor, &entry.model, entry.fast_tier);
        row.input_cost +=
            ModelPricing::tier_cost(entry.usage.input_tokens, p.input, p.input_above_200k);
        row.output_cost +=
            ModelPricing::tier_cost(entry.usage.output_tokens, p.output, p.output_above_200k);
        row.cache_read_cost += ModelPricing::tier_cost(
            entry.usage.cache_read_input_tokens,
            p.cache_input,
            p.cache_input_above_200k,
        );
        row.cache_creation_cost += match vendor {
            // Codex bills reasoning at the output rate; tier rules track output.
            "codex" => ModelPricing::tier_cost(
                entry.usage.reasoning_output_tokens,
                p.output,
                p.output_above_200k,
            ),
            // Gemini's "thoughts" land in cache_creation_input_tokens and bill
            // at the output rate.
            "gemini" => ModelPricing::tier_cost(
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
            r.total_with_cache = match vendor {
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
    vendor: &str,
) -> ModelTimeSeries {
    let mut time_series: ModelTimeSeries = HashMap::new();

    for entry in usage_data {
        if entry.timestamp.is_empty() {
            continue;
        }

        let model_key = if combine_effort {
            if let Some(ref effort) = entry.effort {
                format!("{} ({})", entry.model, effort)
            } else {
                entry.model.clone()
            }
        } else {
            entry.model.clone()
        };

        let tokens = TokenFractions {
            input: entry.usage.input_tokens as f64,
            output: entry.usage.output_tokens as f64,
            cache_creation: match vendor {
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

// Public wrappers for each vendor
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

// Re-export the generic functions for use by vendor modules
pub(crate) use self::calculate_model_breakdown_generic as _calc_breakdown;
pub(crate) use self::calculate_model_token_breakdown_time_series_generic as _calc_time_series;
