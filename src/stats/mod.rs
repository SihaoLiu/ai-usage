pub mod claude;
pub mod codex;
pub mod gemini;
pub mod kimi;
pub mod omp;

use std::collections::HashMap;
use std::sync::OnceLock;
use std::thread;

use chrono::{DateTime, Local};
use rayon::prelude::*;

use crate::constants::{AllPricing, ModelPricing};
use crate::data::UsageEntry;
use crate::model_id::normalize_reasoning_effort;
use crate::time_utils::{
    TokenFractions, distribute_tokens_to_intervals, parse_timestamp, to_interval,
};

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

type EntryPricing<'a> = HashMap<&'a str, HashMap<i8, ModelPricing>>;

fn resolve_entry_pricing<'a>(
    usage_data: &'a [UsageEntry],
    tool: &str,
    pricing: &AllPricing,
) -> EntryPricing<'a> {
    let mut resolved = HashMap::new();
    for entry in usage_data
        .iter()
        .filter(|entry| tool == "omp" || entry.costs.is_none())
    {
        let tiers = resolved
            .entry(entry.model.as_str())
            .or_insert_with(HashMap::new);
        tiers.entry(entry.fast_tier).or_insert_with(|| {
            pricing
                .pricing_for_entry(tool, &entry.model, entry.fast_tier)
                .into_owned()
        });
    }
    resolved
}

/// Model breakdown row (shared across all tools).
/// Cost component fields are populated during aggregation by applying tiered
/// pricing on each individual entry, then summing. This is the only correct
/// way to handle Claude's 1M-context >200k-tier pricing — applying the tier
/// post-aggregation overstates cost when many entries are below the threshold.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Default, PartialEq)]
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

/// Chunk size for parallel aggregation over usage entries. Large enough that
/// per-chunk HashMap merge cost is negligible, small enough to spread a
/// million-entry scan across every core.
pub(crate) const PAR_CHUNK: usize = 16_384;

fn dashboard_aggregation_parallelism(available: usize) -> usize {
    (available / 2).clamp(1, 4)
}

fn dashboard_aggregation_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let available = thread::available_parallelism().map_or(1, usize::from);
        rayon::ThreadPoolBuilder::new()
            .num_threads(dashboard_aggregation_parallelism(available))
            .thread_name(|index| format!("usage-dashboard-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

fn run_dashboard_aggregation<T: Send>(operation: impl FnOnce() -> T + Send) -> T {
    match dashboard_aggregation_pool() {
        Some(pool) => pool.install(operation),
        None => operation(),
    }
}

fn accumulate_breakdown_entry(
    model_stats: &mut HashMap<String, ModelBreakdownRow>,
    entry: &UsageEntry,
    tool: &str,
    combine_effort: bool,
    entry_pricing: &EntryPricing<'_>,
) {
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
        return;
    }

    let p = &entry_pricing[entry.model.as_str()][&entry.fast_tier];
    row.input_cost +=
        ModelPricing::tier_cost(entry.usage.input_tokens, p.input, p.input_above_200k);
    row.output_cost +=
        ModelPricing::tier_cost(entry.usage.output_tokens, p.output, p.output_above_200k);
    row.cache_read_cost += ModelPricing::tier_cost(
        entry.usage.cache_read_input_tokens,
        p.cache_input,
        p.cache_input_above_200k,
    );
    row.cache_creation_cost += match tool {
        "omp" => ModelPricing::tier_cost(
            entry.usage.cache_creation_input_tokens,
            p.cache_output,
            p.cache_output_above_200k,
        ),
        "codex" => ModelPricing::tier_cost(
            entry.usage.reasoning_output_tokens,
            p.output,
            p.output_above_200k,
        ),
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

fn merge_breakdown_rows(a: &mut ModelBreakdownRow, b: ModelBreakdownRow) {
    a.count += b.count;
    a.input += b.input;
    a.output += b.output;
    a.cache_creation += b.cache_creation;
    a.cache_read += b.cache_read;
    a.reasoning += b.reasoning;
    a.thinking += b.thinking;
    a.input_cost += b.input_cost;
    a.output_cost += b.output_cost;
    a.cache_read_cost += b.cache_read_cost;
    a.cache_creation_cost += b.cache_creation_cost;
}

fn merge_breakdown_maps(
    mut a: HashMap<String, ModelBreakdownRow>,
    b: HashMap<String, ModelBreakdownRow>,
) -> HashMap<String, ModelBreakdownRow> {
    for (key, row) in b {
        match a.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                merge_breakdown_rows(slot.get_mut(), row);
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(row);
            }
        }
    }
    a
}

fn strategy_tokens(tool: &str, cache_creation: i64, reasoning: i64) -> i64 {
    match tool {
        "codex" => reasoning,
        _ => cache_creation,
    }
}

pub(crate) fn entry_total_with_cache(entry: &UsageEntry, tool: &str) -> u128 {
    [
        entry.usage.input_tokens,
        entry.usage.output_tokens,
        entry.usage.cache_read_input_tokens,
        strategy_tokens(
            tool,
            entry.usage.cache_creation_input_tokens,
            entry.usage.reasoning_output_tokens,
        ),
    ]
    .into_iter()
    .map(|tokens| tokens.max(0) as u128)
    .sum()
}

fn finish_model_breakdown(
    model_stats: HashMap<String, ModelBreakdownRow>,
    tool: &str,
) -> Vec<ModelBreakdownRow> {
    let mut result: Vec<ModelBreakdownRow> = model_stats
        .into_values()
        .filter(|row| !row.model.contains("<synthetic>"))
        .map(|mut row| {
            row.total = row.input + row.output;
            row.total_with_cache = row.input
                + row.output
                + row.cache_read
                + strategy_tokens(tool, row.cache_creation, row.reasoning);
            row
        })
        .collect();
    result.sort_by(|a, b| b.count.cmp(&a.count));
    result
}

/// Calculate the per-model breakdown across all entries.
///
/// Rows are aggregated per model (optionally split by effort), `<synthetic>`
/// models are dropped, and the result is sorted by message count descending.
/// Cost is computed per-entry using `ModelPricing::tier_cost` so that the
/// 200k-tier premium for Claude 1M-context models is applied correctly.
/// The scan is chunk-parallel: each chunk folds into a local map, maps merge
/// pairwise.
pub(crate) fn calculate_model_breakdown_generic(
    usage_data: &[UsageEntry],
    tool: &str,
    combine_effort: bool,
    pricing: &AllPricing,
) -> Vec<ModelBreakdownRow> {
    let entry_pricing = resolve_entry_pricing(usage_data, tool, pricing);
    let model_stats: HashMap<String, ModelBreakdownRow> = usage_data
        .par_chunks(PAR_CHUNK)
        .fold(HashMap::new, |mut local, chunk| {
            for entry in chunk {
                accumulate_breakdown_entry(&mut local, entry, tool, combine_effort, &entry_pricing);
            }
            local
        })
        .reduce(HashMap::new, merge_breakdown_maps);

    finish_model_breakdown(model_stats, tool)
}

fn accumulate_time_series_entry(
    time_series: &mut ModelTimeSeries,
    entry: &UsageEntry,
    interval_minutes: i64,
    combine_effort: bool,
    tool: &str,
) {
    if entry.timestamp.is_empty() {
        return;
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

fn merge_time_series(mut a: ModelTimeSeries, b: ModelTimeSeries) -> ModelTimeSeries {
    for (interval_time, models) in b {
        let target = a.entry(interval_time).or_default();
        for (model, breakdown) in models {
            let slot = target.entry(model).or_default();
            slot.input += breakdown.input;
            slot.output += breakdown.output;
            slot.cache_creation += breakdown.cache_creation;
            slot.cache_read += breakdown.cache_read;
        }
    }
    a
}

fn accumulate_tool_time_series_entry(
    time_series: &mut ToolTimeSeries,
    entry: &UsageEntry,
    interval_minutes: i64,
    tool: &str,
    tool_label: &str,
) {
    if entry.timestamp.is_empty() {
        return;
    }
    let total = match tool {
        "codex" => {
            entry.usage.input_tokens
                + entry.usage.output_tokens
                + entry.usage.cache_read_input_tokens
                + entry.usage.reasoning_output_tokens
        }
        _ => {
            entry.usage.input_tokens
                + entry.usage.output_tokens
                + entry.usage.cache_read_input_tokens
                + entry.usage.cache_creation_input_tokens
        }
    } as f64;
    let timestamp = entry
        .parsed_timestamp
        .or_else(|| parse_timestamp(&entry.timestamp));
    if let Some(timestamp) = timestamp {
        let interval_time = to_interval(&timestamp, interval_minutes);
        *time_series
            .entry(interval_time)
            .or_default()
            .entry(tool_label.to_string())
            .or_insert(0.0) += total;
    }
}

fn merge_tool_time_series(mut a: ToolTimeSeries, b: ToolTimeSeries) -> ToolTimeSeries {
    for (interval_time, tools) in b {
        let target = a.entry(interval_time).or_default();
        for (label, total) in tools {
            *target.entry(label).or_insert(0.0) += total;
        }
    }
    a
}

pub(crate) fn calculate_model_dashboard_data(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
    tool: &str,
    pricing: &AllPricing,
) -> (Vec<ModelBreakdownRow>, ModelTimeSeries) {
    let entry_pricing = resolve_entry_pricing(usage_data, tool, pricing);
    let (model_stats, time_series) = run_dashboard_aggregation(|| {
        usage_data
            .par_chunks(PAR_CHUNK)
            .fold(
                || (HashMap::new(), HashMap::new()),
                |(mut model_stats, mut time_series), chunk| {
                    for entry in chunk {
                        accumulate_breakdown_entry(
                            &mut model_stats,
                            entry,
                            tool,
                            false,
                            &entry_pricing,
                        );
                        accumulate_time_series_entry(
                            &mut time_series,
                            entry,
                            interval_minutes,
                            false,
                            tool,
                        );
                    }
                    (model_stats, time_series)
                },
            )
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(model_stats_a, time_series_a), (model_stats_b, time_series_b)| {
                    (
                        merge_breakdown_maps(model_stats_a, model_stats_b),
                        merge_time_series(time_series_a, time_series_b),
                    )
                },
            )
    });
    (finish_model_breakdown(model_stats, tool), time_series)
}

pub(crate) fn calculate_comparison_dashboard_data(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
    tool: &str,
    tool_label: &str,
    pricing: &AllPricing,
) -> (Vec<ModelBreakdownRow>, ToolTimeSeries) {
    let entry_pricing = resolve_entry_pricing(usage_data, tool, pricing);
    let (model_stats, time_series) = run_dashboard_aggregation(|| {
        usage_data
            .par_chunks(PAR_CHUNK)
            .fold(
                || (HashMap::new(), HashMap::new()),
                |(mut model_stats, mut time_series), chunk| {
                    for entry in chunk {
                        accumulate_breakdown_entry(
                            &mut model_stats,
                            entry,
                            tool,
                            false,
                            &entry_pricing,
                        );
                        accumulate_tool_time_series_entry(
                            &mut time_series,
                            entry,
                            interval_minutes,
                            tool,
                            tool_label,
                        );
                    }
                    (model_stats, time_series)
                },
            )
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(model_stats_a, time_series_a), (model_stats_b, time_series_b)| {
                    (
                        merge_breakdown_maps(model_stats_a, model_stats_b),
                        merge_tool_time_series(time_series_a, time_series_b),
                    )
                },
            )
    });
    (finish_model_breakdown(model_stats, tool), time_series)
}

/// Calculate model token breakdown time series with interval distribution.
/// Chunk-parallel: the per-entry session-time parsing dominates this scan.
pub(crate) fn calculate_model_token_breakdown_time_series_generic(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
    combine_effort: bool,
    tool: &str,
) -> ModelTimeSeries {
    usage_data
        .par_chunks(PAR_CHUNK)
        .fold(HashMap::new, |mut local, chunk| {
            for entry in chunk {
                accumulate_time_series_entry(
                    &mut local,
                    entry,
                    interval_minutes,
                    combine_effort,
                    tool,
                );
            }
            local
        })
        .reduce(HashMap::new, merge_time_series)
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
            session_id: None,
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
    fn dashboard_aggregation_leaves_cpu_capacity_for_input_and_rendering() {
        assert_eq!(dashboard_aggregation_parallelism(1), 1);
        assert_eq!(dashboard_aggregation_parallelism(2), 1);
        assert_eq!(dashboard_aggregation_parallelism(4), 2);
        assert_eq!(dashboard_aggregation_parallelism(8), 4);
        assert_eq!(dashboard_aggregation_parallelism(64), 4);
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
    fn fused_dashboard_scan_matches_separate_aggregations() {
        let pricing = AllPricing::load_raw().finalize();
        let entries = [
            usage_entry("gpt-5", Some("high"), None),
            usage_entry("gpt-5", Some("max"), None),
        ];
        let expected_rows = calculate_model_breakdown_generic(&entries, "codex", false, &pricing);
        let expected_series =
            calculate_model_token_breakdown_time_series_generic(&entries, 60, false, "codex");

        let (rows, series) = calculate_model_dashboard_data(&entries, 60, "codex", &pricing);

        assert_eq!(rows, expected_rows);
        assert_eq!(series, expected_series);

        let (rows, comparison) =
            calculate_comparison_dashboard_data(&entries, 60, "codex", "Codex", &pricing);
        assert_eq!(rows, expected_rows);
        let timestamp = parse_timestamp("2026-06-15T12:00:00Z").expect("timestamp");
        let bucket = to_interval(&timestamp, 60);
        assert_eq!(comparison[&bucket]["Codex"], 3_200_000.0);
    }

    #[test]
    fn entry_pricing_is_resolved_once_per_model_tier() {
        let pricing = AllPricing::load_raw().finalize();
        let standard = usage_entry("gpt-5", None, None);
        let repeated = usage_entry("gpt-5", Some("high"), None);
        let mut fast = usage_entry("gpt-5", None, None);
        fast.fast_tier = 1;
        let persisted = usage_entry("cached-model", None, Some(UsageCost::default()));
        let entries = [standard, repeated, fast, persisted];

        let resolved = resolve_entry_pricing(&entries, "codex", &pricing);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved["gpt-5"].len(), 2);
        assert!(!resolved.contains_key("cached-model"));
    }
}
