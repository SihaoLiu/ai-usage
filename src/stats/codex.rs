use crate::constants::AllPricing;
use crate::data::UsageEntry;
use crate::stats::{_calc_breakdown, _calc_time_series, ModelBreakdownRow, ModelTimeSeries};

pub fn calculate_codex_model_breakdown(
    usage_data: &[UsageEntry],
    pricing: &AllPricing,
) -> Vec<ModelBreakdownRow> {
    _calc_breakdown(usage_data, "codex", true, pricing)
}

pub fn calculate_codex_model_token_breakdown_time_series(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
) -> ModelTimeSeries {
    _calc_time_series(usage_data, interval_minutes, true, "codex")
}
