use crate::constants::AllPricing;
use crate::data::UsageEntry;
use crate::stats::{_calc_breakdown, _calc_time_series, ModelBreakdownRow, ModelTimeSeries};

pub fn calculate_dsh_model_breakdown(
    usage_data: &[UsageEntry],
    pricing: &AllPricing,
) -> Vec<ModelBreakdownRow> {
    _calc_breakdown(usage_data, "dsh", false, pricing)
}

pub fn calculate_dsh_model_token_breakdown_time_series(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
) -> ModelTimeSeries {
    _calc_time_series(usage_data, interval_minutes, false, "dsh")
}
