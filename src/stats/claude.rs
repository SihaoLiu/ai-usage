use crate::data::UsageEntry;
use crate::stats::{ModelBreakdownRow, ModelTimeSeries, _calc_breakdown, _calc_time_series};

pub fn calculate_model_breakdown(usage_data: &[UsageEntry]) -> Vec<ModelBreakdownRow> {
    _calc_breakdown(usage_data, "claude", false)
}

pub fn calculate_model_token_breakdown_time_series(
    usage_data: &[UsageEntry],
    interval_minutes: i64,
) -> ModelTimeSeries {
    _calc_time_series(usage_data, interval_minutes, false, "claude")
}
