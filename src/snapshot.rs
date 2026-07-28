use std::borrow::Cow;
use std::io::{self, Write};

use chrono::{DateTime, Duration, Local, NaiveDate};
use serde::Serialize;

use crate::constants::AllPricing;
use crate::raw_data::{AllToolData, RawDataCache, RawDataRange, filter_all_tool_data_borrowed};
use crate::stats::ModelBreakdownRow;
use crate::table_view::{DisplayRow, TableView, build_table, table_totals};
use crate::time_utils::TimeWindow;
use crate::tool::Tool;

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct TokenMetrics {
    pub(crate) cache_hit: i64,
    pub(crate) prefill: i64,
    pub(crate) decoding: i64,
    pub(crate) total: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct UsageMetrics {
    pub(crate) request_count: i64,
    pub(crate) tokens: TokenMetrics,
    pub(crate) estimated_api_cost_usd: f64,
    pub(crate) cache_share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SnapshotRange {
    pub(crate) start: String,
    pub(crate) end: String,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct DailyUsage {
    pub(crate) date: NaiveDate,
    pub(crate) usage: UsageMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct TopModel {
    pub(crate) model_id: String,
    pub(crate) display_label: String,
    pub(crate) vendor: String,
    pub(crate) harnesses: Vec<String>,
    pub(crate) tokens: i64,
    pub(crate) estimated_api_cost_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SnapshotDocument {
    pub(crate) schema_version: u32,
    pub(crate) generated_at: String,
    pub(crate) timezone: String,
    pub(crate) range: SnapshotRange,
    pub(crate) window: UsageMetrics,
    pub(crate) daily: Vec<DailyUsage>,
    pub(crate) top_models: Vec<TopModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnapshotQuery {
    pub(crate) days: i64,
    pub(crate) tool: Tool,
    pub(crate) session_id: Option<String>,
}

pub(crate) fn aggregate_metrics(rows: &[ModelBreakdownRow]) -> UsageMetrics {
    let totals = table_totals(rows);
    let token_total = totals.tokens();
    UsageMetrics {
        request_count: totals.count,
        tokens: TokenMetrics {
            cache_hit: totals.cache_hit,
            prefill: totals.prefill,
            decoding: totals.decoding,
            total: token_total,
        },
        estimated_api_cost_usd: totals.cost(),
        cache_share: if token_total == 0 {
            0.0
        } else {
            totals.cache_hit as f64 / token_total as f64
        },
    }
}

pub(crate) fn calendar_dates(today: NaiveDate, days: i64) -> Vec<NaiveDate> {
    (0..days.max(1))
        .rev()
        .map(|offset| today - Duration::days(offset))
        .collect()
}

pub(crate) fn build_document(
    generated_at: DateTime<Local>,
    range_start: DateTime<Local>,
    complete: bool,
    window_rows: &[ModelBreakdownRow],
    daily_rows: Vec<(NaiveDate, Vec<ModelBreakdownRow>)>,
) -> SnapshotDocument {
    let top_models = build_table(window_rows, TableView::Model)
        .into_iter()
        .filter_map(|row| match row {
            DisplayRow::Data(row) => Some(TopModel {
                model_id: row.model_raw,
                display_label: row.model_label,
                vendor: row.vendor.display_name().to_string(),
                harnesses: row
                    .harness_short
                    .split(',')
                    .map(ToOwned::to_owned)
                    .collect(),
                tokens: row.metrics.tokens(),
                estimated_api_cost_usd: row.metrics.cost(),
            }),
            DisplayRow::GroupHeader { .. } | DisplayRow::Subtotal { .. } => None,
        })
        .take(5)
        .collect();
    let daily = daily_rows
        .into_iter()
        .map(|(date, rows)| DailyUsage {
            date,
            usage: aggregate_metrics(&rows),
        })
        .collect();

    SnapshotDocument {
        schema_version: SCHEMA_VERSION,
        generated_at: generated_at.to_rfc3339(),
        timezone: generated_at.format("%Z").to_string(),
        range: SnapshotRange {
            start: range_start.to_rfc3339(),
            end: generated_at.to_rfc3339(),
            complete,
        },
        window: aggregate_metrics(window_rows),
        daily,
        top_models,
    }
}

fn selected_tool(mut data: AllToolData<'_>, tool: Tool) -> AllToolData<'_> {
    if !matches!(tool, Tool::All | Tool::Claude) {
        data.claude = Cow::Borrowed(&[]);
    }
    if !matches!(tool, Tool::All | Tool::Codex) {
        data.codex = Cow::Borrowed(&[]);
    }
    if !matches!(tool, Tool::All | Tool::Gemini) {
        data.gemini = Cow::Borrowed(&[]);
    }
    if !matches!(tool, Tool::All | Tool::Kimi) {
        data.kimi = Cow::Borrowed(&[]);
    }
    if !matches!(tool, Tool::All | Tool::Omp) {
        data.omp = Cow::Borrowed(&[]);
    }
    data
}

fn day_window(date: NaiveDate, now: DateTime<Local>) -> TimeWindow {
    let full_day = TimeWindow::from_date(&date.to_string()).expect("valid calendar date");
    let (start, end) = full_day.bounds(now);
    TimeWindow::ExplicitRange {
        start,
        end: end.min(now),
        projection_days: 1.0,
        page_step: Duration::days(1),
    }
}

pub(crate) fn required_range(query: &SnapshotQuery, now: DateTime<Local>) -> RawDataRange {
    let dates = calendar_dates(now.date_naive(), query.days);
    let (start, _) = day_window(dates[0], now).bounds(now);
    RawDataRange::from_bounds(start, now)
}

pub(crate) fn build_from_cache(
    cache: &RawDataCache,
    query: &SnapshotQuery,
    pricing: &AllPricing,
    now: DateTime<Local>,
) -> SnapshotDocument {
    let dates = calendar_dates(now.date_naive(), query.days);
    let range = required_range(query, now);
    let range_start = range.start();
    let window = TimeWindow::ExplicitRange {
        start: range_start,
        end: now,
        projection_days: query.days as f64,
        page_step: Duration::days(query.days),
    };
    let window_data = selected_tool(
        filter_all_tool_data_borrowed(cache, &window, query.session_id.as_deref(), now),
        query.tool,
    );
    let window_rows = crate::calculate_all_model_breakdown(&window_data, pricing);
    let daily_rows = dates
        .into_iter()
        .map(|date| {
            let day = day_window(date, now);
            let data = selected_tool(
                filter_all_tool_data_borrowed(cache, &day, query.session_id.as_deref(), now),
                query.tool,
            );
            (date, crate::calculate_all_model_breakdown(&data, pricing))
        })
        .collect();
    let complete = cache.has_source_data && cache.range.covers(range);

    build_document(now, range_start, complete, &window_rows, daily_rows)
}

pub(crate) fn write_json(mut writer: impl Write, document: &SnapshotDocument) -> io::Result<()> {
    serde_json::to_writer_pretty(&mut writer, document).map_err(io::Error::other)?;
    writeln!(writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::constants::AllPricing;
    use crate::data::{TokenUsage, UsageCost, UsageEntry};
    use crate::raw_data::{RawDataCache, RawDataRange};
    use crate::stats::ModelBreakdownRow;
    use crate::tool::Tool;
    use chrono::{Local, NaiveDate, TimeZone};

    fn row(tool: &str) -> ModelBreakdownRow {
        ModelBreakdownRow {
            model: format!("{tool}-model"),
            tool: tool.to_string(),
            count: 1,
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
        }
    }

    fn usage_entry(timestamp: DateTime<Local>, model: &str, input_tokens: i64) -> UsageEntry {
        UsageEntry {
            host_id: None,
            session_id: None,
            timestamp: timestamp.to_rfc3339(),
            parsed_timestamp: Some(timestamp),
            session_start_time: timestamp.to_rfc3339(),
            session_end_time: timestamp.to_rfc3339(),
            model: model.to_string(),
            effort: None,
            fast_tier: crate::data::UNKNOWN_FAST_TIER,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 10,
                cache_read_input_tokens: 5,
                cache_creation_input_tokens: 0,
                reasoning_output_tokens: 2,
            },
            costs: Some(UsageCost {
                input: 1.0,
                output: 0.5,
                cache_read: 0.1,
                cache_creation: 0.0,
            }),
        }
    }

    #[test]
    fn aggregate_metrics_reuses_inference_strategy_semantics() {
        let mut codex = row("codex");
        codex.count = 2;
        codex.input = 100;
        codex.output = 20;
        codex.cache_read = 50;
        codex.reasoning = 30;
        codex.input_cost = 1.0;
        codex.output_cost = 0.4;
        codex.cache_read_cost = 0.1;
        codex.cache_creation_cost = 0.2;

        let mut claude = row("claude");
        claude.count = 3;
        claude.input = 10;
        claude.output = 5;
        claude.cache_creation = 7;
        claude.cache_read = 3;
        claude.input_cost = 0.2;
        claude.output_cost = 0.3;
        claude.cache_read_cost = 0.05;
        claude.cache_creation_cost = 0.07;

        let metrics = aggregate_metrics(&[codex, claude]);

        assert_eq!(metrics.request_count, 5);
        assert_eq!(metrics.tokens.cache_hit, 53);
        assert_eq!(metrics.tokens.prefill, 117);
        assert_eq!(metrics.tokens.decoding, 55);
        assert_eq!(metrics.tokens.total, 225);
        assert!((metrics.estimated_api_cost_usd - 2.32).abs() < 1e-9);
        assert!((metrics.cache_share - 53.0 / 225.0).abs() < 1e-9);
    }

    #[test]
    fn calendar_dates_are_contiguous_and_include_today() {
        let today = NaiveDate::from_ymd_opt(2026, 7, 28).unwrap();

        let dates = calendar_dates(today, 7);

        assert_eq!(dates.len(), 7);
        assert_eq!(dates[0], NaiveDate::from_ymd_opt(2026, 7, 22).unwrap());
        assert_eq!(dates[6], today);
        for pair in dates.windows(2) {
            assert_eq!(pair[1].signed_duration_since(pair[0]).num_days(), 1);
        }
    }

    #[test]
    fn snapshot_json_has_a_versioned_stable_shape() {
        let generated_at = Local.with_ymd_and_hms(2026, 7, 28, 14, 37, 0).unwrap();
        let range_start = Local.with_ymd_and_hms(2026, 7, 22, 0, 0, 0).unwrap();
        let mut model = row("codex");
        model.model = "gpt-5.6-codex".to_string();
        model.count = 4;
        model.input = 100;
        model.output = 20;
        model.cache_read = 30;
        model.reasoning = 10;
        model.input_cost = 1.25;

        let document = build_document(
            generated_at,
            range_start,
            true,
            std::slice::from_ref(&model),
            vec![
                (NaiveDate::from_ymd_opt(2026, 7, 27).unwrap(), Vec::new()),
                (
                    NaiveDate::from_ymd_opt(2026, 7, 28).unwrap(),
                    vec![model.clone()],
                ),
            ],
        );
        let json = serde_json::to_value(document).unwrap();

        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["range"]["complete"], true);
        assert_eq!(json["window"]["tokens"]["total"], 160);
        assert_eq!(json["daily"].as_array().unwrap().len(), 2);
        assert_eq!(json["daily"][0]["usage"]["tokens"]["total"], 0);
        assert_eq!(json["top_models"][0]["model_id"], "gpt-5.6-codex");
        assert_eq!(json["top_models"][0]["vendor"], "OpenAI");
        assert_eq!(json["top_models"][0]["harnesses"][0], "Cdx");
        assert!(json.get("generated_at").is_some());
        assert!(json.get("timezone").is_some());
    }

    #[test]
    fn cache_snapshot_filters_tools_and_preserves_empty_days() {
        let now = Local.with_ymd_and_hms(2026, 7, 28, 14, 37, 0).unwrap();
        let cache = RawDataCache {
            claude: vec![usage_entry(now, "claude-opus-4-1", 1000)],
            codex: vec![usage_entry(now, "gpt-5.6-codex", 100)],
            gemini: Vec::new(),
            kimi: Vec::new(),
            omp: Vec::new(),
            range: RawDataRange::from_bounds(now - Duration::days(3), now + Duration::days(1)),
            has_source_data: true,
            local_host_id: None,
            local_record_keys: HashMap::new(),
            persistent_generation: String::new(),
            local_session_metadata_current: true,
        };
        let query = SnapshotQuery {
            days: 2,
            tool: Tool::Codex,
            session_id: None,
        };
        let pricing = AllPricing::load_raw().finalize();

        let document = build_from_cache(&cache, &query, &pricing, now);

        assert_eq!(document.window.request_count, 1);
        assert_eq!(document.window.tokens.total, 117);
        assert_eq!(document.daily.len(), 2);
        assert_eq!(document.daily[0].usage.tokens.total, 0);
        assert_eq!(document.daily[1].usage.tokens.total, 117);
        assert_eq!(document.top_models[0].model_id, "gpt-5.6-codex");
    }

    #[test]
    fn required_range_starts_at_the_oldest_local_midnight() {
        let now = Local.with_ymd_and_hms(2026, 7, 28, 14, 37, 0).unwrap();
        let query = SnapshotQuery {
            days: 2,
            tool: Tool::All,
            session_id: None,
        };

        let range = required_range(&query, now);

        assert_eq!(
            range.start().date_naive(),
            NaiveDate::from_ymd_opt(2026, 7, 27).unwrap()
        );
        assert_eq!(range.start().time(), chrono::NaiveTime::MIN);
        assert_eq!(range.end(), now);
    }

    #[test]
    fn json_writer_emits_one_pretty_document() {
        let now = Local.with_ymd_and_hms(2026, 7, 28, 14, 37, 0).unwrap();
        let document = build_document(now, now, true, &[], Vec::new());
        let mut output = Vec::new();

        write_json(&mut output, &document).unwrap();

        let text = String::from_utf8(output).unwrap();
        assert!(text.starts_with("{\n"));
        assert!(text.ends_with("}\n"));
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["schema_version"], 1);
    }
}
