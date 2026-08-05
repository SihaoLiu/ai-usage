//! Dashboard assembly for the monitor UI.
//!
//! Builds a render-agnostic snapshot of everything one frame shows: the
//! breakdown table rows, the projected cost summary, and the chart series.
//! Rebuilt only when state changes (tool, window, view, data refresh), not
//! per animation tick.

use std::collections::HashSet;

use chrono::{DateTime, Local};

use crate::charts::{self, ChartGranularity};
use crate::stats;
use crate::table_view::{
    CostSummary, DisplayRow, RowMetrics, TableMetric, TableView, build_table, cost_summary,
    table_totals,
};
use crate::time_utils::{generate_interval_times, to_interval};
use crate::tool::Tool;
use crate::{AppState, formatting};

pub struct Series {
    pub name: String,
    pub color: u8,
    pub points: Vec<(f64, f64)>,
}

/// One granularity segment of the x axis: `[start, end]` are inclusive data
/// indices, `total` is the summed token volume shown in the segment header.
pub struct Segment {
    pub start: usize,
    pub end: usize,
    pub total: f64,
    pub anchor: DateTime<Local>,
}

pub struct ChartData {
    pub title: String,
    pub series: Vec<Series>,
    pub max_y: f64,
    pub len: usize,
    pub granularity: ChartGranularity,
    /// Granularity segments in chronological order; a vertical separator is
    /// drawn before each segment after the first.
    pub segments: Vec<Segment>,
    /// X-axis tick labels as `(data index, label)`.
    pub x_ticks: Vec<(usize, String)>,
}

pub struct Dashboard {
    pub tool: Tool,
    pub view: TableView,
    pub sort_metric: TableMetric,
    pub window_label: String,
    pub window_complete: bool,
    pub has_visible_data: bool,
    pub session_id: Option<String>,
    /// Aggregated per-(harness, model) rows the table views are built from;
    /// kept so a view toggle can reshape the table without rescanning data.
    pub model_stats: Vec<crate::stats::ModelBreakdownRow>,
    pub rows: Vec<DisplayRow>,
    pub totals: RowMetrics,
    pub summary: CostSummary,
    pub insight: Option<String>,
    /// Context line for the header: the weighted cross-tool summary in
    /// all-tools mode, or the harness CLI version for a single tool.
    pub headline: Option<String>,
    pub span_label: String,
    pub charts: Vec<ChartData>,
}

fn interval_times(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    interval_minutes: i64,
) -> Vec<DateTime<Local>> {
    let start = to_interval(range_start, interval_minutes);
    generate_interval_times(&start, range_end, interval_minutes)
}

/// Group chronological, uniformly-bucketed times into granularity segments.
fn build_segments(
    times: &[DateTime<Local>],
    granularity: ChartGranularity,
    total_at: impl Fn(usize) -> f64,
) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    for (i, t) in times.iter().enumerate() {
        let anchor = granularity.segment_start(*t);
        match segments.last_mut() {
            Some(seg) if seg.anchor == anchor => {
                seg.end = i;
                seg.total += total_at(i);
            }
            _ => segments.push(Segment {
                start: i,
                end: i,
                total: total_at(i),
                anchor,
            }),
        }
    }
    segments
}

/// X-axis tick labels mapped onto uniform data indices.
fn build_x_ticks(
    times: &[DateTime<Local>],
    interval_minutes: i64,
    granularity: ChartGranularity,
) -> Vec<(usize, String)> {
    let (Some(first), Some(last)) = (times.first(), times.last()) else {
        return Vec::new();
    };
    let (tick_times, tick_interval) = charts::axis_tick_times(first, last, granularity);
    let mut ticks: Vec<(usize, String)> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    for tick in tick_times {
        let offset_minutes = (tick - *first).num_minutes();
        let idx = ((offset_minutes as f64 / interval_minutes.max(1) as f64).round() as i64)
            .clamp(0, times.len() as i64 - 1) as usize;
        if used.insert(idx) {
            ticks.push((idx, charts::x_tick_label(&tick, granularity, tick_interval)));
        }
    }
    ticks
}

/// Flip a chart horizontally so the newest bucket renders on the left, the
/// established orientation of the plain charts. Everything is built in
/// chronological order first, then mirrored once here.
fn mirror_x(chart: &mut ChartData) {
    let n = chart.len;
    if n == 0 {
        return;
    }
    let flip = |i: usize| n - 1 - i;
    for series in &mut chart.series {
        for point in &mut series.points {
            point.0 = flip(point.0 as usize) as f64;
        }
    }
    for seg in &mut chart.segments {
        let (start, end) = (flip(seg.end), flip(seg.start));
        seg.start = start;
        seg.end = end;
    }
    chart.segments.reverse();
    for (idx, _) in &mut chart.x_ticks {
        *idx = flip(*idx);
    }
}

fn max_over(series: &[Series]) -> f64 {
    series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.1))
        .fold(1.0_f64, f64::max)
}

fn interval_label(interval_minutes: i64) -> String {
    if interval_minutes % 1440 == 0 {
        format!("{}d", interval_minutes / 1440)
    } else if interval_minutes % 60 == 0 {
        format!("{}h", interval_minutes / 60)
    } else {
        format!("{}m", interval_minutes)
    }
}

fn model_chart(
    series_map: &stats::ModelTimeSeries,
    included: &HashSet<String>,
    chart_type: &str,
    tool_key: &str,
    times: &[DateTime<Local>],
    interval_minutes: i64,
    granularity: ChartGranularity,
) -> ChartData {
    let mut models: HashSet<String> = series_map
        .values()
        .flat_map(|m| m.keys().cloned())
        .collect();
    models.retain(|m| included.contains(m));

    let series: Vec<Series> = charts::model_series_specs(&models, chart_type, tool_key)
        .into_iter()
        .map(|spec| {
            let points: Vec<(f64, f64)> = times
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let value = series_map
                        .get(t)
                        .and_then(|m| m.get(&spec.model))
                        .map(|b| match spec.token_type {
                            "input" => b.input,
                            "output" => b.output,
                            "cache_creation" => b.cache_creation,
                            "cache_read" => b.cache_read,
                            _ => 0.0,
                        })
                        .unwrap_or(0.0);
                    (i as f64, value)
                })
                .collect();
            Series {
                name: spec.label,
                color: spec.color_index,
                points,
            }
        })
        .collect();

    let max_y = max_over(&series);
    // Segment totals sum every line at a data point, matching the plain chart.
    let segments = build_segments(times, granularity, |idx| {
        series.iter().map(|s| s.points[idx].1).sum()
    });
    let mut chart = ChartData {
        title: charts::model_chart_title(chart_type, tool_key).to_string(),
        series,
        max_y,
        len: times.len(),
        granularity,
        segments,
        x_ticks: build_x_ticks(times, interval_minutes, granularity),
    };
    mirror_x(&mut chart);
    chart
}

fn comparison_chart(
    series_map: &stats::ToolTimeSeries,
    times: &[DateTime<Local>],
    interval_minutes: i64,
    granularity: ChartGranularity,
) -> ChartData {
    let all_tools: HashSet<String> = series_map
        .values()
        .flat_map(|m| m.keys().cloned())
        .collect();
    let order = charts::comparison_tool_order(&all_tools);

    let value_at = |tool: &str, t: &DateTime<Local>| -> f64 {
        series_map
            .get(t)
            .and_then(|m| m.get(tool))
            .copied()
            .unwrap_or(0.0)
    };

    let series: Vec<Series> = order
        .iter()
        .map(|label| {
            let points: Vec<(f64, f64)> = times
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let value = if label == "All" {
                        order
                            .iter()
                            .filter(|l| *l != "All")
                            .map(|l| value_at(l, t))
                            .sum()
                    } else {
                        value_at(label, t)
                    };
                    (i as f64, value)
                })
                .collect();
            Series {
                name: label.clone(),
                color: charts::tool_color_index(label),
                points,
            }
        })
        .collect();

    let max_y = max_over(&series);
    // The aggregate `All` line is the last series; its values are already the
    // per-point totals, so segment sums read from it directly.
    let segments = build_segments(times, granularity, |idx| {
        series.last().map(|s| s.points[idx].1).unwrap_or(0.0)
    });
    let mut chart = ChartData {
        title: "Total Token Consumption by Tool".to_string(),
        series,
        max_y,
        len: times.len(),
        granularity,
        segments,
        x_ticks: build_x_ticks(times, interval_minutes, granularity),
    };
    mirror_x(&mut chart);
    chart
}

/// Reshape the table from the cached aggregation without rescanning data.
pub fn rebuild_table(dash: &mut Dashboard, view: TableView, sort_metric: TableMetric) {
    dash.view = view;
    dash.sort_metric = sort_metric;
    dash.rows = build_table(&dash.model_stats, view, sort_metric);
    dash.insight =
        formatting::top_model_insight_line(&dash.rows, dash.summary.total_cost, dash.tool.is_all());
}

pub fn build(state: &mut AppState) -> Dashboard {
    let now = Local::now();
    let tool = Tool::from_key(&state.tool).unwrap_or(Tool::All);
    let single_tool_headline = if tool.is_all() {
        None
    } else {
        let version = crate::get_version(state, tool.key());
        (!version.is_empty()).then_some(version)
    };
    let view = state.table_view;
    let sort_metric = state.sort_metric;
    let (range_start, range_end) = state.time_window.bounds(now);
    let projection_days = state.time_window.projection_days(now);
    let window_label = crate::showing_data_line(&state.time_window, now);

    let granularity = crate::display_chart_granularity(&range_start, &range_end);
    // Same value the automatic refresh cadence is derived from, so the
    // interval shown in the span line is the one being paced against.
    let interval_minutes = crate::display_interval_minutes_for_window(
        &state.time_window,
        now,
        crate::get_chart_target_width(),
    );
    let times = interval_times(&range_start, &range_end, interval_minutes);

    let window_complete = crate::raw_cache_covers_window(state, now);
    let all_data = crate::load_resident_all_tool_data(state, now);
    let has_visible_data = crate::all_tool_data_has_window_data(&all_data);

    let (model_stats, charts_data, headline) = if tool.is_all() {
        let (model_stats, tool_ts) =
            crate::calculate_all_dashboard_data(&all_data, &state.pricing, interval_minutes);
        let (weighted_cost, total_savings) = crate::calculate_weighted_cost_per_mtok(
            &model_stats,
            projection_days,
            &state.subscription_fees,
        );
        let headline = (weighted_cost > 0.0).then(|| {
            format!(
                "All Tools Comparison, {} / MTok, Monthly Saving ${:.2}",
                formatting::format_cost_per_mtok(weighted_cost),
                total_savings,
            )
        });
        let chart = comparison_chart(&tool_ts, &times, interval_minutes, granularity);
        (model_stats, vec![chart], headline)
    } else {
        let filtered = match tool {
            Tool::Codex => &all_data.codex,
            Tool::Gemini => &all_data.gemini,
            Tool::Kimi => &all_data.kimi,
            Tool::Omp => &all_data.omp,
            _ => &all_data.claude,
        };
        let (model_stats, model_ts) = stats::calculate_model_dashboard_data(
            filtered,
            interval_minutes,
            tool.key(),
            &state.pricing,
        );
        let included: HashSet<String> = model_stats.iter().map(|s| s.model.clone()).collect();
        let io = model_chart(
            &model_ts,
            &included,
            "io",
            tool.key(),
            &times,
            interval_minutes,
            granularity,
        );
        let cache = model_chart(
            &model_ts,
            &included,
            "cache",
            tool.key(),
            &times,
            interval_minutes,
            granularity,
        );
        (model_stats, vec![io, cache], single_tool_headline)
    };

    let rows = build_table(&model_stats, view, sort_metric);
    let totals = table_totals(&model_stats);
    let model_stats_kept = model_stats;
    let subscription_price = state.subscription_fees.get(tool.key());
    let summary = cost_summary(&totals, projection_days, subscription_price);
    let insight = formatting::top_model_insight_line(&rows, summary.total_cost, tool.is_all());

    let span_label = format!(
        "updated {} | {} to {} | interval {} | {} pts",
        now.format("%H:%M:%S"),
        range_start.format("%Y-%m-%d %H:%M"),
        range_end.format("%Y-%m-%d %H:%M"),
        interval_label(interval_minutes),
        times.len(),
    );

    Dashboard {
        tool,
        view,
        sort_metric,
        window_label,
        window_complete,
        has_visible_data,
        session_id: state.session_id.clone(),
        model_stats: model_stats_kept,
        rows,
        totals,
        summary,
        insight,
        headline,
        span_label,
        charts: charts_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn interval_times_are_uniform_and_cover_the_range() {
        let start = Local.with_ymd_and_hms(2026, 7, 20, 10, 5, 0).unwrap();
        let end = Local.with_ymd_and_hms(2026, 7, 20, 13, 0, 0).unwrap();
        let times = interval_times(&start, &end, 60);
        assert!(times.len() >= 3);
        assert!(times[0] <= start);
        assert!(*times.last().unwrap() <= end);
        for pair in times.windows(2) {
            assert_eq!((pair[1] - pair[0]).num_minutes(), 60);
        }
    }

    #[test]
    fn multi_day_chart_times_include_post_dst_usage_buckets() {
        const DST_TEST_CHILD: &str = "AI_USAGE_DST_TEST_CHILD";
        if std::env::var_os(DST_TEST_CHILD).is_none() {
            let output = std::process::Command::new(
                std::env::current_exe().expect("current test executable"),
            )
            .args([
                "--exact",
                "tui::data::tests::multi_day_chart_times_include_post_dst_usage_buckets",
                "--nocapture",
            ])
            .env("TZ", "America/Los_Angeles")
            .env(DST_TEST_CHILD, "1")
            .output()
            .expect("run DST regression test in an isolated process");

            assert!(
                output.status.success(),
                "DST regression subprocess failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        let start = Local.with_ymd_and_hms(2026, 1, 13, 15, 17, 0).unwrap();
        let end = Local.with_ymd_and_hms(2026, 7, 24, 16, 7, 0).unwrap();
        let interval_minutes = 2 * 24 * 60;
        let times = interval_times(&start, &end, interval_minutes);
        let post_dst_usage = Local.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let usage_bucket = to_interval(&post_dst_usage, interval_minutes);

        assert!(
            times.contains(&usage_bucket),
            "chart buckets must include the same {:?} key used by aggregation",
            usage_bucket
        );
    }

    #[test]
    fn segments_group_buckets_by_granularity_and_sum_totals() {
        // Three days of hourly buckets under Day granularity: one segment per
        // day, boundaries at midnight, totals summed per segment.
        let start = Local.with_ymd_and_hms(2026, 7, 20, 6, 0, 0).unwrap();
        let end = Local.with_ymd_and_hms(2026, 7, 22, 18, 0, 0).unwrap();
        let times = interval_times(&start, &end, 60);
        let segments = build_segments(&times, ChartGranularity::Day, |_| 2.0);

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].start, 0);
        // Continuity: each segment starts right after the previous one ends.
        for pair in segments.windows(2) {
            assert_eq!(pair[1].start, pair[0].end + 1);
            // Segment boundary buckets sit at local midnight.
            assert_eq!(times[pair[1].start].format("%H:%M").to_string(), "00:00");
        }
        let total: f64 = segments.iter().map(|s| s.total).sum();
        assert_eq!(total, 2.0 * times.len() as f64);
    }

    #[test]
    fn x_ticks_map_to_data_indices_without_duplicates() {
        let start = Local.with_ymd_and_hms(2026, 7, 20, 0, 0, 0).unwrap();
        let end = Local.with_ymd_and_hms(2026, 7, 22, 0, 0, 0).unwrap();
        let times = interval_times(&start, &end, 60);
        let ticks = build_x_ticks(&times, 60, ChartGranularity::Day);

        assert!(
            ticks.len() >= 4,
            "expected dense ticks, got {}",
            ticks.len()
        );
        let mut seen = HashSet::new();
        for (idx, label) in &ticks {
            assert!(*idx < times.len());
            assert!(seen.insert(*idx));
            assert!(!label.is_empty());
        }
    }

    #[test]
    fn mirror_puts_newest_bucket_on_the_left() {
        let day = |d: u32| Local.with_ymd_and_hms(2026, 7, d, 0, 0, 0).unwrap();
        let mut chart = ChartData {
            title: String::new(),
            series: vec![Series {
                name: "s".to_string(),
                color: 1,
                // Value grows with time: newest bucket has the largest value.
                points: (0..10).map(|i| (i as f64, i as f64)).collect(),
            }],
            max_y: 9.0,
            len: 10,
            granularity: ChartGranularity::Day,
            segments: vec![
                Segment {
                    start: 0,
                    end: 4,
                    total: 10.0,
                    anchor: day(20),
                },
                Segment {
                    start: 5,
                    end: 9,
                    total: 35.0,
                    anchor: day(21),
                },
            ],
            x_ticks: vec![(0, "20".to_string()), (9, "21".to_string())],
        };
        mirror_x(&mut chart);

        // The newest (largest) value now sits at x = 0.
        let newest = chart.series[0]
            .points
            .iter()
            .find(|p| p.1 == 9.0)
            .expect("newest point");
        assert_eq!(newest.0, 0.0);
        // Segments run newest-first and stay contiguous.
        assert_eq!(chart.segments[0].anchor, day(21));
        assert_eq!((chart.segments[0].start, chart.segments[0].end), (0, 4));
        assert_eq!((chart.segments[1].start, chart.segments[1].end), (5, 9));
        // Ticks flip with the axis.
        assert_eq!(chart.x_ticks[0], (9, "20".to_string()));
        assert_eq!(chart.x_ticks[1], (0, "21".to_string()));
    }

    #[test]
    fn interval_labels_use_natural_units() {
        assert_eq!(interval_label(30), "30m");
        assert_eq!(interval_label(120), "2h");
        assert_eq!(interval_label(2880), "2d");
    }
}
