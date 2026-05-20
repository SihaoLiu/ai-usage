use std::collections::{HashMap, HashSet};

use crate::time_utils::{generate_interval_times, round_to_interval_start};
use chrono::{DateTime, Datelike, Duration, Local, TimeZone, Timelike, Weekday};

use crate::formatting::{center_pad, format_total_value, format_y_axis_value};
use crate::stats::{ModelTimeSeries, VendorTimeSeries};

const RESET_COLOR: &str = "\x1b[0m";
const DIM_COLOR: &str = "\x1b[38;5;240m";

/// Visual segmentation unit for time-series charts. Determines where
/// separators are drawn between data points and how the header labels each
/// group. The unit is independent from the data bucketing interval — a chart
/// can bucket data into 5-minute intervals while drawing hour-wide segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartGranularity {
    Hour,
    Day,
    Week,
    Year,
}

impl ChartGranularity {
    /// Pick a granularity from the span of the time window in minutes. The
    /// thresholds aim for roughly 3-15 segments per chart so the header has
    /// room to render labels without overcrowding.
    pub fn from_span_minutes(span_minutes: i64) -> Self {
        let span_hours = span_minutes / 60;
        let span_days = span_minutes / (24 * 60);
        if span_days >= 365 {
            ChartGranularity::Year
        } else if span_days >= 18 {
            ChartGranularity::Week
        } else if span_hours <= 12 {
            ChartGranularity::Hour
        } else {
            ChartGranularity::Day
        }
    }

    /// True when `t` lies on the start of its granularity period (e.g. for
    /// `Day`, only midnight is a boundary).
    pub fn is_boundary(self, t: &DateTime<Local>) -> bool {
        let exact_minute = t.second() == 0 && t.nanosecond() == 0;
        match self {
            ChartGranularity::Hour => t.minute() == 0 && exact_minute,
            ChartGranularity::Day => t.hour() == 0 && t.minute() == 0 && exact_minute,
            ChartGranularity::Week => {
                t.hour() == 0 && t.minute() == 0 && exact_minute && t.weekday() == Weekday::Mon
            }
            ChartGranularity::Year => {
                t.month() == 1 && t.day() == 1 && t.hour() == 0 && t.minute() == 0 && exact_minute
            }
        }
    }

    /// Round `t` down to the start of its enclosing segment. For `Week`, this
    /// is the most-recent Monday midnight; the lookup is DST-safe.
    pub fn segment_start(self, t: DateTime<Local>) -> DateTime<Local> {
        let naive = match self {
            ChartGranularity::Hour => t
                .date_naive()
                .and_hms_opt(t.hour(), 0, 0)
                .expect("hour anchor"),
            ChartGranularity::Day => t.date_naive().and_hms_opt(0, 0, 0).expect("day anchor"),
            ChartGranularity::Week => {
                let days_from_mon = t.weekday().num_days_from_monday() as i64;
                let mon_date = t.date_naive() - Duration::days(days_from_mon);
                mon_date.and_hms_opt(0, 0, 0).expect("week anchor")
            }
            ChartGranularity::Year => chrono::NaiveDate::from_ymd_opt(t.year(), 1, 1)
                .expect("year date")
                .and_hms_opt(0, 0, 0)
                .expect("year anchor"),
        };
        match Local.from_local_datetime(&naive) {
            chrono::LocalResult::Single(d) | chrono::LocalResult::Ambiguous(d, _) => d,
            chrono::LocalResult::None => t,
        }
    }
}

// Model display configuration for Claude (order matters)
fn model_config() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        ("claude-opus-4-6", "Opus 4.6", 0),
        ("claude-opus-4-5-20251101", "Opus 4.5", 1),
        ("claude-opus-4-1-20250805", "Opus 4.1", 2),
        ("claude-sonnet-4-6", "Sonnet 4.6", 3),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", 4),
        ("claude-sonnet-4-20250514", "Sonnet 4", 5),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", 6),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn daily_header_lines_do_not_exceed_chart_width_at_right_boundary() {
        let day = Local
            .with_ymd_and_hms(2026, 5, 7, 12, 0, 0)
            .single()
            .expect("fixed local time");
        let mut columns = Vec::new();
        for sub_col in 0..7 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..13 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        columns.push(ChartColumn::Data {
            data_idx: 0,
            sub_col: 0,
        });
        let layout = ChartLayout {
            columns,
            data_to_col: HashMap::new(),
            sorted_times: vec![day],
        };
        let chart_width = 7 + layout.columns.len();

        let Some((weekday_line, date_line)) =
            segment_header_lines(&layout, ChartGranularity::Day, &|_| 0.0)
        else {
            panic!("header should render");
        };

        assert!(weekday_line.len() <= chart_width);
        assert!(date_line.len() <= chart_width);
        assert!(date_line.ends_with("05 / 07"));
    }

    #[test]
    fn hour_header_aligns_date_slash_under_total_separator() {
        let hour = Local
            .with_ymd_and_hms(2026, 5, 11, 12, 30, 0)
            .single()
            .expect("fixed local time");
        let mut columns = Vec::new();
        for sub_col in 0..20 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        let layout = ChartLayout {
            columns,
            data_to_col: HashMap::new(),
            sorted_times: vec![hour],
        };

        let Some((time_line, date_line)) =
            segment_header_lines(&layout, ChartGranularity::Hour, &|_| 3_260_000.0)
        else {
            panic!("header should render");
        };

        assert!(time_line.contains("12:00 : 3.26M"));
        assert!(date_line.contains("05 / 11"));

        let first_colon = time_line.find(':').expect("time colon");
        let total_separator = time_line.rfind(':').expect("total separator");
        let date_separator = date_line.find('/').expect("date separator");

        assert_ne!(date_separator, first_colon);
        assert_eq!(date_separator, total_separator);
    }

    #[test]
    fn granularity_picks_year_for_annual_spans() {
        assert_eq!(
            ChartGranularity::from_span_minutes(60),
            ChartGranularity::Hour
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(6 * 60),
            ChartGranularity::Hour
        );
        // Boundary: 13 hours falls into Day so daily totals stay legible.
        assert_eq!(
            ChartGranularity::from_span_minutes(13 * 60),
            ChartGranularity::Day
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(3 * 24 * 60),
            ChartGranularity::Day
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(17 * 24 * 60),
            ChartGranularity::Day
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(18 * 24 * 60),
            ChartGranularity::Week
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(90 * 24 * 60),
            ChartGranularity::Week
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(364 * 24 * 60),
            ChartGranularity::Week
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(365 * 24 * 60),
            ChartGranularity::Year
        );
        assert_eq!(
            ChartGranularity::from_span_minutes(730 * 24 * 60),
            ChartGranularity::Year
        );
    }

    #[test]
    fn week_granularity_anchors_to_monday() {
        // 2026-05-07 is a Thursday; the week anchor should be Monday 05/04.
        let thu = Local
            .with_ymd_and_hms(2026, 5, 7, 15, 30, 0)
            .single()
            .expect("fixed local time");
        let anchor = ChartGranularity::Week.segment_start(thu);
        assert_eq!(anchor.weekday(), Weekday::Mon);
        assert_eq!(
            anchor.format("%Y-%m-%d %H:%M").to_string(),
            "2026-05-04 00:00"
        );
    }

    #[test]
    fn year_granularity_anchors_to_january_first() {
        let may = Local
            .with_ymd_and_hms(2026, 5, 7, 15, 30, 0)
            .single()
            .expect("fixed local time");
        let anchor = ChartGranularity::Year.segment_start(may);
        assert_eq!(anchor.month(), 1);
        assert_eq!(anchor.day(), 1);
        assert_eq!(
            anchor.format("%Y-%m-%d %H:%M").to_string(),
            "2026-01-01 00:00"
        );
    }

    #[test]
    fn year_granularity_treats_only_new_year_midnight_as_boundary() {
        let boundary = Local
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .single()
            .expect("new year");
        let later = Local
            .with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
            .single()
            .expect("same day");
        let prior = Local
            .with_ymd_and_hms(2025, 12, 31, 0, 0, 0)
            .single()
            .expect("prior day");
        assert!(ChartGranularity::Year.is_boundary(&boundary));
        assert!(!ChartGranularity::Year.is_boundary(&later));
        assert!(!ChartGranularity::Year.is_boundary(&prior));
    }

    #[test]
    fn year_header_labels_year_and_anchor_date() {
        let day = Local
            .with_ymd_and_hms(2026, 5, 11, 12, 0, 0)
            .single()
            .expect("fixed local time");
        let mut columns = Vec::new();
        for sub_col in 0..20 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        let layout = ChartLayout {
            columns,
            data_to_col: HashMap::new(),
            sorted_times: vec![day],
        };

        let Some((year_line, date_line)) =
            segment_header_lines(&layout, ChartGranularity::Year, &|_| 3_260_000.0)
        else {
            panic!("header should render");
        };

        assert!(year_line.contains("2026 : 3.26M"));
        assert!(date_line.contains("01 / 01"));
    }

    #[test]
    fn hour_granularity_treats_only_minute_zero_as_boundary() {
        let mid = Local
            .with_ymd_and_hms(2026, 5, 7, 13, 0, 0)
            .single()
            .expect("hh:00");
        let off = Local
            .with_ymd_and_hms(2026, 5, 7, 13, 5, 0)
            .single()
            .expect("hh:05");
        assert!(ChartGranularity::Hour.is_boundary(&mid));
        assert!(!ChartGranularity::Hour.is_boundary(&off));
    }

    #[test]
    fn hour_granularity_requires_exact_hour_boundary() {
        let with_seconds = Local
            .with_ymd_and_hms(2026, 5, 7, 13, 0, 30)
            .single()
            .expect("hh:00:ss");
        let with_nanos = Local
            .with_ymd_and_hms(2026, 5, 7, 13, 0, 0)
            .single()
            .expect("hh:00")
            .with_nanosecond(1)
            .expect("nanosecond");

        assert!(!ChartGranularity::Hour.is_boundary(&with_seconds));
        assert!(!ChartGranularity::Hour.is_boundary(&with_nanos));
    }

    #[test]
    fn week_x_axis_ticks_anchor_to_next_monday() {
        let first_time = Local
            .with_ymd_and_hms(2026, 5, 13, 9, 0, 0)
            .single()
            .expect("fixed local time");
        let tick = first_x_axis_tick(&first_time, ChartGranularity::Week, 10_080);

        assert_eq!(tick.weekday(), Weekday::Mon);
        assert_eq!(
            tick.format("%Y-%m-%d %H:%M").to_string(),
            "2026-05-18 00:00"
        );
    }

    fn count_separators(layout: &ChartLayout) -> usize {
        layout
            .columns
            .iter()
            .filter(|c| matches!(c, ChartColumn::Separator))
            .count()
    }

    #[test]
    fn hour_granularity_places_separator_at_each_hour_boundary() {
        // A 6-hour window with 5-minute buckets should produce one
        // separator per internal hour boundary (i.e. 5 separators for
        // boundaries at 13:00..17:00 — the first boundary at 18:00 is
        // skipped because it is the leftmost data point).
        let start = Local
            .with_ymd_and_hms(2026, 5, 7, 12, 0, 0)
            .single()
            .expect("start");
        let end = Local
            .with_ymd_and_hms(2026, 5, 7, 18, 0, 0)
            .single()
            .expect("end");
        let layout = build_chart_layout(&start, &end, 5, ChartGranularity::Hour, Some(160));
        // The leftmost data point (18:00) and the rightmost data point
        // (12:00) are both on-boundary; both should be free of an adjacent
        // separator so the visible chart has 5 internal separators.
        assert_eq!(count_separators(&layout), 5);
    }

    #[test]
    fn week_granularity_places_separators_only_on_internal_mondays() {
        // A 14-day window with daily buckets should produce separators only
        // on internal Mondays.
        let end = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("end");
        let start = end - Duration::days(14);
        let layout = build_chart_layout(&start, &end, 1440, ChartGranularity::Week, Some(160));
        // The internal Mondays are 2026-05-04 and 2026-05-11.
        assert_eq!(count_separators(&layout), 2);
    }

    #[test]
    fn year_granularity_places_separators_only_on_internal_years() {
        let start = Local
            .with_ymd_and_hms(2024, 7, 1, 0, 0, 0)
            .single()
            .expect("start");
        let end = Local
            .with_ymd_and_hms(2026, 5, 14, 0, 0, 0)
            .single()
            .expect("end");
        let layout = build_chart_layout(&start, &end, 1440, ChartGranularity::Year, Some(160));
        assert_eq!(count_separators(&layout), 2);
    }

    #[test]
    fn narrow_left_boundary_segment_drops_when_it_would_collide_with_inner() {
        // Leftmost segment is too narrow to centre the 13-char week label
        // on its mid-column. Even when shifted flush against the chart's
        // left edge, the boundary label would overlap the next inner
        // segment's centred label. The inner segment must win and the
        // boundary label must vanish so it never displaces a full segment.
        let mon = Local
            .with_ymd_and_hms(2026, 5, 11, 12, 0, 0)
            .single()
            .expect("monday");
        let mut columns = Vec::new();
        for sub_col in 0..4 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..20 {
            columns.push(ChartColumn::Data {
                data_idx: 1,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..20 {
            columns.push(ChartColumn::Data {
                data_idx: 2,
                sub_col,
            });
        }
        let prev_mon = Local
            .with_ymd_and_hms(2026, 5, 4, 12, 0, 0)
            .single()
            .expect("prev monday");
        let prev_prev_mon = Local
            .with_ymd_and_hms(2026, 4, 27, 12, 0, 0)
            .single()
            .expect("prev prev monday");
        let layout = ChartLayout {
            columns,
            data_to_col: HashMap::new(),
            sorted_times: vec![mon, prev_mon, prev_prev_mon],
        };

        let Some((head_line, _date_line)) =
            segment_header_lines(&layout, ChartGranularity::Week, &|idx| match idx {
                0 => 1_030_000_000.0,
                1 => 1_540_000_000.0,
                _ => 965_000_000.0,
            })
        else {
            panic!("header should render");
        };

        assert!(
            !head_line.contains("Wk 20"),
            "narrow leftmost boundary label should be dropped to preserve Wk 19"
        );
        assert!(
            head_line.contains("Wk 19"),
            "inner segment must remain visible"
        );
        assert!(
            head_line.contains("Wk 18"),
            "trailing inner segment must remain visible"
        );
    }

    #[test]
    fn boundary_label_kept_with_one_char_gap_uses_natural_alignment() {
        // Leftmost segment is narrow enough that its centred label would
        // run off the left edge, but the inner segment that follows is
        // positioned so the boundary label still has at least one empty
        // column of breathing room. The boundary label should render and
        // stack head/date on a shared left edge (natural alignment), and
        // the inner segment must remain visible too.
        let mon = Local
            .with_ymd_and_hms(2026, 5, 11, 12, 0, 0)
            .single()
            .expect("monday");
        let mut columns = Vec::new();
        for sub_col in 0..12 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..15 {
            columns.push(ChartColumn::Data {
                data_idx: 1,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..13 {
            columns.push(ChartColumn::Data {
                data_idx: 2,
                sub_col,
            });
        }
        let prev_mon = Local
            .with_ymd_and_hms(2026, 5, 4, 12, 0, 0)
            .single()
            .expect("prev monday");
        let prev_prev_mon = Local
            .with_ymd_and_hms(2026, 4, 27, 12, 0, 0)
            .single()
            .expect("prev prev monday");
        let layout = ChartLayout {
            columns,
            data_to_col: HashMap::new(),
            sorted_times: vec![mon, prev_mon, prev_prev_mon],
        };

        let Some((head_line, date_line)) =
            segment_header_lines(&layout, ChartGranularity::Week, &|idx| match idx {
                0 => 1_030_000_000.0,
                1 => 1_540_000_000.0,
                _ => 965_000_000.0,
            })
        else {
            panic!("header should render");
        };

        let boundary_head = head_line.find("Wk 20").expect("Wk 20 head present");
        let boundary_date = date_line.find("05 / 11").expect("05 / 11 date present");
        assert_eq!(
            boundary_head, boundary_date,
            "kept boundary label should stack head and date on the same column"
        );

        let inner_head = head_line.find("Wk 19").expect("Wk 19 head present");
        assert!(
            inner_head > boundary_head + "Wk 20 : 1.03B".len(),
            "inner label must be separated from the boundary label by >= 1 column"
        );
    }

    #[test]
    fn compact_day_partial_edge_drops_when_label_would_touch_inner() {
        // In compact Day mode the leftmost partial-day segment's centred
        // label (e.g. "Th:229M") happens to fit within chart bounds, but
        // its end column collides with the centred label of the adjacent
        // full inner day ("We:61M"). The narrower partial segment must
        // drop so the inner label keeps the required one-column buffer.
        let thu = Local
            .with_ymd_and_hms(2026, 5, 7, 18, 0, 0)
            .single()
            .expect("Thu");
        let wed = Local
            .with_ymd_and_hms(2026, 5, 6, 12, 0, 0)
            .single()
            .expect("Wed");
        let tue = Local
            .with_ymd_and_hms(2026, 5, 5, 12, 0, 0)
            .single()
            .expect("Tue");

        let mut columns = Vec::new();
        for sub_col in 0..5 {
            columns.push(ChartColumn::Data {
                data_idx: 0,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..7 {
            columns.push(ChartColumn::Data {
                data_idx: 1,
                sub_col,
            });
        }
        columns.push(ChartColumn::Separator);
        for sub_col in 0..7 {
            columns.push(ChartColumn::Data {
                data_idx: 2,
                sub_col,
            });
        }
        let layout = ChartLayout {
            columns,
            data_to_col: HashMap::new(),
            sorted_times: vec![thu, wed, tue],
        };

        let Some((head_line, _date_line)) =
            segment_header_lines(&layout, ChartGranularity::Day, &|idx| match idx {
                0 => 229_000_000.0,
                1 => 61_000_000.0,
                _ => 43_000_000.0,
            })
        else {
            panic!("header should render");
        };

        assert!(
            !head_line.contains("Th:229M"),
            "partial Th label must drop to preserve the one-column gap before We"
        );
        assert!(
            head_line.contains("We:61M"),
            "inner We label must remain visible"
        );
        assert!(
            head_line.contains("Tu:43M"),
            "inner Tu label must remain visible"
        );
    }

    #[test]
    fn weekly_layout_honors_target_width_with_multi_day_buckets() {
        let start = Local
            .with_ymd_and_hms(2025, 11, 3, 0, 0, 0)
            .single()
            .expect("start");
        let end = Local
            .with_ymd_and_hms(2026, 5, 14, 23, 59, 59)
            .single()
            .expect("end");
        let layout = build_chart_layout(&start, &end, 2 * 1440, ChartGranularity::Week, Some(160));

        assert!(layout.sorted_times.len() > 50);
        assert!(7 + layout.columns.len() <= 160);
    }

    #[test]
    fn year_granularity_marks_year_change_with_coarse_buckets() {
        let start = Local
            .with_ymd_and_hms(2025, 12, 15, 0, 0, 0)
            .single()
            .expect("start");
        let end = Local
            .with_ymd_and_hms(2026, 1, 20, 0, 0, 0)
            .single()
            .expect("end");
        let layout = build_chart_layout(&start, &end, 14 * 1440, ChartGranularity::Year, Some(80));

        assert_eq!(count_separators(&layout), 1);
    }

    #[test]
    fn day_granularity_skips_boundary_at_earliest_data_point() {
        // The window starts exactly at midnight, which is a Day boundary.
        // Without the skip rule the earliest column would land in its own
        // singleton segment on the right edge.
        let start = Local
            .with_ymd_and_hms(2026, 5, 5, 0, 0, 0)
            .single()
            .expect("start");
        let end = Local
            .with_ymd_and_hms(2026, 5, 7, 23, 0, 0)
            .single()
            .expect("end");
        let layout = build_chart_layout(&start, &end, 60, ChartGranularity::Day, Some(160));
        // 3 days span -> 2 internal Day boundaries (at 05-06 00:00 and
        // 05-07 00:00). The earliest 00:00 (05-05) is the chronologically
        // last data point and should not get a leading separator.
        assert_eq!(count_separators(&layout), 2);
    }
}

fn model_order(model: &str) -> Option<usize> {
    model_config()
        .iter()
        .find(|(m, _, _)| *m == model)
        .map(|(_, _, o)| *o)
}

fn model_short_name(model: &str) -> Option<&'static str> {
    model_config()
        .iter()
        .find(|(m, _, _)| *m == model)
        .map(|(_, s, _)| *s)
}

// Line color configuration (ANSI 256-color)
fn line_color(key: &str) -> &'static str {
    match key {
        "opus_input" => "\x1b[38;5;196m",
        "opus_output" => "\x1b[38;5;203m",
        "sonnet_input" => "\x1b[38;5;33m",
        "sonnet_output" => "\x1b[38;5;75m",
        "haiku_input" => "\x1b[38;5;40m",
        "haiku_output" => "\x1b[38;5;120m",
        "model0_input" => "\x1b[38;5;208m",
        "model0_output" => "\x1b[38;5;215m",
        "model1_input" => "\x1b[38;5;135m",
        "model1_output" => "\x1b[38;5;177m",
        "model2_input" => "\x1b[38;5;37m",
        "model2_output" => "\x1b[38;5;80m",
        "model3_input" => "\x1b[38;5;197m",
        "model3_output" => "\x1b[38;5;218m",
        "model4_input" => "\x1b[38;5;226m",
        "model4_output" => "\x1b[38;5;228m",
        "model5_input" => "\x1b[38;5;51m",
        "model5_output" => "\x1b[38;5;87m",
        _ => "",
    }
}

fn vendor_color(vendor: &str) -> &'static str {
    match vendor {
        "Claude" => "\x1b[38;5;173m",
        "Codex" => "\x1b[38;5;255m",
        "Gemini" => "\x1b[38;5;33m",
        "All" => "\x1b[38;5;226m",
        _ => "\x1b[38;5;135m",
    }
}

fn get_short_model_name_for_chart(model: &str) -> String {
    if let Some(short) = model_short_name(model) {
        return short.to_string();
    }
    if model.contains(" (")
        && model.ends_with(')')
        && let Some(idx) = model.rfind(" (")
    {
        let base = &model[..idx];
        let effort = &model[idx + 2..model.len() - 1];
        let effort_short = match effort {
            "low" => "L",
            "medium" => "M",
            "high" => "H",
            "xhigh" => "XH",
            _ => &effort[..1],
        };
        return format!("{}({})", base, effort_short);
    }
    if model.len() > 12 {
        model[..12].to_string()
    } else {
        model.to_string()
    }
}

fn round_to_nice(value: f64, round_up: bool) -> i64 {
    let unit = if value >= 5_000_000_000.0 {
        5_000_000_000i64
    } else if value >= 5_000_000.0 {
        5_000_000
    } else if value >= 5_000.0 {
        5_000
    } else {
        5
    };
    if round_up {
        ((value as i64 + unit - 1) / unit) * unit
    } else {
        (value as i64 / unit) * unit
    }
}

fn format_total_compact(value: f64) -> String {
    let (scaled, unit) = if value >= 999_500_000.0 {
        (value / 1_000_000_000.0, "G")
    } else if value >= 999_500.0 {
        (value / 1_000_000.0, "M")
    } else if value >= 999.5 {
        (value / 1_000.0, "K")
    } else {
        return format!("{}", value.round() as i64);
    };
    if scaled < 9.95 {
        format!("{:.1}{}", scaled, unit)
    } else {
        format!("{}{}", scaled.round() as i64, unit)
    }
}

#[derive(Clone)]
enum ChartColumn {
    Separator,
    Data { data_idx: usize, sub_col: usize },
}

struct ChartLayout {
    columns: Vec<ChartColumn>,
    data_to_col: HashMap<usize, usize>,
    sorted_times: Vec<DateTime<Local>>,
}

fn should_insert_separator(
    sorted_times: &[DateTime<Local>],
    index: usize,
    granularity: ChartGranularity,
    interval_minutes: i64,
) -> bool {
    let last_idx = sorted_times.len().saturating_sub(1);
    if index == 0 || index == last_idx {
        return false;
    }

    let time = sorted_times[index];
    if interval_minutes > 1440 {
        return granularity.segment_start(sorted_times[index - 1])
            != granularity.segment_start(time);
    }

    granularity.is_boundary(&time)
}

fn build_chart_layout(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    interval_minutes: i64,
    granularity: ChartGranularity,
    target_width: Option<usize>,
) -> ChartLayout {
    let start_rounded = round_to_interval_start(range_start, interval_minutes);

    let mut sorted_times = generate_interval_times(&start_rounded, range_end, interval_minutes);

    if sorted_times.len() > 500 {
        let step = sorted_times.len() / 500;
        sorted_times = sorted_times.into_iter().step_by(step.max(1)).collect();
    }

    // Reverse: most recent on left
    sorted_times.reverse();

    let num_data_points = sorted_times.len();
    // A separator before the chronologically-earliest data point would create
    // a degenerate single-point segment on the right edge, so the final data
    // index is skipped inside `should_insert_separator`.
    let separator_count = sorted_times
        .iter()
        .enumerate()
        .filter(|(i, _)| should_insert_separator(&sorted_times, *i, granularity, interval_minutes))
        .count();

    let y_axis_width = 7usize;
    let x_scale = if let Some(tw) = target_width {
        let available = tw as f64 - y_axis_width as f64 - separator_count as f64;
        (available / num_data_points as f64).max(1.0)
    } else {
        2.4
    };

    let mut columns = Vec::new();
    let mut data_to_col = HashMap::new();
    let mut col_idx = 0usize;
    let mut accumulated = 0.0f64;

    for (i, _) in sorted_times.iter().enumerate().take(num_data_points) {
        if should_insert_separator(&sorted_times, i, granularity, interval_minutes) {
            columns.push(ChartColumn::Separator);
            col_idx += 1;
        }

        accumulated += x_scale;
        let mut cols = accumulated as usize;
        accumulated -= cols as f64;
        if cols < 1 {
            cols = 1;
        }

        data_to_col.insert(i, col_idx);

        for sub_col in 0..cols {
            columns.push(ChartColumn::Data {
                data_idx: i,
                sub_col,
            });
            col_idx += 1;
        }
    }

    ChartLayout {
        columns,
        data_to_col,
        sorted_times,
    }
}

fn print_segment_header(
    layout: &ChartLayout,
    granularity: ChartGranularity,
    line_values_fn: &dyn Fn(usize) -> f64,
    pad: &str,
) {
    if let Some((line1, line2)) = segment_header_lines(layout, granularity, line_values_fn) {
        println!("{}{}", pad, line1);
        println!("{}{}", pad, line2);
    }
}

fn segment_label(
    anchor: &DateTime<Local>,
    granularity: ChartGranularity,
    total: f64,
    compact: bool,
) -> (String, String) {
    match granularity {
        ChartGranularity::Hour => {
            if compact {
                (
                    format!("{:02}h:{}", anchor.hour(), format_total_compact(total)),
                    anchor.format("%m/%d").to_string(),
                )
            } else {
                (
                    format!("{:02}:00 : {}", anchor.hour(), format_total_value(total)),
                    anchor.format(" %m / %d").to_string(),
                )
            }
        }
        ChartGranularity::Day => {
            let wd_idx = anchor.weekday().num_days_from_monday() as usize;
            if compact {
                let weekday_compact = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];
                (
                    format!(
                        "{}:{}",
                        weekday_compact[wd_idx],
                        format_total_compact(total)
                    ),
                    anchor.format("%m/%d").to_string(),
                )
            } else {
                let weekday_normal = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
                (
                    format!("{} : {}", weekday_normal[wd_idx], format_total_value(total)),
                    anchor.format(" %m / %d").to_string(),
                )
            }
        }
        ChartGranularity::Week => {
            let week_num = anchor.iso_week().week();
            if compact {
                (
                    format!("W{:02}:{}", week_num, format_total_compact(total)),
                    anchor.format("%m/%d").to_string(),
                )
            } else {
                (
                    format!("Wk {:02} : {}", week_num, format_total_value(total)),
                    anchor.format(" %m / %d").to_string(),
                )
            }
        }
        ChartGranularity::Year => {
            if compact {
                (
                    format!("{}:{}", anchor.format("%Y"), format_total_compact(total)),
                    anchor.format("%m/%d").to_string(),
                )
            } else {
                (
                    format!("{} : {}", anchor.format("%Y"), format_total_value(total)),
                    anchor.format(" %m / %d").to_string(),
                )
            }
        }
    }
}

fn segment_header_lines(
    layout: &ChartLayout,
    granularity: ChartGranularity,
    line_values_fn: &dyn Fn(usize) -> f64,
) -> Option<(String, String)> {
    // Build visual segments between granularity-boundary separators
    let total_cols = layout.columns.len();
    let sep_positions: Vec<usize> = layout
        .columns
        .iter()
        .enumerate()
        .filter(|(_, col)| matches!(col, ChartColumn::Separator))
        .map(|(i, _)| i)
        .collect();

    let mut segments: Vec<(usize, usize)> = Vec::new();
    if sep_positions.is_empty() {
        if total_cols > 0 {
            segments.push((0, total_cols - 1));
        }
    } else {
        if sep_positions[0] > 0 {
            segments.push((0, sep_positions[0] - 1));
        }
        for i in 0..sep_positions.len() - 1 {
            let start = sep_positions[i] + 1;
            let end = sep_positions[i + 1] - 1;
            if start <= end {
                segments.push((start, end));
            }
        }
        let start = sep_positions.last().unwrap() + 1;
        if start < total_cols {
            segments.push((start, total_cols - 1));
        }
    }

    // For each visual segment, compute midpoint, sum tokens, and pick a
    // representative timestamp. The boundary point inside a segment (e.g.
    // 00:00 inside a Day segment) is treated as the prior segment's tail in
    // the existing convention, so prefer non-boundary times when available.
    let mut segment_totals: Vec<(usize, usize, f64, DateTime<Local>)> = Vec::new();
    for &(seg_start, seg_end) in &segments {
        let mid = (seg_start + seg_end) / 2;
        let seg_width = seg_end - seg_start + 1;
        let mut total = 0.0;
        let mut repr_time: Option<DateTime<Local>> = None;

        for col_idx in seg_start..=seg_end {
            if let ChartColumn::Data { data_idx, sub_col } = &layout.columns[col_idx] {
                let time = layout.sorted_times[*data_idx];
                let is_b = granularity.is_boundary(&time);
                let dominated = repr_time.is_none_or(|r| granularity.is_boundary(&r) && !is_b);
                if dominated {
                    repr_time = Some(time);
                }
                if *sub_col == 0 {
                    total += line_values_fn(*data_idx);
                }
            }
        }

        if let Some(time) = repr_time {
            let anchor = granularity.segment_start(time);
            segment_totals.push((mid, seg_width, total, anchor));
        }
    }

    // Determine display mode based on minimum inner segment width
    let inner_min = if segments.len() > 2 {
        segments[1..segments.len() - 1]
            .iter()
            .map(|(s, e)| e - s + 1)
            .min()
            .unwrap_or(0)
    } else {
        segments.iter().map(|(s, e)| e - s + 1).min().unwrap_or(0)
    };
    let compact = inner_min < 13;
    if compact && inner_min < 7 {
        return None;
    }

    struct Placement {
        head: String,
        date: String,
        start: usize,
        len: usize,
        seg_width: usize,
    }

    let mut placements: Vec<Placement> = Vec::with_capacity(segment_totals.len());
    for (mid_col, seg_width, total, anchor) in &segment_totals {
        let (head_raw, date_raw) = segment_label(anchor, granularity, *total, compact);

        // Slash-colon centered alignment: pad shorter side so '/' sits under ':'.
        let raw_colon = head_raw.rfind(':').unwrap_or(0);
        let raw_slash = date_raw.find('/').unwrap_or(0);
        let (mut head_centered, mut date_centered) = (head_raw.clone(), date_raw.clone());
        if raw_colon > raw_slash {
            date_centered = format!("{}{}", " ".repeat(raw_colon - raw_slash), date_centered);
        } else if raw_slash > raw_colon {
            head_centered = format!("{}{}", " ".repeat(raw_slash - raw_colon), head_centered);
        }
        let centered_len = head_centered.len().max(date_centered.len());
        let centered_colon = head_centered.rfind(':').unwrap_or(0);

        let preferred_start = *mid_col as i64 - centered_colon as i64;
        let preferred_end = preferred_start + centered_len as i64;
        let fits_left = preferred_start >= 0;
        let fits_right = preferred_end <= total_cols as i64;
        let fits_chart = fits_left && fits_right;

        if fits_chart {
            // Centered placement: '/' aligns under ':'.
            let head = format!("{:<width$}", head_centered, width = centered_len);
            let date = format!("{:<width$}", date_centered, width = centered_len);
            placements.push(Placement {
                head,
                date,
                start: preferred_start as usize,
                len: centered_len,
                seg_width: *seg_width,
            });
        } else {
            // Edge placement: drop slash/colon padding, stack labels on a
            // shared left edge, and shift inward to fit within the chart.
            let head_natural = head_raw.clone();
            let date_natural = date_raw.trim_start().to_string();
            let natural_len = head_natural.len().max(date_natural.len());
            let head = format!("{:<width$}", head_natural, width = natural_len);
            let date = format!("{:<width$}", date_natural, width = natural_len);
            let start = if !fits_left {
                0
            } else {
                total_cols.saturating_sub(natural_len)
            };
            placements.push(Placement {
                head,
                date,
                start,
                len: natural_len,
                seg_width: *seg_width,
            });
        }
    }

    // Resolve label collisions: every kept label must have at least one empty
    // column between itself and every other kept label. When two labels would
    // be too close, the one belonging to the narrower underlying segment is
    // dropped so a full inner segment never loses its label to a partial edge
    // segment. Equal-width ties go to the lower index for stability.
    let mut order: Vec<usize> = (0..placements.len()).collect();
    order.sort_by(|&a, &b| {
        placements[b]
            .seg_width
            .cmp(&placements[a].seg_width)
            .then(a.cmp(&b))
    });

    let mut keep = vec![false; placements.len()];
    for &i in &order {
        let p = &placements[i];
        let p_end = p.start + p.len;
        let mut ok = true;
        for (j, q) in placements.iter().enumerate() {
            if j == i || !keep[j] {
                continue;
            }
            let q_end = q.start + q.len;
            if p.start < q_end + 1 && q.start < p_end + 1 {
                ok = false;
                break;
            }
        }
        if ok {
            keep[i] = true;
        }
    }

    let chart_width = 7 + total_cols;
    let mut line1 = " ".repeat(7);
    let mut line2 = " ".repeat(7);
    let mut prev_end = 0usize;

    for (i, p) in placements.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let start_pos = p.start.min(total_cols.saturating_sub(p.len));
        if start_pos < prev_end {
            continue;
        }
        let padding = start_pos.saturating_sub(prev_end);
        line1.push_str(&" ".repeat(padding));
        line2.push_str(&" ".repeat(padding));
        line1.push_str(&p.head);
        line2.push_str(&p.date);
        prev_end = start_pos + p.len;
    }

    line1.truncate(chart_width);
    line2.truncate(chart_width);
    Some((line1, line2))
}

fn local_from_naive_with_fallback(
    naive: chrono::NaiveDateTime,
    fallback: DateTime<Local>,
) -> DateTime<Local> {
    match Local.from_local_datetime(&naive) {
        chrono::LocalResult::Single(d) | chrono::LocalResult::Ambiguous(d, _) => d,
        chrono::LocalResult::None => fallback,
    }
}

fn add_wall_clock_minutes(anchor: DateTime<Local>, minutes: i64) -> DateTime<Local> {
    let naive = anchor.naive_local() + Duration::minutes(minutes);
    local_from_naive_with_fallback(naive, anchor + Duration::minutes(minutes))
}

fn first_x_axis_tick(
    first_time: &DateTime<Local>,
    granularity: ChartGranularity,
    tick_interval: i64,
) -> DateTime<Local> {
    let anchor = granularity.segment_start(*first_time);
    if anchor >= *first_time {
        return anchor;
    }

    let tick_interval = tick_interval.max(1);
    let elapsed_seconds = (*first_time - anchor).num_seconds().max(0);
    let interval_seconds = tick_interval * 60;
    let ticks_since = (elapsed_seconds + interval_seconds - 1) / interval_seconds;
    add_wall_clock_minutes(anchor, ticks_since * tick_interval)
}

fn print_x_axis_labels(
    layout: &ChartLayout,
    _interval_minutes: i64,
    granularity: ChartGranularity,
    pad: &str,
) {
    println!();

    let first_time = layout.sorted_times.last().unwrap(); // oldest (reversed)
    let last_time = *layout.sorted_times.first().unwrap();
    let time_span_minutes = (last_time - *first_time).num_seconds() as f64 / 60.0;
    let target_tick = time_span_minutes * 0.05;

    // Intervals span minutes up to multi-week so very wide windows still get
    // legible ticks. 1440 = 1 day, 10080 = 1 week.
    let standard_intervals = [
        15, 30, 60, 120, 180, 240, 360, 480, 720, 1440, 2880, 4320, 10080, 20160, 40320,
    ];
    let tick_interval = *standard_intervals
        .iter()
        .min_by_key(|&&x| ((x as f64 - target_tick).abs() * 1000.0) as i64)
        .unwrap_or(&60);

    let mut current_tick = first_x_axis_tick(first_time, granularity, tick_interval);

    let mut labels: Vec<String> = Vec::new();
    let mut positions: Vec<usize> = Vec::new();
    let mut used_positions: HashSet<usize> = HashSet::new();

    while current_tick <= last_time {
        let mut closest_idx = None;
        let mut min_diff = i64::MAX;

        for (i, time) in layout.sorted_times.iter().enumerate() {
            let diff = (*time - current_tick).num_seconds().abs();
            if diff < min_diff {
                min_diff = diff;
                closest_idx = Some(i);
            }
        }

        if let Some(idx) = closest_idx
            && let Some(&pos) = layout.data_to_col.get(&idx)
            && !used_positions.contains(&pos)
        {
            let label = if tick_interval < 60 {
                current_tick.format("%H:%M").to_string()
            } else if tick_interval < 1440 {
                current_tick.format("%H").to_string()
            } else {
                current_tick.format("%m/%d").to_string()
            };
            labels.push(label);
            positions.push(pos);
            used_positions.insert(pos);
        }

        current_tick = add_wall_clock_minutes(current_tick, tick_interval);
    }

    let max_label_len = labels.iter().map(|l| l.len()).max().unwrap_or(0);

    for char_idx in 0..max_label_len {
        let mut line = "       ".to_string();
        for (col_idx, col) in layout.columns.iter().enumerate() {
            match col {
                ChartColumn::Separator => line.push('|'),
                ChartColumn::Data { .. } => {
                    let mut ch = ' ';
                    for (label_idx, &pos) in positions.iter().enumerate() {
                        if col_idx == pos && char_idx < labels[label_idx].len() {
                            ch = labels[label_idx].as_bytes()[char_idx] as char;
                            break;
                        }
                    }
                    line.push(ch);
                }
            }
        }
        println!("{}{}", pad, line);
    }

    print_window_pager_hint(layout, pad);
}

/// Persistent reminder anchored to the bottom-right of the chart. The chart
/// width is the y-axis prefix ("       " = 7 cols) plus one column per
/// `layout.columns` entry.
fn print_window_pager_hint(layout: &ChartLayout, pad: &str) {
    const HINT: &str = "PgUp/PgDn: page | <-/->: move | +/-: zoom";
    let chart_width = 7 + layout.columns.len();
    let hint_visible = HINT.chars().count();
    let lead = chart_width.saturating_sub(hint_visible);
    println!(
        "{}{}{}{}{}",
        pad,
        " ".repeat(lead),
        DIM_COLOR,
        HINT,
        RESET_COLOR
    );
}

#[allow(clippy::too_many_arguments)]
fn render_grid(
    layout: &ChartLayout,
    chart_height: usize,
    max_value: i64,
    min_value: i64,
    lines: &[LineConfig],
    line_values: &HashMap<usize, Vec<f64>>,
    use_bold_for_last: bool,
    pad: &str,
) {
    let value_to_row = |value: f64| -> usize {
        if max_value == min_value {
            return 0;
        }
        ((value - min_value as f64) / (max_value - min_value) as f64 * (chart_height - 1) as f64)
            as usize
    };

    let line_rows: HashMap<usize, Vec<usize>> = line_values
        .iter()
        .map(|(&i, values)| (i, values.iter().map(|v| value_to_row(*v)).collect()))
        .collect();

    // grid[row][col] = (line_idx, char_type)
    let num_cols = layout.columns.len();
    let mut grid: Vec<Vec<Option<(usize, &'static str)>>> =
        vec![vec![None; num_cols]; chart_height];

    // Process lines in reverse order (first line draws on top)
    for line_idx in (0..lines.len()).rev() {
        let rows = &line_rows[&line_idx];

        for (col_idx, col) in layout.columns.iter().enumerate() {
            if let ChartColumn::Data {
                data_idx, sub_col, ..
            } = col
            {
                let curr_row = rows[*data_idx];
                let prev_row = if *data_idx > 0 {
                    rows[*data_idx - 1]
                } else {
                    curr_row
                };

                if *sub_col > 0 || prev_row == curr_row {
                    grid[curr_row][col_idx] = Some((line_idx, "flat"));
                } else if prev_row < curr_row {
                    grid[prev_row][col_idx] = Some((line_idx, "up_to_right"));
                    for row in grid.iter_mut().take(curr_row).skip(prev_row + 1) {
                        row[col_idx] = Some((line_idx, "vertical"));
                    }
                    grid[curr_row][col_idx] = Some((line_idx, "up_from_left"));
                } else {
                    grid[prev_row][col_idx] = Some((line_idx, "down_to_right"));
                    for row in grid.iter_mut().take(prev_row).skip(curr_row + 1) {
                        row[col_idx] = Some((line_idx, "vertical"));
                    }
                    grid[curr_row][col_idx] = Some((line_idx, "down_from_left"));
                }
            }
        }
    }

    let char_map = |char_type: &str| -> char {
        match char_type {
            "flat" => '\u{2500}',           // ─
            "up_from_left" => '\u{256D}',   // ╭
            "down_from_left" => '\u{2570}', // ╰
            "down_to_right" => '\u{256E}',  // ╮
            "up_to_right" => '\u{256F}',    // ╯
            "vertical" => '\u{2502}',       // │
            _ => '\u{2500}',
        }
    };

    let char_map_bold = |char_type: &str| -> char {
        match char_type {
            "flat" => '\u{2501}',           // ━
            "up_from_left" => '\u{250F}',   // ┏
            "down_from_left" => '\u{2517}', // ┗
            "down_to_right" => '\u{2513}',  // ┓
            "up_to_right" => '\u{251B}',    // ┛
            "vertical" => '\u{2503}',       // ┃
            _ => '\u{2501}',
        }
    };

    // Draw chart from top to bottom
    for row in (0..chart_height).rev() {
        let y_val = min_value as f64
            + (max_value - min_value) as f64 * row as f64 / (chart_height - 1) as f64;
        let y_label = format!("{} |", format_y_axis_value(y_val));

        let mut line_str = String::new();
        for (col_idx, col) in layout.columns.iter().enumerate() {
            match col {
                ChartColumn::Separator => line_str.push('|'),
                ChartColumn::Data { .. } => {
                    if let Some((line_idx, char_type)) = grid[row][col_idx] {
                        let is_bold = use_bold_for_last && line_idx == lines.len() - 1;
                        let ch = if is_bold {
                            char_map_bold(char_type)
                        } else {
                            char_map(char_type)
                        };
                        let color = &lines[line_idx].color;
                        line_str.push_str(color);
                        line_str.push(ch);
                        line_str.push_str(RESET_COLOR);
                    } else {
                        line_str.push(' ');
                    }
                }
            }
        }
        println!("{}{}{}", pad, y_label, line_str);
    }

    // X-axis
    let mut x_axis = String::new();
    for col in &layout.columns {
        match col {
            ChartColumn::Separator => x_axis.push('\u{2534}'), // ┴
            ChartColumn::Data { .. } => x_axis.push('\u{2500}'), // ─
        }
    }
    println!("{}      \u{2514}{}", pad, x_axis); // └
}

struct LineConfig {
    model: String,
    token_type: String,
    color: String,
    label: String,
}

/// Print a multi-line chart with multiple lines (models x token types).
#[allow(clippy::too_many_arguments)]
pub fn print_multi_line_chart(
    time_series: &ModelTimeSeries,
    height: usize,
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    chart_type: &str,
    show_x_axis: bool,
    target_width: Option<usize>,
    interval_minutes: i64,
    granularity: ChartGranularity,
    vendor: &str,
    included_models: Option<&HashSet<String>>,
    show_legend: bool,
    terminal_width: Option<usize>,
) {
    let layout = build_chart_layout(
        range_start,
        range_end,
        interval_minutes,
        granularity,
        target_width,
    );

    if layout.sorted_times.len() < 2 {
        println!("Not enough data points for chart.");
        return;
    }

    let (token_types, type_labels, chart_title) = if chart_type == "io" {
        (
            vec!["input", "output"],
            HashMap::from([("input", "Input"), ("output", "Output")]),
            "Models Input / Output Token Consumption".to_string(),
        )
    } else {
        let labels = match vendor {
            "codex" => HashMap::from([
                ("cache_read", "Cache Read In"),
                ("cache_creation", "Reasoning Out"),
            ]),
            "gemini" => HashMap::from([
                ("cache_read", "Cache Read In"),
                ("cache_creation", "Thinking Out"),
            ]),
            _ => HashMap::from([
                ("cache_read", "Cache Read In"),
                ("cache_creation", "Cache Create In"),
            ]),
        };
        let title = match vendor {
            "codex" => "Models Cache Read Input / Reasoning Output Token Consumption",
            "gemini" => "Models Cache Read Input / Thinking Output Token Consumption",
            _ => "Models Cache Read Input / Cache Creation Input Token Consumption",
        };
        (
            vec!["cache_read", "cache_creation"],
            labels,
            title.to_string(),
        )
    };

    // Collect all models
    let mut all_models: HashSet<String> = HashSet::new();
    for time in &layout.sorted_times {
        if let Some(model_map) = time_series.get(time) {
            all_models.extend(model_map.keys().cloned());
        }
    }
    if let Some(included) = included_models {
        all_models.retain(|m| included.contains(m));
    }

    let config = model_config();
    let known_set: HashSet<&str> = config.iter().map(|(m, _, _)| *m).collect();

    let mut known_models: Vec<String> = all_models
        .iter()
        .filter(|m| known_set.contains(m.as_str()))
        .cloned()
        .collect();
    let mut other_models: Vec<String> = all_models
        .iter()
        .filter(|m| !known_set.contains(m.as_str()))
        .cloned()
        .collect();

    known_models.sort_by_key(|m| model_order(m).unwrap_or(99));
    other_models.sort();

    let all_models_sorted: Vec<String> = known_models.into_iter().chain(other_models).collect();

    // Build line configurations
    let mut lines: Vec<LineConfig> = Vec::new();
    for (model_idx, model) in all_models_sorted.iter().enumerate() {
        if known_set.contains(model.as_str()) {
            let short_label = model_short_name(model).unwrap_or(model);
            let model_short = short_label.to_lowercase();
            let model_prefix = model_short.split_whitespace().next().unwrap_or("unknown");

            for token_type in &token_types {
                let color_suffix = if *token_type == "input" || *token_type == "cache_read" {
                    "input"
                } else {
                    "output"
                };
                let color_key = format!("{}_{}", model_prefix, color_suffix);
                lines.push(LineConfig {
                    model: model.clone(),
                    token_type: token_type.to_string(),
                    color: line_color(&color_key).to_string(),
                    label: format!("{} {}", short_label, type_labels[token_type]),
                });
            }
        } else {
            let short_label = get_short_model_name_for_chart(model);
            let color_idx = model_idx % 6;
            for token_type in &token_types {
                let color_suffix = if *token_type == "input" || *token_type == "cache_read" {
                    "input"
                } else {
                    "output"
                };
                let color_key = format!("model{}_{}", color_idx, color_suffix);
                lines.push(LineConfig {
                    model: model.clone(),
                    token_type: token_type.to_string(),
                    color: line_color(&color_key).to_string(),
                    label: format!("{} {}", short_label, type_labels[token_type]),
                });
            }
        }
    }

    // Calculate values for each line
    let mut line_values: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut all_values: Vec<f64> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let values: Vec<f64> = layout
            .sorted_times
            .iter()
            .map(|time| {
                time_series
                    .get(time)
                    .and_then(|model_map| model_map.get(&line.model))
                    .map(|breakdown| match line.token_type.as_str() {
                        "input" => breakdown.input,
                        "output" => breakdown.output,
                        "cache_creation" => breakdown.cache_creation,
                        "cache_read" => breakdown.cache_read,
                        _ => 0.0,
                    })
                    .unwrap_or(0.0)
            })
            .collect();
        all_values.extend_from_slice(&values);
        line_values.insert(i, values);
    }

    let max_value_raw = all_values.iter().cloned().fold(0.0f64, f64::max);
    let max_value_raw = if max_value_raw == 0.0 {
        1.0
    } else {
        max_value_raw
    };
    let max_value = round_to_nice(max_value_raw, true);
    let min_value = 0i64;
    let max_value = if max_value == min_value {
        min_value + 5000
    } else {
        max_value
    };

    // Print title
    let chart_width = layout.columns.len() + 7;
    let tw = terminal_width.unwrap_or(chart_width);
    let pad = center_pad(tw, chart_width);
    if !show_x_axis {
        println!();
    }
    println!("{}{:^width$}", pad, chart_title, width = chart_width);
    println!("{}{}", pad, "=".repeat(chart_width));

    // Segment header — sum all lines per data index for the totals shown
    // next to each segment label.
    print_segment_header(
        &layout,
        granularity,
        &|data_idx| {
            lines
                .iter()
                .enumerate()
                .map(|(i, _)| line_values[&i][data_idx])
                .sum::<f64>()
        },
        &pad,
    );

    render_grid(
        &layout,
        height,
        max_value,
        min_value,
        &lines,
        &line_values,
        false,
        &pad,
    );

    if show_x_axis {
        print_x_axis_labels(&layout, interval_minutes, granularity, &pad);
    }

    if show_legend && !lines.is_empty() {
        let legend_parts: Vec<String> = lines
            .iter()
            .map(|l| format!("{}\u{2500}{} {}", l.color, RESET_COLOR, l.label))
            .collect();
        println!("{}Legend: {}", pad, legend_parts.join("  "));
    }
}

/// Print a vendor comparison chart.
#[allow(clippy::too_many_arguments)]
pub fn print_vendor_comparison_chart(
    time_series: &VendorTimeSeries,
    height: usize,
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    target_width: Option<usize>,
    interval_minutes: i64,
    granularity: ChartGranularity,
    show_legend: bool,
    terminal_width: Option<usize>,
) {
    let layout = build_chart_layout(
        range_start,
        range_end,
        interval_minutes,
        granularity,
        target_width,
    );

    if layout.sorted_times.len() < 2 {
        println!("Not enough data points for chart.");
        return;
    }

    // Collect all vendors
    let mut all_vendors: HashSet<String> = HashSet::new();
    for time in &layout.sorted_times {
        if let Some(vendor_map) = time_series.get(time) {
            all_vendors.extend(vendor_map.keys().cloned());
        }
    }

    let vendor_order = ["Claude", "Codex", "Gemini"];
    let mut vendors_sorted: Vec<String> = vendor_order
        .iter()
        .filter(|v| all_vendors.contains(**v))
        .map(|v| v.to_string())
        .collect();
    let mut remaining: Vec<&String> = all_vendors
        .iter()
        .filter(|v| !vendor_order.contains(&v.as_str()))
        .collect();
    remaining.sort_unstable();
    for v in remaining {
        vendors_sorted.push(v.clone());
    }

    // Add "All" as last entry
    if !vendors_sorted.is_empty() {
        vendors_sorted.push("All".to_string());
    }

    // Build vendor data
    let mut vendor_data: HashMap<String, Vec<f64>> = HashMap::new();
    for vendor in &vendors_sorted {
        if vendor == "All" {
            continue;
        }
        let values: Vec<f64> = layout
            .sorted_times
            .iter()
            .map(|time| {
                time_series
                    .get(time)
                    .and_then(|vm| vm.get(vendor))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        vendor_data.insert(vendor.clone(), values);
    }

    // Calculate "All" as sum
    let all_values: Vec<f64> = (0..layout.sorted_times.len())
        .map(|i| {
            vendors_sorted
                .iter()
                .filter(|v| *v != "All")
                .map(|v| vendor_data.get(v).map(|vals| vals[i]).unwrap_or(0.0))
                .sum()
        })
        .collect();
    if vendors_sorted.iter().any(|v| v == "All") {
        vendor_data.insert("All".to_string(), all_values.clone());
    }

    // Find max value
    let max_value_raw = vendor_data
        .values()
        .flat_map(|vals| vals.iter())
        .cloned()
        .fold(0.0f64, f64::max);
    let max_value_raw = if max_value_raw == 0.0 {
        1.0
    } else {
        max_value_raw
    };
    let max_value = round_to_nice(max_value_raw, true);
    let min_value = 0i64;
    let max_value = if max_value == min_value {
        min_value + 5000
    } else {
        max_value
    };

    // Build line configs
    let lines: Vec<LineConfig> = vendors_sorted
        .iter()
        .map(|v| LineConfig {
            model: v.clone(),
            token_type: String::new(),
            color: vendor_color(v).to_string(),
            label: v.clone(),
        })
        .collect();

    let line_values: HashMap<usize, Vec<f64>> = vendors_sorted
        .iter()
        .enumerate()
        .map(|(i, v)| (i, vendor_data[v].clone()))
        .collect();

    let chart_title = "Total Token Consumption by Vendor";
    let chart_width = layout.columns.len() + 7;
    let tw = terminal_width.unwrap_or(chart_width);
    let pad = center_pad(tw, chart_width);
    println!();
    println!("{}{:^width$}", pad, chart_title, width = chart_width);
    println!("{}{}", pad, "=".repeat(chart_width));

    // Segment header using "All" totals
    print_segment_header(&layout, granularity, &|data_idx| all_values[data_idx], &pad);

    render_grid(
        &layout,
        height,
        max_value,
        min_value,
        &lines,
        &line_values,
        true,
        &pad,
    );

    print_x_axis_labels(&layout, interval_minutes, granularity, &pad);

    if show_legend && !vendors_sorted.is_empty() {
        // Calculate vendor totals and percentages
        let vendor_totals: HashMap<String, f64> = vendors_sorted
            .iter()
            .filter(|v| *v != "All")
            .map(|v| (v.clone(), vendor_data[v].iter().sum::<f64>()))
            .collect();
        let grand_total: f64 = vendor_totals.values().sum();

        let mut legend_vendors: Vec<&String> =
            vendors_sorted.iter().filter(|v| *v != "All").collect();
        legend_vendors.sort_by(|a, b| {
            vendor_totals[*b]
                .partial_cmp(&vendor_totals[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });

        let mut legend_items: Vec<String> = Vec::new();
        for v in legend_vendors {
            let color = vendor_color(v);
            let pct = if grand_total > 0.0 {
                vendor_totals[v] / grand_total * 100.0
            } else {
                0.0
            };
            legend_items.push(format!(
                "{}\u{2500}{} {}({:.1}%)",
                color, RESET_COLOR, v, pct
            ));
        }
        let all_color = vendor_color("All");
        legend_items.push(format!("{}\u{2501}{} All(100%)", all_color, RESET_COLOR));
        println!("{}Legend: {}", pad, legend_items.join("  "));
    }
}
