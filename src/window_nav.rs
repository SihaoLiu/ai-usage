//! Time-window navigation and chart sizing for the display layer.

use chrono::{DateTime, Duration, Local};
use crossterm::event::{Event, KeyCode, KeyEvent};
use std::collections::VecDeque;

use crate::charts;
use crate::get_terminal_size;
use crate::time_utils::TimeWindow;

/// Calculate chart height(s) that fit within the terminal.
/// For one tool: returns per-chart height (2 charts displayed).
/// For all tools: returns the single chart height.
/// Also returns whether the layout fits (true) or overflows (false).
pub(crate) fn calculate_chart_height(
    is_monitor_mode: bool,
    table_printed: bool,
    num_models: usize,
    is_all_tool: bool,
) -> (usize, bool) {
    let (_, height) = get_terminal_size();
    let th = height as usize;

    // Header: "Calculating...", "Showing data...", "Monitor mode..." (or 2 if --once)
    let header_lines = if is_monitor_mode { 3 } else { 2 };

    // Table: 1 blank + 1 title + 1 =border + 1 header + 1 -border
    //        + num_models rows + 1 -border + 1 TOTAL + 1 Cost + 1 =border
    //        + 1 cost summary + 1 insight summary = 11 + num_models
    let table_lines = if table_printed { 11 + num_models } else { 0 };

    // Time span info: 1 line + 1 blank
    let time_span_lines = 2;

    // Per-chart overhead: 1 blank + 1 title + 1 =border + 2 daily_header
    //                     + 1 x-axis bottom line + 1 legend = 7
    // X-axis labels: typically 2 lines (blank + label chars), up to 5
    let chart_overhead = 7;
    let x_axis_label_lines = 4; // blank + typically 2 label rows + 1 pager hint

    // Monitor prompt: 1 blank + 1 version + 1 separator + 1 "> " = 4
    let prompt_lines = if is_monitor_mode { 4 } else { 0 };

    let min_chart = 5usize;

    if is_all_tool {
        // Single chart
        let fixed = header_lines
            + table_lines
            + time_span_lines
            + chart_overhead
            + x_axis_label_lines
            + prompt_lines;
        let available = th.saturating_sub(fixed);
        let chart_height = available.max(min_chart).min(60);
        let fits = th >= fixed + min_chart;
        (chart_height, fits)
    } else {
        // Two charts: chart1 (io, no x-axis labels) + chart2 (cache, with x-axis labels)
        let chart1_fixed = chart_overhead;
        let chart2_fixed = chart_overhead + x_axis_label_lines;
        let fixed = header_lines
            + table_lines
            + time_span_lines
            + chart1_fixed
            + chart2_fixed
            + prompt_lines;
        let available = th.saturating_sub(fixed);
        let per_chart = (available / 2).max(min_chart).min(60);
        let fits = th >= fixed + min_chart * 2;
        (per_chart, fits)
    }
}

pub(crate) fn calculate_optimal_interval_minutes(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    target_width: usize,
    granularity: charts::ChartGranularity,
) -> f64 {
    let total_minutes = ((*range_end - *range_start).num_seconds() as f64 / 60.0).max(1.0);
    let min_interval = total_minutes / 100.0;
    let y_axis_width = 7.0;
    let separator_estimate =
        estimate_chart_separator_count(range_start, range_end, granularity) as f64;
    let chart_width = (target_width as f64 - y_axis_width - separator_estimate).max(50.0);
    let terminal_interval = total_minutes / chart_width;
    min_interval.max(terminal_interval)
}

pub(crate) fn estimate_chart_separator_count(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
    granularity: charts::ChartGranularity,
) -> usize {
    let span_minutes = ((*range_end - *range_start).num_seconds() as f64 / 60.0).max(1.0);
    let segment_minutes = match granularity {
        charts::ChartGranularity::Hour => 60.0,
        charts::ChartGranularity::Day => 24.0 * 60.0,
        charts::ChartGranularity::Week => 7.0 * 24.0 * 60.0,
        charts::ChartGranularity::Month => 30.0 * 24.0 * 60.0,
        charts::ChartGranularity::Year => 365.0 * 24.0 * 60.0,
    };
    (span_minutes / segment_minutes).ceil().max(1.0) as usize
}

pub(crate) fn display_chart_granularity(
    range_start: &DateTime<Local>,
    range_end: &DateTime<Local>,
) -> charts::ChartGranularity {
    let span_minutes = ((*range_end - *range_start).num_seconds() / 60).max(1);
    charts::ChartGranularity::from_span_minutes(span_minutes)
}

pub(crate) fn is_current_rolling_days_preset(window: &TimeWindow, days: i64) -> bool {
    window.is_rolling_days(days)
}

pub(crate) fn round_to_nice_interval(optimal: f64) -> i64 {
    let nice = [
        1i64, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320, 5760, 10080, 20160, 40320,
        80640,
    ];
    for &n in &nice {
        if n as f64 >= optimal {
            return n;
        }
    }
    *nice.last().unwrap()
}

#[derive(Clone, Copy)]
pub(crate) enum IntervalSlideDirection {
    Older,
    Newer,
}

pub(crate) fn display_interval_minutes_for_window(
    window: &TimeWindow,
    now: DateTime<Local>,
    target_width: usize,
) -> i64 {
    let (range_start, range_end) = window.bounds(now);
    let granularity = display_chart_granularity(&range_start, &range_end);
    let optimal =
        calculate_optimal_interval_minutes(&range_start, &range_end, target_width, granularity);
    round_to_nice_interval(optimal)
}

pub(crate) fn slide_window_by_display_interval(
    window: &TimeWindow,
    now: DateTime<Local>,
    target_width: usize,
    direction: IntervalSlideDirection,
) -> Option<TimeWindow> {
    let interval_minutes = display_interval_minutes_for_window(window, now, target_width);
    let step = Duration::minutes(interval_minutes.max(1));
    match direction {
        IntervalSlideDirection::Older => window.slide_back_by(now, step),
        IntervalSlideDirection::Newer => window.slide_forward_by(now, step),
    }
}

pub(crate) fn apply_interval_slide_directions<I>(
    window: &TimeWindow,
    now: DateTime<Local>,
    target_width: usize,
    directions: I,
) -> Option<TimeWindow>
where
    I: IntoIterator<Item = IntervalSlideDirection>,
{
    let mut current = window.clone();
    let mut changed = false;
    for direction in directions {
        if let Some(next) = slide_window_by_display_interval(&current, now, target_width, direction)
        {
            current = next;
            changed = true;
        }
    }
    changed.then_some(current)
}

pub(crate) fn interval_slide_direction_for_event(event: &Event) -> Option<IntervalSlideDirection> {
    match event {
        Event::Key(KeyEvent {
            code: KeyCode::Left,
            ..
        }) => Some(IntervalSlideDirection::Newer),
        Event::Key(KeyEvent {
            code: KeyCode::Right,
            ..
        }) => Some(IntervalSlideDirection::Older),
        _ => None,
    }
}

pub(crate) fn collect_interval_slide_directions(
    pending_events: &mut VecDeque<Event>,
    first_direction: IntervalSlideDirection,
) -> Vec<IntervalSlideDirection> {
    let mut directions = vec![first_direction];
    while crossterm::event::poll(std::time::Duration::ZERO).unwrap_or(false) {
        match crossterm::event::read() {
            Ok(event) => {
                if let Some(direction) = interval_slide_direction_for_event(&event) {
                    directions.push(direction);
                } else {
                    pending_events.push_back(event);
                    break;
                }
            }
            Err(_) => break,
        }
    }
    directions
}

pub(crate) fn get_chart_target_width() -> usize {
    let (width, _) = get_terminal_size();
    (width as f64 * 0.99) as usize
}

pub(crate) fn showing_data_line(window: &TimeWindow, now: DateTime<Local>) -> String {
    format!("Showing data from {}", window.display_label(now))
}

pub(crate) fn parse_time_window_command(
    command: &str,
    current_window: &TimeWindow,
    now: DateTime<Local>,
) -> Option<Result<TimeWindow, String>> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    match parts.as_slice() {
        ["date", date] => Some(TimeWindow::from_date(date)),
        ["date"] => Some(Err("Usage: date YYYY-MM-DD".to_string())),
        ["range", start, end] => Some(TimeWindow::from_range(start, end)),
        ["range"] | ["range", _] => Some(Err("Usage: range YYYY-MM-DD YYYY-MM-DD".to_string())),
        ["latest"] | ["last"] => Some(Ok(current_window.follow_latest(now))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use chrono::TimeZone;

    #[test]
    fn date_command_selects_single_inclusive_day() {
        let command = parse_time_window_command(
            "date 2026-05-07",
            &TimeWindow::rolling_days(3),
            Local::now(),
        )
        .expect("recognized command")
        .expect("valid date");
        let TimeWindow::ExplicitRange {
            start,
            end,
            projection_days,
            ..
        } = command
        else {
            panic!("date command should create an explicit window");
        };

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-07 00:00:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-07 23:59:59"
        );
        assert_eq!(projection_days, 1.0);
    }

    #[test]
    fn range_command_selects_inclusive_date_span() {
        let command = parse_time_window_command(
            "range 2026-05-01 2026-05-07",
            &TimeWindow::rolling_days(3),
            Local::now(),
        )
        .expect("recognized command")
        .expect("valid range");

        assert_eq!(command.projection_days(Local::now()), 7.0);
    }

    #[test]
    fn interval_slide_older_moves_rolling_window_by_display_interval() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        let slid =
            slide_window_by_display_interval(&window, now, 160, IntervalSlideDirection::Older)
                .expect("slide older");
        let (start, end) = slid.bounds(now);

        assert_eq!(end, now - chrono::Duration::hours(1));
        assert_eq!(
            start,
            now - chrono::Duration::days(3) - chrono::Duration::hours(1)
        );
        assert_eq!(slid.page_step(), chrono::Duration::days(3));
    }

    #[test]
    fn interval_slide_newer_clamps_to_present() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);
        let older =
            slide_window_by_display_interval(&window, now, 160, IntervalSlideDirection::Older)
                .expect("slide older");

        let newer =
            slide_window_by_display_interval(&older, now, 160, IntervalSlideDirection::Newer)
                .expect("slide newer");
        let (start, end) = newer.bounds(now);

        assert_eq!(end, now);
        assert_eq!(start, now - chrono::Duration::days(3));
    }

    #[test]
    fn interval_slide_newer_on_current_window_is_noop() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        assert!(
            slide_window_by_display_interval(&window, now, 160, IntervalSlideDirection::Newer)
                .is_none()
        );
    }

    #[test]
    fn batched_interval_slide_matches_repeated_right_arrow_slides() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);
        let target_width = 160;
        let directions = [IntervalSlideDirection::Older; 5];

        let batched = apply_interval_slide_directions(&window, now, target_width, directions)
            .expect("batched slide");
        let mut sequential = window.clone();
        for _ in 0..directions.len() {
            sequential = slide_window_by_display_interval(
                &sequential,
                now,
                target_width,
                IntervalSlideDirection::Older,
            )
            .expect("sequential slide");
        }

        assert_eq!(batched.bounds(now), sequential.bounds(now));
        assert_eq!(batched.page_step(), sequential.page_step());
    }

    #[test]
    fn batched_interval_slide_preserves_noop_then_right_arrow_slide() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);
        let target_width = 160;
        let directions = [IntervalSlideDirection::Newer, IntervalSlideDirection::Older];

        let batched = apply_interval_slide_directions(&window, now, target_width, directions)
            .expect("batched slide");
        let expected = slide_window_by_display_interval(
            &window,
            now,
            target_width,
            IntervalSlideDirection::Older,
        )
        .expect("right arrow slide");

        assert_eq!(batched.bounds(now), expected.bounds(now));
        assert_eq!(batched.page_step(), expected.page_step());
    }

    #[test]
    fn display_interval_scales_past_daily_for_multi_month_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2025-11-03", "2026-05-14").expect("range");

        let interval = display_interval_minutes_for_window(&window, now, 160);

        assert!(interval > 1440);
        assert_eq!(interval, 2880);
    }

    #[test]
    fn display_interval_uses_month_granularity_for_quarter_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2026-02-14", "2026-05-14").expect("range");
        let (range_start, range_end) = window.bounds(now);

        let granularity = display_chart_granularity(&range_start, &range_end);
        let interval = display_interval_minutes_for_window(&window, now, 160);

        assert_eq!(granularity, charts::ChartGranularity::Month);
        assert_eq!(interval, 1440);
    }

    #[test]
    fn display_interval_scales_past_daily_for_year_granularity_windows() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 14, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2024-01-01", "2026-05-14").expect("range");

        let interval = display_interval_minutes_for_window(&window, now, 160);

        assert!(interval > 1440);
        assert_eq!(interval, 20160);
    }

    #[test]
    fn latest_command_returns_to_rolling_days() {
        let current = TimeWindow::rolling_days(5);
        let command = parse_time_window_command("latest", &current, Local::now())
            .expect("recognized command")
            .expect("valid latest command");

        assert!(command.is_rolling_days(5));
    }

    #[test]
    fn rolling_days_preset_is_current_only_for_matching_rolling_window() {
        assert!(is_current_rolling_days_preset(
            &TimeWindow::rolling_days(30),
            30
        ));
        assert!(!is_current_rolling_days_preset(
            &TimeWindow::rolling_days(7),
            30
        ));
    }

    #[test]
    fn zoomed_rolling_days_window_is_not_current_preset() {
        let now = Local
            .with_ymd_and_hms(2026, 6, 15, 12, 0, 0)
            .single()
            .expect("fixed now");
        let zoomed = TimeWindow::rolling_days(30).zoom_in(now).expect("zoom in");

        assert!(!is_current_rolling_days_preset(&zoomed, 30));
    }
}
