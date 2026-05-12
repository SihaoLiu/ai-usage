use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc};

/// Time range selected for usage aggregation and display.
#[derive(Clone, Debug)]
pub enum TimeWindow {
    RollingDays {
        days: i64,
    },
    ExplicitRange {
        start: DateTime<Local>,
        end: DateTime<Local>,
        projection_days: f64,
        /// How far to translate the window when the user pages back or
        /// forward. Date-only ranges use whole-day steps so paging stays
        /// aligned to midnight; date-time ranges use the exact window span.
        page_step: Duration,
    },
}

impl TimeWindow {
    pub fn rolling_days(days: i64) -> Self {
        Self::RollingDays { days }
    }

    pub fn from_date(input: &str) -> Result<Self, String> {
        let date = NaiveDate::parse_from_str(input, "%Y-%m-%d")
            .map_err(|_| "Usage: date YYYY-MM-DD".to_string())?;
        let start = local_midnight(date)?;
        let end = local_day_end(date)?;
        Ok(Self::ExplicitRange {
            start,
            end,
            projection_days: 1.0,
            page_step: Duration::days(1),
        })
    }

    /// Build an explicit range from two endpoints. The caller may pass them
    /// in either chronological order; this function picks the earlier as the
    /// start and the later as the end so the user never has to remember
    /// which argument comes first. For date-only inputs the bounds expand to
    /// cover the entire local day (midnight to 23:59:59).
    pub fn from_range(first_input: &str, second_input: &str) -> Result<Self, String> {
        let first = parse_range_endpoint(first_input)?;
        let second = parse_range_endpoint(second_input)?;

        let (earlier, later) = if first.start <= second.start {
            (first, second)
        } else {
            (second, first)
        };

        let start = earlier.start;
        let end = later.end;

        let (projection_days, page_step) = if earlier.is_date_only && later.is_date_only {
            let s_date = start.date_naive();
            let e_date = end.date_naive();
            let n = e_date
                .signed_duration_since(s_date)
                .num_days()
                .saturating_add(1)
                .max(1);
            (n as f64, Duration::days(n))
        } else {
            let span = end - start;
            let proj = ((end - start).num_seconds() as f64 / 86_400.0).max(1.0 / 1440.0);
            let step = if span <= Duration::zero() {
                Duration::seconds(1)
            } else {
                span
            };
            (proj, step)
        };

        Ok(Self::ExplicitRange {
            start,
            end,
            projection_days,
            page_step,
        })
    }

    pub fn bounds(&self, now: DateTime<Local>) -> (DateTime<Local>, DateTime<Local>) {
        match self {
            Self::RollingDays { days } => (now - Duration::days(*days), now),
            Self::ExplicitRange { start, end, .. } => (*start, *end),
        }
    }

    pub fn projection_days(&self, _now: DateTime<Local>) -> f64 {
        match self {
            Self::RollingDays { days } => (*days).max(1) as f64,
            Self::ExplicitRange {
                projection_days, ..
            } => *projection_days,
        }
    }

    /// Span used for PageUp/PageDown paging. Always positive.
    pub fn page_step(&self) -> Duration {
        match self {
            Self::RollingDays { days } => Duration::days((*days).max(1)),
            Self::ExplicitRange { page_step, .. } => *page_step,
        }
    }

    pub fn display_label(&self, _now: DateTime<Local>) -> String {
        match self {
            Self::RollingDays { days } => format!("last {} days", days),
            Self::ExplicitRange { start, end, .. } => {
                format!(
                    "{} to {}",
                    start.format("%Y-%m-%d %H:%M"),
                    end.format("%Y-%m-%d %H:%M")
                )
            }
        }
    }

    /// Translate the window backward by `page_step`. Always returns an
    /// `ExplicitRange` so the new bounds are no longer anchored to `now`.
    pub fn slide_back(&self, now: DateTime<Local>) -> Option<Self> {
        self.slide_back_by(now, self.page_step())
    }

    /// Translate the window backward by an arbitrary positive duration while
    /// keeping the PageUp/PageDown step tied to the original window width.
    pub fn slide_back_by(&self, now: DateTime<Local>, step: Duration) -> Option<Self> {
        if step <= Duration::zero() {
            return None;
        }
        let (start, end) = self.bounds(now);
        Some(Self::ExplicitRange {
            start: start - step,
            end: end - step,
            projection_days: self.projection_days(now),
            page_step: self.page_step(),
        })
    }

    /// Translate the window forward by `page_step`. Clamps so the new end
    /// never exceeds `now`, which makes paging into the future a no-op
    /// (returns `None`) when the window already touches the present.
    pub fn slide_forward(&self, now: DateTime<Local>) -> Option<Self> {
        self.slide_forward_by(now, self.page_step())
    }

    /// Translate the window forward by an arbitrary positive duration while
    /// preserving the original window width and PageUp/PageDown step.
    pub fn slide_forward_by(&self, now: DateTime<Local>, step: Duration) -> Option<Self> {
        if step <= Duration::zero() {
            return None;
        }
        let (start, end) = self.bounds(now);
        let mut new_start = start + step;
        let mut new_end = end + step;
        if new_end > now {
            let overshoot = new_end - now;
            new_end -= overshoot;
            new_start -= overshoot;
        }
        if new_start == start && new_end == end {
            return None;
        }
        Some(Self::ExplicitRange {
            start: new_start,
            end: new_end,
            projection_days: self.projection_days(now),
            page_step: self.page_step(),
        })
    }

    pub fn zoom_in(&self, now: DateTime<Local>) -> Option<Self> {
        let span = self.span(now)?;
        let min_span = min_zoom_span();
        if span <= min_span {
            return None;
        }
        let new_span = (span / 2).max(min_span);
        self.zoom_to_span(now, new_span)
    }

    pub fn zoom_out(&self, now: DateTime<Local>) -> Option<Self> {
        let span = self.span(now)?;
        self.zoom_to_span(now, span * 2)
    }

    fn span(&self, now: DateTime<Local>) -> Option<Duration> {
        let (start, end) = self.bounds(now);
        let span = end - start;
        if span <= Duration::zero() {
            return None;
        }
        Some(span)
    }

    fn zoom_to_span(&self, now: DateTime<Local>, new_span: Duration) -> Option<Self> {
        if new_span <= Duration::zero() {
            return None;
        }
        let (start, end) = self.bounds(now);
        let span = end - start;

        let (new_start, new_end) = if matches!(self, Self::RollingDays { .. }) {
            (now - new_span, now)
        } else {
            let center = start + span / 2;
            let mut new_start = center - new_span / 2;
            let mut new_end = new_start + new_span;
            if new_end > now {
                let overshoot = new_end - now;
                new_start -= overshoot;
                new_end -= overshoot;
            }
            (new_start, new_end)
        };

        Some(Self::ExplicitRange {
            start: new_start,
            end: new_end,
            projection_days: projection_days_for_span(new_span),
            page_step: new_span,
        })
    }
}

fn min_zoom_span() -> Duration {
    Duration::minutes(1)
}

fn projection_days_for_span(span: Duration) -> f64 {
    (span.num_seconds() as f64 / 86_400.0).max(1.0 / 1440.0)
}

fn local_from_naive(naive: NaiveDateTime) -> Result<DateTime<Local>, String> {
    match Local.from_local_datetime(&naive) {
        chrono::LocalResult::Single(dt) => Ok(dt),
        chrono::LocalResult::Ambiguous(dt, _) => Ok(dt),
        chrono::LocalResult::None => Err("Local time does not exist in this timezone.".to_string()),
    }
}

fn local_midnight(date: NaiveDate) -> Result<DateTime<Local>, String> {
    let naive = date
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| "Invalid date.".to_string())?;
    local_from_naive(naive)
}

fn local_day_end(date: NaiveDate) -> Result<DateTime<Local>, String> {
    let next_date = date
        .succ_opt()
        .ok_or_else(|| "Date is out of supported range.".to_string())?;
    local_midnight(next_date)?
        .checked_sub_signed(Duration::nanoseconds(1))
        .ok_or_else(|| "Date is out of supported range.".to_string())
}

/// Span covered by a single range endpoint. Date-only inputs expand to the
/// full local day so callers can pick the outermost bounds regardless of
/// which argument was typed first; date-time inputs collapse to a single
/// instant (`start == end`).
struct EndpointSpan {
    start: DateTime<Local>,
    end: DateTime<Local>,
    is_date_only: bool,
}

fn parse_range_endpoint(input: &str) -> Result<EndpointSpan, String> {
    if let Ok(date) = NaiveDate::parse_from_str(input, "%Y-%m-%d") {
        return Ok(EndpointSpan {
            start: local_midnight(date)?,
            end: local_day_end(date)?,
            is_date_only: true,
        });
    }

    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(input, fmt) {
            let dt = local_from_naive(naive)?;
            return Ok(EndpointSpan {
                start: dt,
                end: dt,
                is_date_only: false,
            });
        }
    }

    Err("Use YYYY-MM-DD or YYYY-MM-DDTHH:MM.".to_string())
}

/// Parse an ISO timestamp string to a local DateTime.
/// Handles formats like "2025-12-11T23:18:08.351Z" and "2025-12-11T23:18:08+00:00".
pub fn parse_timestamp(timestamp_str: &str) -> Option<DateTime<Local>> {
    // Try parsing as RFC3339/ISO8601 with timezone
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        return Some(dt.with_timezone(&Local));
    }
    // Handle 'Z' suffix (should be caught by rfc3339 but just in case)
    let s = timestamp_str.replace('Z', "+00:00");
    if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
        return Some(dt.with_timezone(&Local));
    }
    // Try parsing with milliseconds but no timezone (assume UTC)
    if let Ok(ndt) = NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%dT%H:%M:%S%.f")
        && let Some(utc_dt) = Utc.from_local_datetime(&ndt).single()
    {
        return Some(utc_dt.with_timezone(&Local));
    }
    // Try epoch milliseconds (numeric string)
    if let Ok(ms) = timestamp_str.parse::<i64>()
        && let Some(dt) = DateTime::from_timestamp_millis(ms)
    {
        return Some(dt.with_timezone(&Local));
    }
    None
}

/// Round a DateTime down to the nearest interval boundary.
/// DST-safe: uses `from_local_datetime` to handle spring-forward gaps.
pub fn to_interval(dt: &DateTime<Local>, interval_minutes: i64) -> DateTime<Local> {
    let total_minutes = (dt.hour() as i64) * 60 + (dt.minute() as i64);
    let interval_start = (total_minutes / interval_minutes) * interval_minutes;
    let hour = (interval_start / 60) as u32;
    let minute = (interval_start % 60) as u32;

    let target_naive = dt.date_naive().and_hms_opt(hour, minute, 0).unwrap();
    match Local.from_local_datetime(&target_naive) {
        chrono::LocalResult::Single(d) => d,
        chrono::LocalResult::Ambiguous(d, _) => d,
        chrono::LocalResult::None => {
            // DST gap: the target wall-clock time doesn't exist.
            // Advance to the next interval boundary that does exist.
            let next_start = interval_start + interval_minutes;
            if next_start < 1440 {
                let nh = (next_start / 60) as u32;
                let nm = (next_start % 60) as u32;
                let next_naive = dt.date_naive().and_hms_opt(nh, nm, 0).unwrap();
                match Local.from_local_datetime(&next_naive) {
                    chrono::LocalResult::Single(d) | chrono::LocalResult::Ambiguous(d, _) => d,
                    _ => dt.with_second(0).unwrap().with_nanosecond(0).unwrap(),
                }
            } else {
                dt.with_second(0).unwrap().with_nanosecond(0).unwrap()
            }
        }
    }
}

/// Round a start_time down to its interval boundary, handling DST gaps.
pub fn round_to_interval_start(dt: &DateTime<Local>, interval_minutes: i64) -> DateTime<Local> {
    to_interval(dt, interval_minutes)
}

/// Generate all wall-clock-aligned interval times between `start` (inclusive) and
/// `end` (inclusive). Skips times that fall in DST gaps.
pub fn generate_interval_times(
    start: &DateTime<Local>,
    end: &DateTime<Local>,
    interval_minutes: i64,
) -> Vec<DateTime<Local>> {
    let start_date = start.date_naive();
    let end_date = end.date_naive();
    let intervals_per_day = 1440 / interval_minutes;

    let mut times = Vec::new();
    let mut d = start_date;
    while d <= end_date {
        for i in 0..intervals_per_day {
            let minutes = i * interval_minutes;
            let hour = (minutes / 60) as u32;
            let minute = (minutes % 60) as u32;
            let naive = d.and_hms_opt(hour, minute, 0).unwrap();
            match Local.from_local_datetime(&naive) {
                chrono::LocalResult::Single(local_dt)
                | chrono::LocalResult::Ambiguous(local_dt, _) => {
                    if local_dt >= *start && local_dt <= *end {
                        times.push(local_dt);
                    }
                }
                chrono::LocalResult::None => {
                    // DST gap: skip
                }
            }
        }
        d = d.succ_opt().unwrap_or(d);
    }
    times
}

/// Token breakdown for distribution across intervals.
#[derive(Clone, Debug, Default)]
pub struct TokenFractions {
    pub input: f64,
    pub output: f64,
    pub cache_creation: f64,
    pub cache_read: f64,
}

/// Distribute tokens evenly across time intervals within a session time span.
pub fn distribute_tokens_to_intervals(
    session_start_str: &str,
    session_end_str: &str,
    tokens: &TokenFractions,
    interval_minutes: i64,
) -> Vec<(DateTime<Local>, TokenFractions)> {
    let start_local = match parse_timestamp(session_start_str) {
        Some(dt) => dt,
        None => return Vec::new(),
    };
    let end_local = match parse_timestamp(session_end_str) {
        Some(dt) => dt,
        None => return Vec::new(),
    };

    let start_interval = to_interval(&start_local, interval_minutes);
    let end_interval = to_interval(&end_local, interval_minutes);

    let intervals = generate_interval_times(&start_interval, &end_interval, interval_minutes);

    if intervals.is_empty() {
        return Vec::new();
    }

    let n = intervals.len() as f64;
    intervals
        .into_iter()
        .map(|interval_time| {
            let fraction = TokenFractions {
                input: tokens.input / n,
                output: tokens.output / n,
                cache_creation: tokens.cache_creation / n,
                cache_read: tokens.cache_read / n,
            };
            (interval_time, fraction)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};

    #[test]
    fn date_window_includes_the_whole_local_day() {
        let window = TimeWindow::from_date("2026-05-07").expect("valid date");
        let (start, end) = window.bounds(Local::now());

        assert_eq!(start.year(), 2026);
        assert_eq!(start.month(), 5);
        assert_eq!(start.day(), 7);
        assert_eq!(start.hour(), 0);
        assert_eq!(start.minute(), 0);
        assert_eq!(start.second(), 0);

        assert_eq!(end.year(), 2026);
        assert_eq!(end.month(), 5);
        assert_eq!(end.day(), 7);
        assert_eq!(end.hour(), 23);
        assert_eq!(end.minute(), 59);
        assert_eq!(end.second(), 59);
    }

    #[test]
    fn date_range_includes_the_whole_end_date() {
        let window = TimeWindow::from_range("2026-05-01", "2026-05-07").expect("valid date range");
        let (start, end) = window.bounds(Local::now());

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 00:00:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-07 23:59:59"
        );
        assert_eq!(window.projection_days(Local::now()), 7.0);
    }

    #[test]
    fn date_time_range_is_interpreted_as_local_time() {
        let window = TimeWindow::from_range("2026-05-01T08:30", "2026-05-01T10:00")
            .expect("valid local date-time range");
        let (start, end) = window.bounds(Local::now());

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 08:30:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 10:00:00"
        );
    }

    #[test]
    fn reversed_date_range_is_silently_swapped() {
        let forward = TimeWindow::from_range("2026-05-01", "2026-05-07").expect("forward");
        let reversed = TimeWindow::from_range("2026-05-07", "2026-05-01").expect("reversed");

        let (fwd_start, fwd_end) = forward.bounds(Local::now());
        let (rev_start, rev_end) = reversed.bounds(Local::now());

        assert_eq!(fwd_start, rev_start);
        assert_eq!(fwd_end, rev_end);
        assert_eq!(
            forward.projection_days(Local::now()),
            reversed.projection_days(Local::now())
        );
    }

    #[test]
    fn reversed_date_time_range_is_silently_swapped() {
        let reversed = TimeWindow::from_range("2026-05-01T10:00", "2026-05-01T08:30")
            .expect("reversed date-time range should be accepted");
        let (start, end) = reversed.bounds(Local::now());

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 08:30:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 10:00:00"
        );
    }

    #[test]
    fn slide_back_from_rolling_keeps_width_and_drops_anchor() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        let slid = window.slide_back(now).expect("slide back");
        let (start, end) = slid.bounds(now);

        // The new window covers (now - 6d, now - 3d).
        assert_eq!(end, now - Duration::days(3));
        assert_eq!(start, now - Duration::days(6));
        assert_eq!(slid.page_step(), Duration::days(3));
    }

    #[test]
    fn slide_back_on_date_range_steps_in_whole_days() {
        let window = TimeWindow::from_range("2026-05-01", "2026-05-07").expect("range");
        let slid = window.slide_back(Local::now()).expect("slide back");

        let (start, end) = slid.bounds(Local::now());
        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-04-24 00:00:00"
        );
        // The new end is the previous start minus 1ns, i.e. end-of-day 4-30.
        assert_eq!(end.format("%Y-%m-%d").to_string(), "2026-04-30");
        assert_eq!(end.hour(), 23);
        assert_eq!(end.minute(), 59);
        assert_eq!(end.second(), 59);
    }

    #[test]
    fn slide_forward_on_rolling_window_is_a_no_op_due_to_now_clamp() {
        let now = Local::now();
        let window = TimeWindow::rolling_days(3);

        assert!(window.slide_forward(now).is_none());
    }

    #[test]
    fn slide_forward_after_back_returns_to_anchor() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        let back = window.slide_back(now).expect("slide back");
        let forward = back.slide_forward(now).expect("slide forward");

        let (orig_start, orig_end) = window.bounds(now);
        let (start, end) = forward.bounds(now);
        assert_eq!(start, orig_start);
        assert_eq!(end, orig_end);
    }

    #[test]
    fn slide_forward_clamps_end_at_now_preserving_width() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(7);

        // Slide back twice, then slide forward; result should still be 7
        // days wide and end no later than `now`.
        let back2 = window
            .slide_back(now)
            .expect("back")
            .slide_back(now)
            .expect("back2");
        let forward = back2.slide_forward(now).expect("forward");
        let (start, end) = forward.bounds(now);

        assert_eq!(end - start, Duration::days(7));
        assert!(end <= now);
    }

    #[test]
    fn zoom_in_on_rolling_window_keeps_present_edge() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(4);

        let zoomed = window.zoom_in(now).expect("zoom in");
        let (start, end) = zoomed.bounds(now);

        assert_eq!(end, now);
        assert_eq!(start, now - Duration::days(2));
        assert_eq!(zoomed.page_step(), Duration::days(2));
        assert_eq!(zoomed.projection_days(now), 2.0);
    }

    #[test]
    fn zoom_in_on_explicit_window_keeps_center() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::from_range("2026-05-01T00:00", "2026-05-03T00:00").expect("range");

        let zoomed = window.zoom_in(now).expect("zoom in");
        let (start, end) = zoomed.bounds(now);

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 12:00:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-02 12:00:00"
        );
        assert_eq!(zoomed.page_step(), Duration::days(1));
        assert_eq!(zoomed.projection_days(now), 1.0);
    }

    #[test]
    fn zoom_out_clamps_to_present_while_preserving_new_width() {
        let now = Local
            .with_ymd_and_hms(2026, 5, 10, 12, 0, 0)
            .single()
            .expect("fixed now");
        let window = TimeWindow::rolling_days(3);

        let zoomed = window.zoom_out(now).expect("zoom out");
        let (start, end) = zoomed.bounds(now);

        assert_eq!(end, now);
        assert_eq!(start, now - Duration::days(6));
        assert_eq!(zoomed.page_step(), Duration::days(6));
        assert_eq!(zoomed.projection_days(now), 6.0);
    }

    #[test]
    fn mixed_endpoint_kinds_pick_outermost_bounds() {
        let window =
            TimeWindow::from_range("2026-05-07", "2026-05-01T08:30").expect("mixed-endpoint range");
        let (start, end) = window.bounds(Local::now());

        assert_eq!(
            start.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-01 08:30:00"
        );
        assert_eq!(
            end.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-05-07 23:59:59"
        );
    }
}
