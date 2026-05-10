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

        let projection_days = if earlier.is_date_only && later.is_date_only {
            let s_date = start.date_naive();
            let e_date = end.date_naive();
            e_date
                .signed_duration_since(s_date)
                .num_days()
                .saturating_add(1)
                .max(1) as f64
        } else {
            ((end - start).num_seconds() as f64 / 86_400.0).max(1.0 / 1440.0)
        };

        Ok(Self::ExplicitRange {
            start,
            end,
            projection_days,
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

    pub fn file_scan_days(&self, _now: DateTime<Local>) -> Option<i64> {
        match self {
            Self::RollingDays { days } => Some(*days),
            Self::ExplicitRange { .. } => None,
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
    if let Ok(ndt) = NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%dT%H:%M:%S%.f") {
        if let Some(utc_dt) = Utc.from_local_datetime(&ndt).single() {
            return Some(utc_dt.with_timezone(&Local));
        }
    }
    // Try epoch milliseconds (numeric string)
    if let Ok(ms) = timestamp_str.parse::<i64>() {
        if let Some(dt) = DateTime::from_timestamp_millis(ms) {
            return Some(dt.with_timezone(&Local));
        }
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
    fn mixed_endpoint_kinds_pick_outermost_bounds() {
        let window = TimeWindow::from_range("2026-05-07", "2026-05-01T08:30")
            .expect("mixed-endpoint range");
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
