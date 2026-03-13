use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Timelike, Utc};

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
