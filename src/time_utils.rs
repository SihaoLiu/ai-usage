use chrono::{DateTime, Duration, Local, NaiveDateTime, TimeZone, Timelike, Utc};

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
pub fn to_interval(dt: &DateTime<Local>, interval_minutes: i64) -> DateTime<Local> {
    let total_minutes = (dt.hour() as i64) * 60 + (dt.minute() as i64);
    let interval_start = (total_minutes / interval_minutes) * interval_minutes;
    let hour = (interval_start / 60) as u32;
    let minute = (interval_start % 60) as u32;
    dt.with_hour(hour)
        .unwrap()
        .with_minute(minute)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap()
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

    let mut intervals = Vec::new();
    let mut current = start_interval;
    while current <= end_interval {
        intervals.push(current);
        current = current + Duration::minutes(interval_minutes);
    }

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
