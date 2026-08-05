//! Monitor-mode refresh cadence: automatic (derived from the displayed chart
//! interval) or an explicit value the user asked for.

use std::time::Duration;

use chrono::{DateTime, Local};

use crate::AppState;

/// Bounds of the automatic cadence: never faster than a minute, never slower
/// than an hour, whatever the chart interval is.
const AUTO_MIN: Duration = Duration::from_secs(60);
const AUTO_MAX: Duration = Duration::from_secs(60 * 60);

/// Longest cadence an explicit `i <seconds>` may request. Deadlines are
/// `Instant + Duration`, which panics once the sum leaves the clock's range,
/// so the accepted range is bounded well inside it.
const MANUAL_MAX_SECONDS: u64 = 24 * 60 * 60;

/// How often monitor mode reloads data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RefreshInterval {
    /// Follow the chart interval of the displayed window, clamped to
    /// `[AUTO_MIN, AUTO_MAX]`. The startup mode.
    #[default]
    Auto,
    /// A cadence in seconds the user set explicitly; kept until `i auto`.
    Manual(u64),
}

impl RefreshInterval {
    /// Parse the argument of the `i` / `interval` command.
    pub(crate) fn parse(arg: &str) -> Result<Self, &'static str> {
        if arg.eq_ignore_ascii_case("auto") {
            return Ok(RefreshInterval::Auto);
        }
        match arg.parse::<u64>() {
            Ok(seconds) if (1..=MANUAL_MAX_SECONDS).contains(&seconds) => {
                Ok(RefreshInterval::Manual(seconds))
            }
            Ok(_) => Err("Interval must be 1 to 86400 seconds, or auto."),
            Err(_) => Err("Invalid interval value."),
        }
    }
}

/// Clamp one chart interval to the automatic refresh bounds.
pub(crate) fn auto_refresh_interval(chart_interval_minutes: i64) -> Duration {
    let minutes = u64::try_from(chart_interval_minutes).unwrap_or(0);
    Duration::from_secs(minutes.saturating_mul(60)).clamp(AUTO_MIN, AUTO_MAX)
}

/// The cadence the next reload is scheduled with.
pub(crate) fn effective_refresh_interval(state: &AppState, now: DateTime<Local>) -> Duration {
    match state.refresh_interval {
        RefreshInterval::Manual(seconds) => Duration::from_secs(seconds),
        RefreshInterval::Auto => auto_refresh_interval(crate::display_interval_minutes_for_window(
            &state.time_window,
            now,
            crate::get_chart_target_width(),
        )),
    }
}

/// Prompt-facing form of the active cadence.
pub(crate) fn describe_refresh_interval(state: &AppState, now: DateTime<Local>) -> String {
    let seconds = effective_refresh_interval(state, now).as_secs();
    match state.refresh_interval {
        RefreshInterval::Auto => format!("auto ({seconds} seconds)"),
        RefreshInterval::Manual(_) => format!("{seconds} seconds"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_follows_the_chart_interval_between_one_minute_and_one_hour() {
        assert_eq!(auto_refresh_interval(15), Duration::from_secs(900));
        assert_eq!(auto_refresh_interval(30), Duration::from_secs(1_800));
        assert_eq!(auto_refresh_interval(60), AUTO_MAX);
    }

    #[test]
    fn auto_clamps_chart_intervals_outside_the_refresh_bounds() {
        assert_eq!(auto_refresh_interval(1), AUTO_MIN);
        assert_eq!(auto_refresh_interval(0), AUTO_MIN);
        assert_eq!(auto_refresh_interval(-5), AUTO_MIN);
        assert_eq!(auto_refresh_interval(480), AUTO_MAX);
        assert_eq!(auto_refresh_interval(i64::MAX), AUTO_MAX);
    }

    #[test]
    fn monitor_mode_starts_in_auto() {
        assert_eq!(RefreshInterval::default(), RefreshInterval::Auto);
    }

    #[test]
    fn interval_argument_accepts_auto_or_whole_seconds() {
        assert_eq!(RefreshInterval::parse("auto"), Ok(RefreshInterval::Auto));
        assert_eq!(RefreshInterval::parse("AUTO"), Ok(RefreshInterval::Auto));
        assert_eq!(
            RefreshInterval::parse("30"),
            Ok(RefreshInterval::Manual(30))
        );
        assert!(RefreshInterval::parse("0").is_err());
        assert!(RefreshInterval::parse("-5").is_err());
        assert!(RefreshInterval::parse("90s").is_err());
        assert!(RefreshInterval::parse("").is_err());
    }

    /// Deadlines are `Instant + Duration`; an unbounded cadence would panic
    /// the monitor loop instead of being rejected at the prompt.
    #[test]
    fn manual_interval_stays_within_the_schedulable_range() {
        assert_eq!(
            RefreshInterval::parse("86400"),
            Ok(RefreshInterval::Manual(MANUAL_MAX_SECONDS))
        );
        assert!(RefreshInterval::parse("86401").is_err());
        assert!(RefreshInterval::parse(&u64::MAX.to_string()).is_err());

        let longest = Duration::from_secs(MANUAL_MAX_SECONDS);
        assert!(std::time::Instant::now().checked_add(longest).is_some());
    }
}
