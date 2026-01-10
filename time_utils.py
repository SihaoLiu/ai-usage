"""Shared timestamp parsing and filtering utilities for usage analysis."""

from datetime import datetime, timedelta

# Precompute local timezone at module load (avoids repeated calls)
LOCAL_TZ = datetime.now().astimezone().tzinfo

# Cache for parsed timestamps: {timestamp_str: datetime_local}
_TIMESTAMP_CACHE = {}


def parse_timestamp(timestamp_str):
    """Parse ISO timestamp string to local datetime with caching."""
    if timestamp_str in _TIMESTAMP_CACHE:
        return _TIMESTAMP_CACHE[timestamp_str]
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        timestamp_local = timestamp.astimezone(LOCAL_TZ)
        _TIMESTAMP_CACHE[timestamp_str] = timestamp_local
        return timestamp_local
    except Exception:
        return None


def filter_usage_data_by_days(usage_data, days_back):
    """Filter usage data to only include entries from the last N days.

    Optimized single-pass filtering using pre-parsed timestamps.
    Works for all vendors (Claude, Codex, Gemini) since they use the
    same normalized data format with '_parsed_timestamp' and 'timestamp' fields.
    """
    if not usage_data:
        return []

    # Single pass: find latest timestamp and collect entries with parsed timestamps
    latest_time = None
    entries_with_ts = []

    for entry in usage_data:
        # Use pre-parsed timestamp if available (from caching layer)
        timestamp_local = entry.get('_parsed_timestamp')
        if timestamp_local is None:
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                timestamp_local = parse_timestamp(timestamp_str)

        if timestamp_local:
            entries_with_ts.append((entry, timestamp_local))
            if latest_time is None or timestamp_local > latest_time:
                latest_time = timestamp_local

    if latest_time is None:
        return usage_data

    # Calculate start time based on days_back
    start_time = latest_time - timedelta(days=days_back)

    # Filter in single pass (already have parsed timestamps)
    return [entry for entry, ts in entries_with_ts if ts >= start_time]


def distribute_tokens_to_intervals(session_start_str, session_end_str, tokens, interval_minutes):
    """Distribute tokens evenly across time intervals within a session time span.

    Args:
        session_start_str: ISO timestamp string for session start
        session_end_str: ISO timestamp string for session end
        tokens: Dictionary with token counts (input, output, cache_creation, cache_read)
        interval_minutes: Interval in minutes for bucketing

    Returns:
        List of (interval_time, fraction_tokens) tuples where fraction_tokens has the
        same keys as input tokens dict but with proportionally distributed values.
    """
    start_local = parse_timestamp(session_start_str)
    end_local = parse_timestamp(session_end_str)

    if start_local is None or end_local is None:
        return []

    # Calculate start and end intervals
    def to_interval(dt):
        total_minutes = dt.hour * 60 + dt.minute
        interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
        interval_hour = interval_start_minutes // 60
        interval_minute = interval_start_minutes % 60
        return dt.replace(hour=interval_hour, minute=interval_minute, second=0, microsecond=0)

    start_interval = to_interval(start_local)
    end_interval = to_interval(end_local)

    # Generate all intervals between start and end (inclusive)
    intervals = []
    current = start_interval
    while current <= end_interval:
        intervals.append(current)
        current += timedelta(minutes=interval_minutes)

    if not intervals:
        return []

    # Distribute tokens evenly across intervals
    num_intervals = len(intervals)
    result = []
    for interval_time in intervals:
        fraction_tokens = {key: value / num_intervals for key, value in tokens.items()}
        result.append((interval_time, fraction_tokens))

    return result
