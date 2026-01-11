"""Statistics calculation functions for Claude Code usage analysis."""

from collections import defaultdict

from time_utils import parse_timestamp, distribute_tokens_to_intervals


def calculate_overall_stats(usage_data):
    """Calculate overall usage statistics."""
    stats = {
        'total_messages': len(usage_data),
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_creation_tokens': 0,
        'cache_read_tokens': 0,
    }

    for entry in usage_data:
        usage = entry['message']['usage']
        stats['input_tokens'] += usage.get('input_tokens', 0)
        stats['output_tokens'] += usage.get('output_tokens', 0)
        stats['cache_creation_tokens'] += usage.get('cache_creation_input_tokens', 0)
        stats['cache_read_tokens'] += usage.get('cache_read_input_tokens', 0)

    stats['total_tokens'] = stats['input_tokens'] + stats['output_tokens']

    return stats


def calculate_model_breakdown(usage_data):
    """Calculate usage breakdown by model."""
    model_stats = defaultdict(lambda: {
        'count': 0,
        'input': 0,
        'output': 0,
        'cache_creation': 0,
        'cache_read': 0,
    })

    for entry in usage_data:
        model = entry['message'].get('model', 'unknown')
        usage = entry['message']['usage']

        model_stats[model]['count'] += 1
        model_stats[model]['input'] += usage.get('input_tokens', 0)
        model_stats[model]['output'] += usage.get('output_tokens', 0)
        model_stats[model]['cache_creation'] += usage.get('cache_creation_input_tokens', 0)
        model_stats[model]['cache_read'] += usage.get('cache_read_input_tokens', 0)

    # Calculate totals and sort by total tokens
    result = []

    # First pass: calculate total message count and populate result
    total_messages = sum(stats['count'] for stats in model_stats.values())
    threshold = total_messages * 0.01  # 1% threshold

    for model, stats in model_stats.items():
        # Skip models with less than 1% of total messages
        if stats['count'] < threshold:
            continue
        stats['model'] = model
        stats['total'] = stats['input'] + stats['output']
        stats['total_with_cache'] = stats['input'] + stats['output'] + stats['cache_creation'] + stats['cache_read']
        result.append(stats)

    result.sort(key=lambda x: x['total'], reverse=True)
    return result


def calculate_time_series(usage_data, interval_hours=1):
    """Calculate token usage over time in specified hour intervals (local timezone)."""
    time_series = defaultdict(lambda: defaultdict(int))

    for entry in usage_data:
        # Use pre-parsed timestamp if available (from caching layer)
        timestamp_local = entry.get('_parsed_timestamp')
        if timestamp_local is None:
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                timestamp_local = parse_timestamp(timestamp_str)

        if timestamp_local is None:
            continue

        try:
            # Round down to the nearest interval
            hour = timestamp_local.hour
            interval_hour = (hour // interval_hours) * interval_hours
            interval_time = timestamp_local.replace(hour=interval_hour, minute=0, second=0, microsecond=0)

            model = entry['message'].get('model', 'unknown')
            usage = entry['message']['usage']

            total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)

            time_series[interval_time][model] += total_tokens
        except Exception:
            continue

    return time_series


def calculate_all_tokens_time_series(usage_data, interval_hours=1):
    """Calculate ALL token usage (input + output + cache) over time in specified hour intervals (local timezone)."""
    time_series = defaultdict(lambda: defaultdict(int))

    for entry in usage_data:
        # Use pre-parsed timestamp if available (from caching layer)
        timestamp_local = entry.get('_parsed_timestamp')
        if timestamp_local is None:
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                timestamp_local = parse_timestamp(timestamp_str)

        if timestamp_local is None:
            continue

        try:
            # Round down to the nearest interval
            hour = timestamp_local.hour
            interval_hour = (hour // interval_hours) * interval_hours
            interval_time = timestamp_local.replace(hour=interval_hour, minute=0, second=0, microsecond=0)

            usage = entry['message']['usage']

            total_tokens = (usage.get('input_tokens', 0) +
                          usage.get('output_tokens', 0) +
                          usage.get('cache_creation_input_tokens', 0) +
                          usage.get('cache_read_input_tokens', 0))

            time_series[interval_time]['all'] += total_tokens
        except Exception:
            continue

    return time_series


def calculate_token_breakdown_time_series(usage_data, interval_hours=1):
    """Calculate token usage breakdown (input/output/cache_creation/cache_read) over time in specified hour intervals (local timezone)."""
    time_series = defaultdict(lambda: {
        'input': 0,
        'output': 0,
        'cache_creation': 0,
        'cache_read': 0
    })

    for entry in usage_data:
        # Use pre-parsed timestamp if available (from caching layer)
        timestamp_local = entry.get('_parsed_timestamp')
        if timestamp_local is None:
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                timestamp_local = parse_timestamp(timestamp_str)

        if timestamp_local is None:
            continue

        try:
            # Round down to the nearest interval
            hour = timestamp_local.hour
            interval_hour = (hour // interval_hours) * interval_hours
            interval_time = timestamp_local.replace(hour=interval_hour, minute=0, second=0, microsecond=0)

            usage = entry['message']['usage']

            time_series[interval_time]['input'] += usage.get('input_tokens', 0)
            time_series[interval_time]['output'] += usage.get('output_tokens', 0)
            time_series[interval_time]['cache_creation'] += usage.get('cache_creation_input_tokens', 0)
            time_series[interval_time]['cache_read'] += usage.get('cache_read_input_tokens', 0)
        except Exception:
            continue

    return time_series


def calculate_model_token_breakdown_time_series(usage_data, interval_minutes=60):
    """Calculate token usage breakdown by model over time in specified minute intervals (local timezone).

    This version distributes tokens evenly across the session time span to produce
    smoother charts. Long-running sessions will have their token usage spread across
    all intervals they span, rather than being concentrated at a single timestamp.

    Args:
        usage_data: List of usage data entries
        interval_minutes: Interval in minutes for bucketing (default: 60 = 1 hour)

    Returns a time series where each time interval contains per-model token breakdowns:
    {
        interval_time: {
            'model_name': {
                'input': 0,
                'output': 0,
                'cache_creation': 0,
                'cache_read': 0
            },
            ...
        },
        ...
    }
    """
    time_series = defaultdict(lambda: defaultdict(lambda: {
        'input': 0,
        'output': 0,
        'cache_creation': 0,
        'cache_read': 0
    }))

    for entry in usage_data:
        timestamp_str = entry.get('timestamp')
        if not timestamp_str:
            continue

        model = entry['message'].get('model', 'unknown')
        usage = entry['message']['usage']

        # Get session time span (if available)
        session_start = entry.get('session_start_time', timestamp_str)
        session_end = entry.get('session_end_time', timestamp_str)

        tokens = {
            'input': usage.get('input_tokens', 0),
            'output': usage.get('output_tokens', 0),
            'cache_creation': usage.get('cache_creation_input_tokens', 0),
            'cache_read': usage.get('cache_read_input_tokens', 0)
        }

        # Distribute tokens across intervals within session time span
        distributed = distribute_tokens_to_intervals(
            session_start, session_end, tokens, interval_minutes
        )

        if distributed:
            for interval_time, fraction_tokens in distributed:
                time_series[interval_time][model]['input'] += fraction_tokens['input']
                time_series[interval_time][model]['output'] += fraction_tokens['output']
                time_series[interval_time][model]['cache_creation'] += fraction_tokens['cache_creation']
                time_series[interval_time][model]['cache_read'] += fraction_tokens['cache_read']
        else:
            # Fallback: use original timestamp-based bucketing
            timestamp_local = parse_timestamp(timestamp_str)
            if timestamp_local is None:
                continue

            total_minutes = timestamp_local.hour * 60 + timestamp_local.minute
            interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
            interval_hour = interval_start_minutes // 60
            interval_minute = interval_start_minutes % 60

            interval_time = timestamp_local.replace(
                hour=interval_hour, minute=interval_minute, second=0, microsecond=0
            )

            time_series[interval_time][model]['input'] += tokens['input']
            time_series[interval_time][model]['output'] += tokens['output']
            time_series[interval_time][model]['cache_creation'] += tokens['cache_creation']
            time_series[interval_time][model]['cache_read'] += tokens['cache_read']

    return time_series
