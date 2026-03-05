"""Statistics calculation functions for Gemini CLI usage analysis."""

from collections import defaultdict

from time_utils import parse_timestamp, distribute_tokens_to_intervals


def calculate_gemini_model_breakdown(usage_data):
    """Calculate usage breakdown by model for Gemini.

    Gemini has different token types:
    - input_tokens (non-cached)
    - output_tokens
    - cache_read_input_tokens (cached input)
    - cache_creation_input_tokens (thoughts/thinking tokens)
    """
    model_stats = defaultdict(lambda: {
        'count': 0,
        'input': 0,
        'output': 0,
        'cache_read': 0,
        'thinking': 0,
        'cache_creation': 0,  # For compatibility with Claude formatting
    })

    for entry in usage_data:
        model = entry['message'].get('model', 'unknown')
        usage = entry['message']['usage']

        model_stats[model]['count'] += 1
        model_stats[model]['input'] += usage.get('input_tokens', 0)
        model_stats[model]['output'] += usage.get('output_tokens', 0)
        model_stats[model]['cache_read'] += usage.get('cache_read_input_tokens', 0)
        # cache_creation stores thoughts tokens for Gemini
        thinking_tokens = usage.get('cache_creation_input_tokens', 0)
        model_stats[model]['thinking'] += thinking_tokens
        model_stats[model]['cache_creation'] += thinking_tokens

    # Calculate totals and sort by total tokens
    result = []

    total_messages = sum(stats['count'] for stats in model_stats.values())
    threshold = total_messages * 0.01  # 1% threshold

    for model, stats in model_stats.items():
        # Skip models with less than 1% of total messages
        if stats['count'] < threshold:
            continue
        stats['model'] = model
        stats['total'] = stats['input'] + stats['output']
        stats['total_with_cache'] = (stats['input'] + stats['output'] +
                                      stats['cache_read'] + stats['thinking'])
        result.append(stats)

    result.sort(key=lambda x: x['total'], reverse=True)
    return result


def calculate_gemini_model_token_breakdown_time_series(usage_data, interval_minutes=60):
    """Calculate token usage breakdown by model over time for Gemini.

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
                'cache_creation': 0,  # thinking tokens for Gemini
                'cache_read': 0,
            },
            ...
        },
        ...
    }

    Note: For Gemini, 'cache_creation' field stores thinking tokens
    so Chart 1 shows (input + output) and Chart 2 shows (cache_read + thinking)
    """
    time_series = defaultdict(lambda: defaultdict(lambda: {
        'input': 0,
        'output': 0,
        'cache_creation': 0,  # For Gemini: stores thinking tokens
        'cache_read': 0,
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
            'cache_read': usage.get('cache_read_input_tokens', 0),
            'cache_creation': usage.get('cache_creation_input_tokens', 0),  # thinking for Gemini
        }

        # Distribute tokens across intervals within session time span
        distributed = distribute_tokens_to_intervals(
            session_start, session_end, tokens, interval_minutes
        )

        if distributed:
            for interval_time, fraction_tokens in distributed:
                time_series[interval_time][model]['input'] += fraction_tokens['input']
                time_series[interval_time][model]['output'] += fraction_tokens['output']
                time_series[interval_time][model]['cache_read'] += fraction_tokens['cache_read']
                time_series[interval_time][model]['cache_creation'] += fraction_tokens['cache_creation']
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
            time_series[interval_time][model]['cache_read'] += tokens['cache_read']
            time_series[interval_time][model]['cache_creation'] += tokens['cache_creation']

    return time_series
