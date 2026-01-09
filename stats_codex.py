"""Statistics calculation functions for Codex (OpenAI) usage analysis."""

from datetime import datetime, timedelta
from collections import defaultdict


def calculate_codex_model_breakdown(usage_data):
    """Calculate usage breakdown by model (including effort level).

    Codex has different token types than Claude:
    - input_tokens (non-cached)
    - output_tokens (non-reasoning)
    - cache_read_input_tokens (cached input)
    - reasoning_output_tokens (reasoning output)
    """
    model_stats = defaultdict(lambda: {
        'count': 0,
        'input': 0,
        'output': 0,
        'cache_read': 0,
        'reasoning': 0,
        # For compatibility with Claude formatting
        'cache_creation': 0,
    })

    for entry in usage_data:
        model = entry['message'].get('model', 'unknown')
        effort = entry['message'].get('effort', 'unknown')
        usage = entry['message']['usage']

        # Combine model and effort for display
        model_key = f"{model} ({effort})"

        model_stats[model_key]['count'] += 1
        model_stats[model_key]['input'] += usage.get('input_tokens', 0)
        model_stats[model_key]['output'] += usage.get('output_tokens', 0)
        model_stats[model_key]['cache_read'] += usage.get('cache_read_input_tokens', 0)
        model_stats[model_key]['reasoning'] += usage.get('reasoning_output_tokens', 0)

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
        # Total IO (non-cached input + non-reasoning output)
        stats['total'] = stats['input'] + stats['output']
        # Total with all tokens
        stats['total_with_cache'] = (stats['input'] + stats['output'] +
                                      stats['cache_read'] + stats['reasoning'])
        result.append(stats)

    result.sort(key=lambda x: x['total'], reverse=True)
    return result


def calculate_codex_model_token_breakdown_time_series(usage_data, interval_minutes=60):
    """Calculate token usage breakdown by model over time for Codex.

    Args:
        usage_data: List of usage data entries
        interval_minutes: Interval in minutes for bucketing (default: 60 = 1 hour)

    Returns a time series where each time interval contains per-model token breakdowns:
    {
        interval_time: {
            'model_name (effort)': {
                'input': 0,           # non-cached input
                'output': 0,          # non-reasoning output
                'cache_creation': 0,  # reasoning_output for Codex (replaces cache_creation)
                'cache_read': 0,      # cached input
            },
            ...
        },
        ...
    }

    Note: For Codex, 'cache_creation' field stores reasoning_output tokens
    so Chart 1 shows (input + output) and Chart 2 shows (cache_read + reasoning)
    """
    # Get local timezone automatically
    local_tz = datetime.now().astimezone().tzinfo

    # Group by time interval and model with breakdown
    time_series = defaultdict(lambda: defaultdict(lambda: {
        'input': 0,
        'output': 0,
        'cache_creation': 0,  # For Codex: stores reasoning_output
        'cache_read': 0,
    }))

    for entry in usage_data:
        timestamp_str = entry.get('timestamp')
        if not timestamp_str:
            continue

        try:
            # Parse ISO timestamp and convert to local timezone
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            timestamp_local = timestamp.astimezone(local_tz)

            # Round down to the nearest interval
            total_minutes = timestamp_local.hour * 60 + timestamp_local.minute
            interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes

            # Convert back to hour and minute
            interval_hour = interval_start_minutes // 60
            interval_minute = interval_start_minutes % 60

            interval_time = timestamp_local.replace(
                hour=interval_hour, minute=interval_minute, second=0, microsecond=0
            )

            model = entry['message'].get('model', 'unknown')
            effort = entry['message'].get('effort', 'unknown')
            model_key = f"{model} ({effort})"
            usage = entry['message']['usage']

            # Accumulate each token type
            time_series[interval_time][model_key]['input'] += usage.get('input_tokens', 0)
            time_series[interval_time][model_key]['output'] += usage.get('output_tokens', 0)
            time_series[interval_time][model_key]['cache_read'] += usage.get('cache_read_input_tokens', 0)
            # For Codex: use cache_creation field to store reasoning_output
            time_series[interval_time][model_key]['cache_creation'] += usage.get('reasoning_output_tokens', 0)

        except Exception:
            continue

    return time_series
