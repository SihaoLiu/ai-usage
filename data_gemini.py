"""Data reading and filtering functions for Gemini CLI usage analysis."""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta


def get_gemini_dir():
    """Get Gemini configuration directory."""
    gemini_dir = os.environ.get('GEMINI_CONFIG_DIR', os.path.expanduser('~/.gemini'))
    return Path(gemini_dir)


def read_gemini_json_files(tmp_dir):
    """Read all session JSON files from Gemini tmp directory.

    Gemini stores sessions in ~/.gemini/tmp/<project_hash>/chats/session-*.json

    Token data is in messages with type == "gemini" containing a tokens object:
    {
        "input": 8493,      # total input tokens
        "output": 37,       # output tokens
        "cached": 3207,     # cached input tokens
        "thoughts": 190,    # thinking/reasoning tokens
        "tool": 0,          # tool tokens
        "total": 8720       # total
    }

    Returns normalized data similar to Claude format for compatibility:
    [
        {
            'timestamp': '2025-12-11T23:18:08.351Z',
            'message': {
                'model': 'gemini-3-pro-preview',
                'usage': {
                    'input_tokens': 5286,       # non-cached input (input - cached)
                    'output_tokens': 37,        # output tokens
                    'cache_read_input_tokens': 3207,  # cached input
                    'cache_creation_input_tokens': 190,  # thoughts (for chart compatibility)
                }
            }
        },
        ...
    ]
    """
    usage_data = []

    # Find all session JSON files in tmp directory
    # Pattern: ~/.gemini/tmp/<project_hash>/chats/session-*.json
    for session_file in tmp_dir.rglob('chats/session-*.json'):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract messages array
            messages = data.get('messages', [])

            for msg in messages:
                # Only process gemini type messages with tokens
                if msg.get('type') != 'gemini':
                    continue

                tokens = msg.get('tokens')
                if not tokens:
                    continue

                timestamp = msg.get('timestamp')
                if not timestamp:
                    continue

                model = msg.get('model', 'unknown')

                # Extract token values
                total_input = tokens.get('input', 0)
                cached_input = tokens.get('cached', 0)
                output_tokens = tokens.get('output', 0)
                thoughts_tokens = tokens.get('thoughts', 0)

                # Calculate non-cached input
                non_cached_input = total_input - cached_input

                # Normalize to Claude-like format
                normalized_entry = {
                    'timestamp': timestamp,
                    'message': {
                        'model': model,
                        'usage': {
                            'input_tokens': non_cached_input,
                            'output_tokens': output_tokens,
                            'cache_read_input_tokens': cached_input,
                            # Use cache_creation to store thoughts (for chart compatibility)
                            'cache_creation_input_tokens': thoughts_tokens,
                        }
                    }
                }
                usage_data.append(normalized_entry)

        except (json.JSONDecodeError, KeyError):
            continue
        except Exception:
            continue

    return usage_data


def filter_gemini_usage_data_by_days(usage_data, days_back):
    """Filter Gemini usage data to only include entries from the last N days."""
    if not usage_data:
        return []

    # Get local timezone automatically
    local_tz = datetime.now().astimezone().tzinfo

    # Find the latest timestamp in the data
    latest_time = None
    for entry in usage_data:
        timestamp_str = entry.get('timestamp')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                timestamp_local = timestamp.astimezone(local_tz)
                if latest_time is None or timestamp_local > latest_time:
                    latest_time = timestamp_local
            except Exception:
                continue

    if latest_time is None:
        return usage_data

    # Calculate start time based on days_back
    start_time = latest_time - timedelta(days=days_back)

    # Filter data
    filtered_data = []
    for entry in usage_data:
        timestamp_str = entry.get('timestamp')
        if not timestamp_str:
            continue

        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            timestamp_local = timestamp.astimezone(local_tz)
            if timestamp_local >= start_time:
                filtered_data.append(entry)
        except Exception:
            continue

    return filtered_data
