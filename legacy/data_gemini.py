"""Data reading and filtering functions for Gemini CLI usage analysis."""

import json
import os
from pathlib import Path

from time_utils import parse_timestamp, filter_usage_data_by_days

# File-level cache for usage data
# Structure: {file_path: {'mtime': float, 'data': list}}
_FILE_CACHE = {}


# Re-export filter function with Gemini-specific name for backward compatibility
def filter_gemini_usage_data_by_days(usage_data, days_back):
    """Filter Gemini usage data to only include entries from the last N days."""
    return filter_usage_data_by_days(usage_data, days_back)


def get_gemini_dir():
    """Get Gemini configuration directory."""
    gemini_dir = os.environ.get('GEMINI_CONFIG_DIR', os.path.expanduser('~/.gemini'))
    return Path(gemini_dir)


def _read_single_gemini_file(session_file):
    """Read a single Gemini session JSON file with caching based on mtime."""
    file_path_str = str(session_file)
    try:
        current_mtime = session_file.stat().st_mtime
    except Exception:
        return []

    # Check cache
    if file_path_str in _FILE_CACHE:
        cached = _FILE_CACHE[file_path_str]
        if cached['mtime'] == current_mtime:
            return cached['data']

    # Read and parse file
    file_data = []
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

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

            normalized_entry = {
                'timestamp': timestamp,
                'session_start_time': timestamp,
                'session_end_time': timestamp,
                'message': {
                    'model': model,
                    'usage': {
                        'input_tokens': non_cached_input,
                        'output_tokens': output_tokens,
                        'cache_read_input_tokens': cached_input,
                        'cache_creation_input_tokens': thoughts_tokens,
                    }
                }
            }
            # Pre-parse timestamp for faster filtering later
            parsed_ts = parse_timestamp(timestamp)
            if parsed_ts:
                normalized_entry['_parsed_timestamp'] = parsed_ts
            file_data.append(normalized_entry)

    except (json.JSONDecodeError, KeyError):
        return []
    except Exception:
        return []

    # Update cache
    _FILE_CACHE[file_path_str] = {'mtime': current_mtime, 'data': file_data}
    return file_data


def read_gemini_json_files(tmp_dir):
    """Read all session JSON files from Gemini tmp directory with caching.

    Gemini stores sessions in ~/.gemini/tmp/<project_hash>/chats/session-*.json

    Token data is in messages with type == "gemini" containing a tokens object:
    {
        "input": 8493,
        "output": 37,
        "cached": 3207,
        "thoughts": 190,
        "tool": 0,
        "total": 8720
    }

    Uses file mtime-based caching to avoid re-reading unchanged files.

    Returns normalized data similar to Claude format for compatibility:
    [
        {
            'timestamp': '2025-12-11T23:18:08.351Z',
            'session_start_time': '2025-12-11T23:18:08.351Z',
            'session_end_time': '2025-12-11T23:18:08.351Z',
            'message': {
                'model': 'gemini-3-pro-preview',
                'usage': {
                    'input_tokens': 5286,
                    'output_tokens': 37,
                    'cache_read_input_tokens': 3207,
                    'cache_creation_input_tokens': 190,
                }
            }
        },
        ...
    ]
    """
    usage_data = []

    for session_file in tmp_dir.rglob('chats/session-*.json'):
        file_data = _read_single_gemini_file(session_file)
        usage_data.extend(file_data)

    return usage_data
