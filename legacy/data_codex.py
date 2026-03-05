"""Data reading and filtering functions for Codex (OpenAI) usage analysis."""

import json
import os
from pathlib import Path

from time_utils import parse_timestamp, filter_usage_data_by_days

# File-level cache for usage data
# Structure: {file_path: {'mtime': float, 'data': list}}
_FILE_CACHE = {}

# Re-export filter function with Codex-specific name for backward compatibility
def filter_codex_usage_data_by_days(usage_data, days_back):
    """Filter Codex usage data to only include entries from the last N days."""
    return filter_usage_data_by_days(usage_data, days_back)


def get_codex_dir():
    """Get Codex configuration directory."""
    codex_dir = os.environ.get('CODEX_CONFIG_DIR', os.path.expanduser('~/.codex'))
    return Path(codex_dir)


def _read_single_codex_file(jsonl_file):
    """Read a single Codex JSONL file with caching based on mtime."""
    file_path_str = str(jsonl_file)
    try:
        current_mtime = jsonl_file.stat().st_mtime
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
        # First pass: find session start time
        session_start_time = None
        all_timestamps = []
        file_entries = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    file_entries.append(data)
                    entry_type = data.get('type')
                    timestamp = data.get('timestamp')

                    if timestamp:
                        all_timestamps.append(timestamp)

                    # Get session start time from session_meta
                    if entry_type == 'session_meta':
                        payload = data.get('payload', {})
                        if payload.get('timestamp'):
                            session_start_time = payload['timestamp']
                        elif timestamp:
                            session_start_time = timestamp
                except json.JSONDecodeError:
                    continue

        # Fallback: use first timestamp as session start if not found
        if not session_start_time and all_timestamps:
            session_start_time = min(all_timestamps)

        # Second pass: process entries and add session time span
        current_model = 'unknown'
        current_effort = 'unknown'
        last_token_usage = None

        for data in file_entries:
            entry_type = data.get('type')

            # Update model/effort from turn_context
            if entry_type == 'turn_context':
                payload = data.get('payload', {})
                if payload.get('model'):
                    current_model = payload['model']
                if payload.get('effort'):
                    current_effort = payload['effort']

            # Extract token usage from event_msg with token_count
            elif entry_type == 'event_msg':
                payload = data.get('payload', {})
                if payload.get('type') == 'token_count':
                    info = payload.get('info')
                    if info and 'last_token_usage' in info:
                        token_usage = info['last_token_usage']

                        # Skip if this is the same as last one (duplicates)
                        usage_key = (
                            token_usage.get('input_tokens', 0),
                            token_usage.get('cached_input_tokens', 0),
                            token_usage.get('output_tokens', 0),
                            token_usage.get('reasoning_output_tokens', 0)
                        )
                        if usage_key == last_token_usage:
                            continue
                        last_token_usage = usage_key

                        # Normalize to Claude-like format
                        # In Codex: input_tokens is TOTAL, cached is subset
                        total_input = token_usage.get('input_tokens', 0)
                        cached_input = token_usage.get('cached_input_tokens', 0)
                        non_cached_input = total_input - cached_input

                        total_output = token_usage.get('output_tokens', 0)
                        reasoning_output = token_usage.get('reasoning_output_tokens', 0)
                        non_reasoning_output = total_output - reasoning_output

                        entry_timestamp = data.get('timestamp')
                        normalized_entry = {
                            'timestamp': entry_timestamp,
                            'session_start_time': entry_timestamp,
                            'session_end_time': entry_timestamp,
                            'message': {
                                'model': current_model,
                                'effort': current_effort,
                                'usage': {
                                    'input_tokens': non_cached_input,
                                    'output_tokens': non_reasoning_output,
                                    'cache_read_input_tokens': cached_input,
                                    'reasoning_output_tokens': reasoning_output,
                                    'cache_creation_input_tokens': 0,
                                }
                            }
                        }
                        # Pre-parse timestamp for faster filtering later
                        if entry_timestamp:
                            parsed_ts = parse_timestamp(entry_timestamp)
                            if parsed_ts:
                                normalized_entry['_parsed_timestamp'] = parsed_ts
                        file_data.append(normalized_entry)

    except Exception:
        return []

    # Update cache
    _FILE_CACHE[file_path_str] = {'mtime': current_mtime, 'data': file_data}
    return file_data


def read_codex_jsonl_files(sessions_dir):
    """Read all JSONL files from Codex sessions directory with caching.

    Codex stores sessions in ~/.codex/sessions/YYYY/MM/DD/*.jsonl

    Token data is in event_msg entries with payload.type == "token_count"
    Model/effort info is in turn_context entries.
    Session start time is in session_meta entries.

    Uses file mtime-based caching to avoid re-reading unchanged files.

    Returns normalized data similar to Claude format for compatibility:
    [
        {
            'timestamp': '2025-12-11T23:18:08.351Z',
            'session_start_time': '2025-12-11T23:18:08.351Z',
            'session_end_time': '2025-12-11T23:18:08.351Z',
            'message': {
                'model': 'gpt-5-codex',
                'effort': 'high',
                'usage': {
                    'input_tokens': 984,
                    'output_tokens': 45,
                    'cache_read_input_tokens': 2048,
                    'reasoning_output_tokens': 0,
                }
            }
        },
        ...
    ]
    """
    usage_data = []

    for jsonl_file in sessions_dir.rglob('*.jsonl'):
        file_data = _read_single_codex_file(jsonl_file)
        usage_data.extend(file_data)

    return usage_data
