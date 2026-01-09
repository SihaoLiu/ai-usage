"""Data reading and filtering functions for Codex (OpenAI) usage analysis."""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta


def get_codex_dir():
    """Get Codex configuration directory."""
    codex_dir = os.environ.get('CODEX_CONFIG_DIR', os.path.expanduser('~/.codex'))
    return Path(codex_dir)


def read_codex_jsonl_files(sessions_dir):
    """Read all JSONL files from Codex sessions directory.

    Codex stores sessions in ~/.codex/sessions/YYYY/MM/DD/*.jsonl

    Token data is in event_msg entries with payload.type == "token_count"
    Model/effort info is in turn_context entries.
    Session start time is in session_meta entries.

    Returns normalized data similar to Claude format for compatibility:
    [
        {
            'timestamp': '2025-12-11T23:18:08.351Z',
            'session_start_time': '2025-12-11T23:18:08.351Z',  # Same as timestamp
            'session_end_time': '2025-12-11T23:18:08.351Z',    # Same as timestamp
            'message': {
                'model': 'gpt-5-codex',
                'effort': 'high',
                'usage': {
                    'input_tokens': 984,  # non-cached input
                    'output_tokens': 45,  # non-reasoning output
                    'cache_read_input_tokens': 2048,  # cached input
                    'reasoning_output_tokens': 0,  # reasoning tokens
                }
            }
        },
        ...
    ]
    """
    usage_data = []

    # Find all JSONL files in sessions directory (recursive)
    for jsonl_file in sessions_dir.rglob('*.jsonl'):
        try:
            # First pass: find session start time and last timestamp
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

            # Calculate session end time (latest timestamp in file)
            session_end_time = max(all_timestamps) if all_timestamps else None

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
                            # Use entry's own timestamp (no session-level spreading)
                            # Each token_count entry represents a specific API call
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
                                        # Codex doesn't have cache_creation, set to 0
                                        'cache_creation_input_tokens': 0,
                                    }
                                }
                            }
                            usage_data.append(normalized_entry)

        except Exception:
            continue

    return usage_data


def filter_codex_usage_data_by_days(usage_data, days_back):
    """Filter Codex usage data to only include entries from the last N days."""
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
