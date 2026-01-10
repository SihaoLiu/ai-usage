"""Data reading and filtering functions for Claude Code usage analysis."""

import json
import os
from pathlib import Path

from time_utils import parse_timestamp, filter_usage_data_by_days

# File-level cache for usage data
# Structure: {file_path: {'mtime': float, 'data': list}}
_FILE_CACHE = {}

# Re-export filter function for backward compatibility
__all__ = ['get_claude_dir', 'read_jsonl_files', 'filter_usage_data_by_days']


def get_claude_dir():
    """Get Claude configuration directory."""
    claude_dir = os.environ.get('CLAUDE_CONFIG_DIR', os.path.expanduser('~/.claude'))
    return Path(claude_dir)


def _read_single_jsonl_file(jsonl_file):
    """Read a single JSONL file with caching based on mtime."""
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
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Only include entries with usage data
                    if data.get('message') and data['message'].get('usage'):
                        timestamp = data.get('timestamp')
                        data['session_start_time'] = timestamp
                        data['session_end_time'] = timestamp
                        # Pre-parse timestamp for faster filtering later
                        if timestamp:
                            parsed_ts = parse_timestamp(timestamp)
                            if parsed_ts:
                                data['_parsed_timestamp'] = parsed_ts
                        file_data.append(data)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    # Update cache
    _FILE_CACHE[file_path_str] = {'mtime': current_mtime, 'data': file_data}
    return file_data


def read_jsonl_files(projects_dir):
    """Read all JSONL files from projects directory with caching.

    For Claude, each usage entry represents a single API call with its own timestamp.
    We use each entry's own timestamp for time bucketing (no session-level spreading).
    This ensures token consumption is attributed to the actual time of the API call.

    Uses file mtime-based caching to avoid re-reading unchanged files.
    """
    usage_data = []

    for jsonl_file in projects_dir.rglob('*.jsonl'):
        file_data = _read_single_jsonl_file(jsonl_file)
        usage_data.extend(file_data)

    return usage_data
