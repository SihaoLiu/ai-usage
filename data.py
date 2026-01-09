"""Data reading and filtering functions for Claude Code usage analysis."""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta


def get_claude_dir():
    """Get Claude configuration directory."""
    claude_dir = os.environ.get('CLAUDE_CONFIG_DIR', os.path.expanduser('~/.claude'))
    return Path(claude_dir)


def read_jsonl_files(projects_dir):
    """Read all JSONL files from projects directory."""
    usage_data = []

    for jsonl_file in projects_dir.rglob('*.jsonl'):
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
                            usage_data.append(data)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return usage_data


def filter_usage_data_by_days(usage_data, days_back):
    """Filter usage data to only include entries from the last N days."""
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
