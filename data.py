"""Data reading and filtering functions for Claude Code usage analysis."""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta


def get_claude_dirs():
    """Get Claude configuration directories.

    Returns a list of paths to check:
    - If CLAUDE_CONFIG_DIR is set, uses those paths (comma-separated)
    - Otherwise, returns both XDG path (~/.config/claude) and legacy path (~/.claude)
    """
    claude_dir_env = os.environ.get('CLAUDE_CONFIG_DIR')
    if claude_dir_env:
        # Support comma-separated paths like ccusage does
        return [Path(p.strip()) for p in claude_dir_env.split(',') if p.strip()]

    # Return both XDG-compliant path (new default) and legacy path (old default)
    return [
        Path(os.path.expanduser('~/.config/claude')),  # New XDG-compliant path
        Path(os.path.expanduser('~/.claude')),          # Legacy path
    ]


def get_claude_dir():
    """Get Claude configuration directory (legacy function for compatibility)."""
    dirs = get_claude_dirs()
    # Return first directory that has a projects subdirectory
    for d in dirs:
        if (d / 'projects').exists():
            return d
    # Fallback to first directory
    return dirs[0] if dirs else Path(os.path.expanduser('~/.claude'))


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


def read_all_jsonl_files():
    """Read all JSONL files from all Claude configuration directories.

    Combines data from both XDG path (~/.config/claude/projects) and
    legacy path (~/.claude/projects), deduplicating entries by message ID + request ID.
    """
    usage_data = []
    seen_hashes = set()

    for claude_dir in get_claude_dirs():
        projects_dir = claude_dir / 'projects'
        if not projects_dir.exists():
            continue

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
                            if not (data.get('message') and data['message'].get('usage')):
                                continue

                            # Deduplicate by message ID + request ID (like ccusage does)
                            message_id = data.get('message', {}).get('id')
                            request_id = data.get('requestId')
                            if message_id and request_id:
                                hash_key = f"{message_id}:{request_id}"
                                if hash_key in seen_hashes:
                                    continue
                                seen_hashes.add(hash_key)

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
