#!/usr/bin/env python3
"""Main entry point for Claude Code / Codex usage analysis."""

import os
import sys
import argparse
import time
import select
import subprocess
import re
import shutil
import signal
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError

# Claude imports
from data import get_claude_dir, read_jsonl_files, filter_usage_data_by_days
from stats import calculate_model_breakdown, calculate_model_token_breakdown_time_series
# Codex imports
from data_codex import get_codex_dir, read_codex_jsonl_files, filter_codex_usage_data_by_days
from stats_codex import calculate_codex_model_breakdown, calculate_codex_model_token_breakdown_time_series
# Gemini imports
from data_gemini import get_gemini_dir, read_gemini_json_files, filter_gemini_usage_data_by_days
from stats_gemini import calculate_gemini_model_breakdown, calculate_gemini_model_token_breakdown_time_series
# Shared imports
from formatting import print_model_breakdown, format_cost_per_mtok
from charts import print_multi_line_chart, print_vendor_comparison_chart
from constants import (
    SUBSCRIPTION_PRICE, CODEX_SUBSCRIPTION_PRICE, GEMINI_SUBSCRIPTION_PRICE,
    MODEL_PRICING, DEFAULT_PRICING,
    CODEX_MODEL_PRICING, CODEX_DEFAULT_PRICING,
    GEMINI_MODEL_PRICING, GEMINI_DEFAULT_PRICING,
)


# Minimum terminal size requirements
MIN_TERMINAL_WIDTH = 60
MIN_TERMINAL_HEIGHT = 35


def get_terminal_width():
    """Get current terminal width."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80  # Fallback default


def get_terminal_height():
    """Get current terminal height."""
    try:
        return shutil.get_terminal_size().lines
    except Exception:
        return 24  # Fallback default


def check_terminal_size():
    """Check if terminal meets minimum size requirements.

    Returns:
        tuple: (is_ok, width, height) where is_ok is True if terminal is large enough
    """
    width = get_terminal_width()
    height = get_terminal_height()
    is_ok = width >= MIN_TERMINAL_WIDTH and height >= MIN_TERMINAL_HEIGHT
    return is_ok, width, height


def print_terminal_too_small(width, height):
    """Print centered message when terminal is too small (similar to btop)."""
    # Clear screen
    os.system('clear' if os.name != 'nt' else 'cls')

    # Build the message lines
    lines = [
        "Terminal size too small:",
        f"  Width = {width}  Height = {height}",
        "",
        "Needed for current config:",
        f"  Width = {MIN_TERMINAL_WIDTH}  Height = {MIN_TERMINAL_HEIGHT}",
    ]

    # Find the maximum line length for centering
    max_line_len = max(len(line) for line in lines)

    # Calculate vertical padding to center the message
    total_lines = len(lines)
    top_padding = max(0, (height - total_lines) // 2)

    # Print vertical padding
    for _ in range(top_padding):
        print()

    # Print each line centered horizontally
    for line in lines:
        left_padding = max(0, (width - max_line_len) // 2)
        print(" " * left_padding + line)


def get_chart_target_width():
    """Get target chart width (99% of terminal width)."""
    terminal_width = get_terminal_width()
    return int(terminal_width * 0.99)


def calculate_chart_height(is_monitor_mode=False, table_printed=True):
    """Calculate optimal chart height based on terminal height.

    The terminal layout consists of:
    - Header section: ~5 lines (title, days, monitor info, blank)
    - Breakdown table: ~15 lines (header, models, totals, costs) - if printed
    - Chart 1 (IO): overhead (6) + chart_height
    - Chart 2 (Cache): overhead (13) + chart_height (includes x-axis labels + legend)
    - Final spacing: 1 line
    - Monitor prompt: ~4 lines (if monitor mode)

    Each chart gets half of the remaining available height.

    Args:
        is_monitor_mode: Whether running in monitor mode (needs prompt space)
        table_printed: Whether the breakdown table was printed (False if hidden)
    """
    terminal_height = get_terminal_height()

    # Fixed overhead for non-chart elements
    header_lines = 5
    breakdown_table_lines = 15 if table_printed else 0  # Table may be hidden
    chart1_overhead = 6   # blank, title, =, weekday, date, x-axis
    chart2_overhead = 13  # same as chart1 + x-axis labels (~2) + blank + = + info + legend
    final_lines = 1
    monitor_prompt_lines = 4 if is_monitor_mode else 0

    fixed_overhead = (header_lines + breakdown_table_lines +
                      chart1_overhead + chart2_overhead +
                      final_lines + monitor_prompt_lines)

    # Available height for both chart bodies
    available_height = terminal_height - fixed_overhead

    # Each chart gets half the available height
    chart_height = available_height // 2

    # Enforce minimum and maximum bounds
    min_height = 10
    max_height = 60  # Prevent excessively tall charts

    return max(min_height, min(max_height, chart_height))


def calculate_optimal_interval_minutes(days_back, target_width):
    """Calculate optimal interval in minutes based on days and terminal width.

    The interval is the larger of:
    1. 1% of total time range (minimum granularity per user requirement)
    2. Interval needed to fit within terminal width
    """
    total_minutes = days_back * 24 * 60

    # 1% of total time range (minimum granularity)
    min_interval = total_minutes / 100

    # Terminal-based: calculate how many data points fit
    y_axis_width = 7
    chart_width = target_width - y_axis_width
    # Account for day separators (roughly days_back separators)
    chart_width -= days_back

    if chart_width <= 0:
        chart_width = 50  # fallback

    # Each data point should occupy at least 1 column
    terminal_interval = total_minutes / chart_width

    # Use the larger interval (coarser granularity)
    return max(min_interval, terminal_interval)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze Claude Code usage statistics')
    parser.add_argument('--days', type=int, default=3,
                        help='Number of days to look back (default: 3)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit (default: monitor mode with 1 hour refresh)')
    parser.add_argument('--vendor', choices=['claude', 'codex', 'gemini', 'all'], default='all',
                        help='Vendor to collect statistics from (default: all). Use "all" to compare all vendors.')
    args = parser.parse_args()

    # Mutable state for vendor and days (allows switching in monitor mode)
    # Using a dict to allow modification in nested functions
    vendor_state = {
        'vendor': args.vendor,
        'data_dir': None,
        'vendor_name': None
    }
    days_state = {
        'days': args.days
    }
    monitor_state = {
        'interval': 3600  # Default: 1 hour
    }

    def update_vendor_state(vendor):
        """Update vendor state based on vendor choice."""
        if vendor == 'codex':
            vendor_state['data_dir'] = get_codex_dir() / 'sessions'
            vendor_state['vendor_name'] = "Codex"
        elif vendor == 'gemini':
            vendor_state['data_dir'] = get_gemini_dir() / 'tmp'
            vendor_state['vendor_name'] = "Gemini CLI"
        elif vendor == 'all':
            vendor_state['data_dir'] = None  # Not used for 'all'
            vendor_state['vendor_name'] = "All Vendors"
        else:
            vendor_state['data_dir'] = get_claude_dir() / 'projects'
            vendor_state['vendor_name'] = "Claude Code"
        vendor_state['vendor'] = vendor

    # Initialize vendor state
    update_vendor_state(args.vendor)

    # For 'all' vendor, we don't check data_dir (handled separately)
    if args.vendor != 'all' and not vendor_state['data_dir'].exists():
        print(f"Error: Data directory not found at {vendor_state['data_dir']}")
        sys.exit(1)

    def get_latest_claude_version_from_npm():
        """Fetch the latest version from Claude Code's NPM registry."""
        import json
        npm_url = "https://registry.npmjs.org/@anthropic-ai/claude-code/latest"
        try:
            with urlopen(npm_url, timeout=10) as response:
                content = response.read().decode('utf-8')
                data = json.loads(content)
                return data.get('version')
        except (URLError, OSError, TimeoutError, json.JSONDecodeError):
            pass
        return None

    def get_latest_codex_version_from_npm():
        """Fetch the latest version from Codex's NPM registry."""
        import json
        npm_url = "https://registry.npmjs.org/@openai/codex/latest"
        try:
            with urlopen(npm_url, timeout=10) as response:
                content = response.read().decode('utf-8')
                data = json.loads(content)
                return data.get('version')
        except (URLError, OSError, TimeoutError, json.JSONDecodeError):
            pass
        return None

    def get_claude_version():
        """Get the Claude Code version string with update status."""
        current_version = None
        try:
            result = subprocess.run(
                ['claude', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Output is like "2.0.61 (Claude Code)"
                current_version = result.stdout.strip().split()[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        if not current_version:
            return "Claude Code"

        # Check for latest version
        latest_version = get_latest_claude_version_from_npm()

        if latest_version is None:
            # Couldn't fetch latest version, just show current
            return f"Claude Code ({current_version})"
        elif current_version == latest_version:
            return f"Claude Code ({current_version}, up-to-date)"
        else:
            return f"Claude Code ({current_version}, a newer version {latest_version} available)"

    def get_codex_version():
        """Get the Codex version string with update status."""
        current_version = None
        try:
            result = subprocess.run(
                ['codex', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Output may be like "0.72.0" or similar
                output = result.stdout.strip()
                # Extract version number
                match = re.search(r'(\d+\.\d+\.\d+)', output)
                if match:
                    current_version = match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        if not current_version:
            return "Codex"

        # Check for latest version
        latest_version = get_latest_codex_version_from_npm()

        if latest_version is None:
            # Couldn't fetch latest version, just show current
            return f"Codex ({current_version})"
        elif current_version == latest_version:
            return f"Codex ({current_version}, up-to-date)"
        else:
            return f"Codex ({current_version}, a newer version {latest_version} available)"

    def get_latest_gemini_version_from_npm():
        """Fetch the latest version from Gemini CLI's NPM registry."""
        import json
        npm_url = "https://registry.npmjs.org/@google/gemini-cli/latest"
        try:
            with urlopen(npm_url, timeout=10) as response:
                content = response.read().decode('utf-8')
                data = json.loads(content)
                return data.get('version')
        except (URLError, OSError, TimeoutError, json.JSONDecodeError):
            pass
        return None

    def get_gemini_version():
        """Get the Gemini CLI version string with update status."""
        current_version = None
        try:
            result = subprocess.run(
                ['gemini', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                # Extract version number (e.g., "0.1.2" or similar)
                match = re.search(r'(\d+\.\d+\.\d+)', output)
                if match:
                    current_version = match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        if not current_version:
            return "Gemini CLI"

        # Check for latest version
        latest_version = get_latest_gemini_version_from_npm()

        if latest_version is None:
            # Couldn't fetch latest version, just show current
            return f"Gemini CLI ({current_version})"
        elif current_version == latest_version:
            return f"Gemini CLI ({current_version}, up-to-date)"
        else:
            return f"Gemini CLI ({current_version}, a newer version {latest_version} available)"

    def calculate_weighted_cost_per_mtok():
        """Calculate weighted average cost per MTok and total savings across all vendors.

        Returns tuple: (weighted_cost_per_mtok, total_monthly_savings, vendor_info)
        where vendor_info is dict of {vendor: (tokens, percentage, cost_per_mtok)}
        """
        from collections import defaultdict

        vendor_data = {}  # {vendor: {'tokens': total, 'api_cost': cost}}

        # Helper to calculate API cost for a vendor
        def calculate_api_cost(filtered_data, pricing_table, default_pricing, vendor_type):
            """Calculate total API cost for filtered usage data."""
            # Group by model first
            model_stats = defaultdict(lambda: {
                'input': 0, 'output': 0, 'cache_read': 0, 'cache_creation': 0
            })
            for entry in filtered_data:
                model = entry.get('message', {}).get('model', 'unknown')
                usage = entry.get('message', {}).get('usage', {})
                model_stats[model]['input'] += usage.get('input_tokens', 0)
                model_stats[model]['output'] += usage.get('output_tokens', 0)
                model_stats[model]['cache_read'] += usage.get('cache_read_input_tokens', 0)
                if vendor_type == 'codex':
                    model_stats[model]['cache_creation'] += usage.get('reasoning_output_tokens', 0)
                elif vendor_type == 'gemini':
                    model_stats[model]['cache_creation'] += usage.get('thinking_output_tokens', 0)
                else:
                    model_stats[model]['cache_creation'] += usage.get('cache_creation_input_tokens', 0)

            # Calculate cost
            total_cost = 0
            for model, stats in model_stats.items():
                # For Codex, extract base model name for pricing lookup
                if vendor_type == 'codex' and ' (' in model and model.endswith(')'):
                    base_model = model.rsplit(' (', 1)[0]
                else:
                    base_model = model
                pricing = pricing_table.get(base_model, default_pricing)
                total_cost += stats['input'] * pricing['input'] / 1_000_000
                total_cost += stats['output'] * pricing['output'] / 1_000_000
                total_cost += stats['cache_read'] * pricing['cache_input'] / 1_000_000
                if vendor_type in ('codex', 'gemini'):
                    # Reasoning/thinking tokens billed at output rate
                    total_cost += stats['cache_creation'] * pricing['output'] / 1_000_000
                else:
                    total_cost += stats['cache_creation'] * pricing['cache_output'] / 1_000_000
            return total_cost

        # Get Claude data
        claude_dir = get_claude_dir() / 'projects'
        if claude_dir.exists():
            claude_data = read_jsonl_files(claude_dir)
            claude_filtered = filter_usage_data_by_days(claude_data, days_state['days'])
            total = 0
            for entry in claude_filtered:
                usage = entry.get('message', {}).get('usage', {})
                total += (usage.get('input_tokens', 0) +
                         usage.get('output_tokens', 0) +
                         usage.get('cache_read_input_tokens', 0) +
                         usage.get('cache_creation_input_tokens', 0))
            if total > 0:
                api_cost = calculate_api_cost(claude_filtered, MODEL_PRICING, DEFAULT_PRICING, 'claude')
                vendor_data['Claude'] = {'tokens': total, 'api_cost': api_cost}

        # Get Codex data
        codex_dir = get_codex_dir() / 'sessions'
        if codex_dir.exists():
            codex_data = read_codex_jsonl_files(codex_dir)
            codex_filtered = filter_codex_usage_data_by_days(codex_data, days_state['days'])
            total = 0
            for entry in codex_filtered:
                usage = entry.get('message', {}).get('usage', {})
                total += (usage.get('input_tokens', 0) +
                         usage.get('output_tokens', 0) +
                         usage.get('cache_read_input_tokens', 0) +
                         usage.get('reasoning_output_tokens', 0))
            if total > 0:
                api_cost = calculate_api_cost(codex_filtered, CODEX_MODEL_PRICING, CODEX_DEFAULT_PRICING, 'codex')
                vendor_data['Codex'] = {'tokens': total, 'api_cost': api_cost}

        # Get Gemini data
        gemini_dir = get_gemini_dir() / 'sessions'
        if gemini_dir.exists():
            gemini_data = read_gemini_json_files(gemini_dir)
            gemini_filtered = filter_gemini_usage_data_by_days(gemini_data, days_state['days'])
            total = 0
            for entry in gemini_filtered:
                usage = entry.get('message', {}).get('usage', {})
                total += (usage.get('input_tokens', 0) +
                         usage.get('output_tokens', 0) +
                         usage.get('cache_read_input_tokens', 0) +
                         usage.get('thinking_output_tokens', 0))
            if total > 0:
                api_cost = calculate_api_cost(gemini_filtered, GEMINI_MODEL_PRICING, GEMINI_DEFAULT_PRICING, 'gemini')
                vendor_data['Gemini'] = {'tokens': total, 'api_cost': api_cost}

        grand_total = sum(v['tokens'] for v in vendor_data.values())
        if grand_total == 0:
            return 0, 0, {}

        # Subscription prices per vendor
        subscription_prices = {
            'Claude': SUBSCRIPTION_PRICE,
            'Codex': CODEX_SUBSCRIPTION_PRICE,
            'Gemini': GEMINI_SUBSCRIPTION_PRICE,
        }

        # Calculate cost per MTok and savings for each vendor
        vendor_info = {}
        weighted_cost = 0
        total_monthly_savings = 0

        for vendor, data in vendor_data.items():
            tokens = data['tokens']
            api_cost = data['api_cost']
            percentage = tokens / grand_total
            monthly_tokens = (tokens / days_state['days']) * 30 if days_state['days'] > 0 else 0

            if monthly_tokens > 0:
                cost_per_mtok = subscription_prices[vendor] / (monthly_tokens / 1_000_000)
            else:
                cost_per_mtok = 0

            # Calculate savings: monthly_api_cost - subscription_price
            daily_api_cost = api_cost / days_state['days'] if days_state['days'] > 0 else 0
            monthly_api_cost = daily_api_cost * 30
            savings = monthly_api_cost - subscription_prices[vendor]

            vendor_info[vendor] = (tokens, percentage, cost_per_mtok)
            weighted_cost += percentage * cost_per_mtok
            total_monthly_savings += savings

        return weighted_cost, total_monthly_savings, vendor_info

    def get_vendor_version():
        """Get the version string for the current vendor."""
        if vendor_state['vendor'] == 'codex':
            return get_codex_version()
        elif vendor_state['vendor'] == 'gemini':
            return get_gemini_version()
        elif vendor_state['vendor'] == 'all':
            weighted_cost, total_savings, _ = calculate_weighted_cost_per_mtok()
            if weighted_cost > 0:
                return f"All Vendors Comparison, {format_cost_per_mtok(weighted_cost)} / MTok, Monthly Saving ${total_savings:.2f}"
            return "All Vendors Comparison"
        else:
            return get_claude_version()

    def calculate_vendor_aggregate_time_series(interval_minutes=60):
        """Calculate total token usage per vendor over time.

        This version distributes tokens evenly across the session time span to produce
        smoother charts. Long-running sessions will have their token usage spread across
        all intervals they span, rather than being concentrated at a single timestamp.

        Returns a time series where each time interval contains per-vendor totals:
        {
            interval_time: {
                'Claude': total_tokens,
                'Codex': total_tokens,
                'Gemini': total_tokens,
            },
            ...
        }
        """
        from collections import defaultdict
        from datetime import timedelta

        # Get local timezone
        local_tz = datetime.now().astimezone().tzinfo

        time_series = defaultdict(lambda: defaultdict(float))

        def to_interval(dt):
            """Round datetime to the nearest interval boundary."""
            total_minutes = dt.hour * 60 + dt.minute
            interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
            interval_hour = interval_start_minutes // 60
            interval_minute = interval_start_minutes % 60
            return dt.replace(hour=interval_hour, minute=interval_minute, second=0, microsecond=0)

        def distribute_tokens(session_start_str, session_end_str, total_tokens):
            """Distribute tokens evenly across intervals within session time span."""
            try:
                start = datetime.fromisoformat(session_start_str.replace('Z', '+00:00'))
                end = datetime.fromisoformat(session_end_str.replace('Z', '+00:00'))
                start_local = start.astimezone(local_tz)
                end_local = end.astimezone(local_tz)
            except Exception:
                return []

            start_interval = to_interval(start_local)
            end_interval = to_interval(end_local)

            intervals = []
            current = start_interval
            while current <= end_interval:
                intervals.append(current)
                current += timedelta(minutes=interval_minutes)

            if not intervals:
                return []

            tokens_per_interval = total_tokens / len(intervals)
            return [(interval_time, tokens_per_interval) for interval_time in intervals]

        # Helper to process usage data and add to time series
        def process_usage_data(usage_data, vendor_label):
            for entry in usage_data:
                timestamp_str = entry.get('timestamp')
                if not timestamp_str:
                    continue

                usage = entry['message']['usage']
                # Sum all token types
                total = (usage.get('input_tokens', 0) +
                         usage.get('output_tokens', 0) +
                         usage.get('cache_read_input_tokens', 0) +
                         usage.get('cache_creation_input_tokens', 0))

                # Get session time span (if available)
                session_start = entry.get('session_start_time', timestamp_str)
                session_end = entry.get('session_end_time', timestamp_str)

                # Distribute tokens across intervals
                distributed = distribute_tokens(session_start, session_end, total)

                if distributed:
                    for interval_time, tokens in distributed:
                        time_series[interval_time][vendor_label] += tokens
                else:
                    # Fallback: use original timestamp-based bucketing
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamp_local = timestamp.astimezone(local_tz)
                        interval_time = to_interval(timestamp_local)
                        time_series[interval_time][vendor_label] += total
                    except Exception:
                        continue

        # Read and process Claude data
        claude_dir = get_claude_dir() / 'projects'
        if claude_dir.exists():
            claude_data = read_jsonl_files(claude_dir)
            claude_filtered = filter_usage_data_by_days(claude_data, days_state['days'])
            process_usage_data(claude_filtered, 'Claude')

        # Read and process Codex data
        codex_dir = get_codex_dir() / 'sessions'
        if codex_dir.exists():
            codex_data = read_codex_jsonl_files(codex_dir)
            codex_filtered = filter_codex_usage_data_by_days(codex_data, days_state['days'])
            process_usage_data(codex_filtered, 'Codex')

        # Read and process Gemini data
        gemini_dir = get_gemini_dir() / 'tmp'
        if gemini_dir.exists():
            gemini_data = read_gemini_json_files(gemini_dir)
            gemini_filtered = filter_gemini_usage_data_by_days(gemini_data, days_state['days'])
            process_usage_data(gemini_filtered, 'Gemini')

        return dict(time_series)

    def print_stats_all():
        """Print vendor comparison statistics (for --vendor all mode)."""
        # Clear screen in monitor mode
        if not args.once:
            os.system('clear' if os.name != 'nt' else 'cls')

        print("Calculating usage across all vendors...")
        print(f"Showing data from last {days_state['days']} days")
        if not args.once:
            print(f"Monitor mode: Refreshing every {monitor_state['interval']} seconds (Press Ctrl+C to exit)")

        # Get terminal dimensions
        target_width = get_chart_target_width()
        chart_height = calculate_chart_height(
            is_monitor_mode=not args.once,
            table_printed=False  # No table in 'all' mode
        )

        # Calculate optimal interval
        optimal_interval = calculate_optimal_interval_minutes(days_state['days'], target_width)
        nice_intervals = [1, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440]
        interval_minutes = min(nice_intervals, key=lambda x: abs(x - optimal_interval) if x >= optimal_interval else float('inf'))
        if interval_minutes < optimal_interval:
            for ni in nice_intervals:
                if ni >= optimal_interval:
                    interval_minutes = ni
                    break
            else:
                interval_minutes = nice_intervals[-1]

        # Calculate vendor aggregate time series
        vendor_time_series = calculate_vendor_aggregate_time_series(interval_minutes=interval_minutes)

        if not vendor_time_series:
            print("No usage data found from any vendor.")
            return False

        # Calculate time span info for display
        from datetime import timedelta
        now = datetime.now()
        start_time = now - timedelta(days=days_state['days'])
        # Round to interval
        total_minutes = start_time.hour * 60 + start_time.minute
        interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
        start_time_rounded = start_time.replace(
            hour=interval_start_minutes // 60,
            minute=interval_start_minutes % 60,
            second=0, microsecond=0
        )
        # Count data points
        data_points = 0
        current_time = start_time_rounded
        while current_time <= now:
            data_points += 1
            current_time += timedelta(minutes=interval_minutes)

        # Format interval string
        if interval_minutes >= 60:
            if interval_minutes % 60 == 0:
                interval_str = f"{interval_minutes // 60}h"
            else:
                interval_str = f"{interval_minutes // 60}h{interval_minutes % 60}m"
        else:
            interval_str = f"{interval_minutes}m"

        # Print time span info
        terminal_width = get_terminal_width()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_str_full = now.strftime('%Y-%m-%d %H:%M')
        end_str_full = start_time_rounded.strftime('%Y-%m-%d %H:%M')
        start_str_short = now.strftime('%m/%d %H:%M')
        end_str_short = start_time_rounded.strftime('%m/%d %H:%M')
        now_str_short = datetime.now().strftime('%m/%d %H:%M:%S')

        full_line = f"Last updated: {now_str} | Time span: {start_str_full} to {end_str_full} | Interval: {interval_str} | Data points: {data_points}"
        short_line = f"Updated: {now_str_short} | Span: {start_str_short} - {end_str_short} | {interval_str} | {data_points} dp"

        if terminal_width >= len(full_line):
            print(full_line)
        else:
            print(short_line)
        print()

        # Print vendor comparison chart
        print_vendor_comparison_chart(
            vendor_time_series,
            height=chart_height * 2,  # Use more height since no table
            days_back=days_state['days'],
            target_width=target_width,
            interval_minutes=interval_minutes,
            show_legend=True
        )

        return True

    def print_stats():
        """Print all statistics (for both one-time and monitor mode).

        Returns:
            bool: True if stats were printed, False if no data,
                  None if terminal too small
        """
        # Check terminal size first
        size_ok, width, height = check_terminal_size()
        if not size_ok:
            print_terminal_too_small(width, height)
            return None  # Signal terminal too small

        # Handle 'all' vendor mode separately
        if vendor_state['vendor'] == 'all':
            return print_stats_all()

        # Clear screen in monitor mode
        if not args.once:
            os.system('clear' if os.name != 'nt' else 'cls')

        current_vendor = vendor_state['vendor']
        current_data_dir = vendor_state['data_dir']
        current_vendor_name = vendor_state['vendor_name']

        print(f"Calculating {current_vendor_name} usage...")
        print(f"Showing data from last {days_state['days']} days")
        if not args.once:
            print(f"Monitor mode: Refreshing every {monitor_state['interval']} seconds (Press Ctrl+C to exit)")

        # Read data based on vendor
        if current_vendor == 'codex':
            usage_data = read_codex_jsonl_files(current_data_dir)
        elif current_vendor == 'gemini':
            usage_data = read_gemini_json_files(current_data_dir)
        else:
            usage_data = read_jsonl_files(current_data_dir)

        if not usage_data:
            print("No usage data found.")
            return False

        # Filter data based on days parameter
        if current_vendor == 'codex':
            filtered_usage_data = filter_codex_usage_data_by_days(usage_data, days_state['days'])
        elif current_vendor == 'gemini':
            filtered_usage_data = filter_gemini_usage_data_by_days(usage_data, days_state['days'])
        else:
            filtered_usage_data = filter_usage_data_by_days(usage_data, days_state['days'])

        if not filtered_usage_data:
            print(f"No usage data found in the last {days_state['days']} days.")
            return False

        # Get terminal dimensions
        terminal_width = get_terminal_width()
        terminal_height = get_terminal_height()

        # Calculate and print statistics using filtered data
        if current_vendor == 'codex':
            model_stats = calculate_codex_model_breakdown(filtered_usage_data)
        elif current_vendor == 'gemini':
            model_stats = calculate_gemini_model_breakdown(filtered_usage_data)
        else:
            model_stats = calculate_model_breakdown(filtered_usage_data)

        table_printed = print_model_breakdown(
            model_stats,
            days_in_data=days_state['days'],
            terminal_width=terminal_width,
            terminal_height=terminal_height,
            vendor=current_vendor
        )

        # Get target width and height for charts
        target_width = get_chart_target_width()
        chart_height = calculate_chart_height(
            is_monitor_mode=not args.once,
            table_printed=table_printed
        )

        # Calculate optimal interval based on days and terminal width
        # Minimum granularity is 1% of total time range
        optimal_interval = calculate_optimal_interval_minutes(days_state['days'], target_width)
        # Round to a nice value (1, 5, 10, 15, 30, 60, 120, 240, 480 minutes)
        nice_intervals = [1, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440]
        interval_minutes = min(nice_intervals, key=lambda x: abs(x - optimal_interval) if x >= optimal_interval else float('inf'))
        if interval_minutes < optimal_interval:
            # Find the next larger nice interval
            for ni in nice_intervals:
                if ni >= optimal_interval:
                    interval_minutes = ni
                    break
            else:
                interval_minutes = nice_intervals[-1]

        # Calculate and print token breakdown time series (multi-line charts by model)
        if current_vendor == 'codex':
            model_breakdown_time_series = calculate_codex_model_token_breakdown_time_series(
                filtered_usage_data, interval_minutes=interval_minutes
            )
        elif current_vendor == 'gemini':
            model_breakdown_time_series = calculate_gemini_model_token_breakdown_time_series(
                filtered_usage_data, interval_minutes=interval_minutes
            )
        else:
            model_breakdown_time_series = calculate_model_token_breakdown_time_series(
                filtered_usage_data, interval_minutes=interval_minutes
            )

        # Extract model names that passed the threshold filter (from model_stats)
        # Only these models should appear in charts
        included_models = {stats['model'] for stats in model_stats}

        # Calculate time span info for display
        if model_breakdown_time_series:
            from datetime import timedelta
            now = datetime.now()
            start_time = now - timedelta(days=days_state['days'])
            # Round to interval
            total_minutes = start_time.hour * 60 + start_time.minute
            interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
            start_time_rounded = start_time.replace(
                hour=interval_start_minutes // 60,
                minute=interval_start_minutes % 60,
                second=0, microsecond=0
            )
            # Count data points
            data_points = 0
            current_time = start_time_rounded
            while current_time <= now:
                data_points += 1
                current_time += timedelta(minutes=interval_minutes)

            # Format interval string
            if interval_minutes >= 60:
                if interval_minutes % 60 == 0:
                    interval_str = f"{interval_minutes // 60}h"
                else:
                    interval_str = f"{interval_minutes // 60}h{interval_minutes % 60}m"
            else:
                interval_str = f"{interval_minutes}m"

            # Print time span info line (dynamic based on terminal width)
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Note: start is most recent (left), end is oldest (right) - matches chart
            start_str_full = now.strftime('%Y-%m-%d %H:%M')
            end_str_full = start_time_rounded.strftime('%Y-%m-%d %H:%M')
            start_str_short = now.strftime('%m/%d %H:%M')
            end_str_short = start_time_rounded.strftime('%m/%d %H:%M')
            now_str_short = datetime.now().strftime('%m/%d %H:%M:%S')

            # Full version
            full_line = f"Last updated: {now_str} | Time span: {start_str_full} to {end_str_full} | Interval: {interval_str} | Data points: {data_points}"
            # Short version
            short_line = f"Updated: {now_str_short} | Span: {start_str_short} - {end_str_short} | {interval_str} | {data_points} dp"

            if terminal_width >= len(full_line):
                print(full_line)
            else:
                print(short_line)
            print()

        # Print two separate charts: I/O tokens and Cache/Reasoning tokens
        # Height is dynamically calculated based on terminal height
        print_multi_line_chart(model_breakdown_time_series, height=chart_height, days_back=days_state['days'],
                               chart_type='io', show_x_axis=False, target_width=target_width,
                               interval_minutes=interval_minutes, vendor=current_vendor,
                               included_models=included_models, show_legend=True)
        print_multi_line_chart(model_breakdown_time_series, height=chart_height, days_back=days_state['days'],
                               chart_type='cache', show_x_axis=True, target_width=target_width,
                               interval_minutes=interval_minutes, vendor=current_vendor,
                               included_models=included_models, show_legend=True)

        return True

    # Monitor mode: interactive continuous refresh
    if not args.once:
        print("\n" + "=" * get_terminal_width())
        print("Interactive Monitor Mode (type h for help)")
        print("=" * get_terminal_width())
        print(f"Auto-refresh: {monitor_state['interval']}s | Vendor: {vendor_state['vendor']} | Days: {days_state['days']}")
        print("=" * get_terminal_width() + "\n")

        # Track terminal resize
        terminal_resized = [False]  # Use list for mutable state in closure
        last_terminal_width = [get_terminal_width()]
        last_terminal_height = [get_terminal_height()]

        def handle_sigwinch(signum, frame):
            """Handle terminal resize signal."""
            current_width = get_terminal_width()
            current_height = get_terminal_height()
            if current_width != last_terminal_width[0] or current_height != last_terminal_height[0]:
                terminal_resized[0] = True
                last_terminal_width[0] = current_width
                last_terminal_height[0] = current_height

        # Register SIGWINCH handler (only on Unix-like systems)
        if hasattr(signal, 'SIGWINCH'):
            signal.signal(signal.SIGWINCH, handle_sigwinch)

        # Track if terminal is currently too small
        terminal_too_small = [False]

        # Initial display
        result = print_stats()
        terminal_too_small[0] = (result is None)

        next_refresh_time = time.time() + monitor_state['interval']

        def show_prompt():
            """Display the command prompt (only if terminal is large enough)."""
            if terminal_too_small[0]:
                return  # Don't show prompt when terminal is too small
            print("\n" + get_vendor_version())
            print("-" * get_terminal_width())
            print("> ", end='', flush=True)

        # Show initial prompt
        show_prompt()

        try:
            while True:
                now = time.time()

                # Check if terminal was resized
                if terminal_resized[0]:
                    terminal_resized[0] = False
                    # Clear the current line (prompt) only if we were showing one
                    if not terminal_too_small[0]:
                        print("\r" + " " * (get_terminal_width() + 2) + "\r", end='')
                        print("-" * get_terminal_width())
                        print("\n" + "=" * get_terminal_width())
                        print(f"TERMINAL RESIZED (width: {last_terminal_width[0]}, height: {last_terminal_height[0]})")
                        print("=" * get_terminal_width() + "\n")
                    result = print_stats()
                    terminal_too_small[0] = (result is None)
                    next_refresh_time = time.time() + monitor_state['interval']
                    show_prompt()
                    continue

                # Check if it's time for auto-refresh
                if now >= next_refresh_time:
                    # Clear the current line (prompt) only if we were showing one
                    if not terminal_too_small[0]:
                        print("\r" + " " * (get_terminal_width() + 2) + "\r", end='')
                        print("-" * get_terminal_width())
                        print("\n" + "=" * get_terminal_width())
                        print("AUTO-REFRESH")
                        print("=" * get_terminal_width() + "\n")
                    result = print_stats()
                    terminal_too_small[0] = (result is None)
                    next_refresh_time = time.time() + monitor_state['interval']
                    show_prompt()

                # Wait for input with timeout using select
                time_until_refresh = next_refresh_time - time.time()
                timeout = min(1.0, max(0.1, time_until_refresh))

                ready, _, _ = select.select([sys.stdin], [], [], timeout)

                if ready:
                    raw_input = sys.stdin.readline()
                    # Ctrl-D (EOF) returns empty string without newline
                    if raw_input == '':
                        print("\n" + "-" * get_terminal_width())
                        print("\nExiting monitor mode...")
                        break
                    command = raw_input.strip()

                    if command in ("refresh", "r"):
                        print("-" * get_terminal_width())
                        print("\n" + "=" * get_terminal_width())
                        print("MANUAL REFRESH")
                        print("=" * get_terminal_width() + "\n")
                        result = print_stats()
                        terminal_too_small[0] = (result is None)
                        # Reset auto-refresh timer
                        next_refresh_time = time.time() + monitor_state['interval']
                        show_prompt()
                    elif command.startswith("vendor") or command.startswith("v "):
                        # Parse vendor command
                        parts = command.split()
                        if len(parts) == 2 and parts[1] in ('claude', 'codex', 'gemini', 'all'):
                            new_vendor = parts[1]
                            if new_vendor == vendor_state['vendor']:
                                print(f"Already monitoring {new_vendor}.")
                                show_prompt()
                            else:
                                # Check if new vendor's data directory exists (skip for 'all')
                                if new_vendor == 'all':
                                    new_data_dir = None  # Not used for 'all'
                                elif new_vendor == 'codex':
                                    new_data_dir = get_codex_dir() / 'sessions'
                                elif new_vendor == 'gemini':
                                    new_data_dir = get_gemini_dir() / 'tmp'
                                else:
                                    new_data_dir = get_claude_dir() / 'projects'

                                if new_data_dir is not None and not new_data_dir.exists():
                                    print(f"Error: Data directory not found at {new_data_dir}")
                                    show_prompt()
                                else:
                                    # Switch vendor
                                    update_vendor_state(new_vendor)
                                    print("-" * get_terminal_width())
                                    print("\n" + "=" * get_terminal_width())
                                    print(f"SWITCHED TO {vendor_state['vendor_name'].upper()}")
                                    print("=" * get_terminal_width() + "\n")
                                    result = print_stats()
                                    terminal_too_small[0] = (result is None)
                                    # Reset auto-refresh timer
                                    next_refresh_time = time.time() + monitor_state['interval']
                                    show_prompt()
                        else:
                            print("Usage: v, vendor [claude|codex|gemini|all]")
                            print(f"Current vendor: {vendor_state['vendor']}")
                            show_prompt()
                    elif command in ("v", "vendor"):
                        # Show current vendor if no argument provided
                        print(f"Current vendor: {vendor_state['vendor']}")
                        print("Usage: v, vendor [claude|codex|gemini|all]")
                        show_prompt()
                    elif command == "n":
                        # Rotate to next vendor: all -> claude -> codex -> gemini -> all
                        vendor_rotation = ['all', 'claude', 'codex', 'gemini']
                        current_idx = vendor_rotation.index(vendor_state['vendor'])
                        next_idx = (current_idx + 1) % len(vendor_rotation)
                        new_vendor = vendor_rotation[next_idx]

                        # Check if new vendor's data directory exists (skip for 'all')
                        if new_vendor == 'all':
                            new_data_dir = None
                        elif new_vendor == 'codex':
                            new_data_dir = get_codex_dir() / 'sessions'
                        elif new_vendor == 'gemini':
                            new_data_dir = get_gemini_dir() / 'tmp'
                        else:
                            new_data_dir = get_claude_dir() / 'projects'

                        if new_data_dir is not None and not new_data_dir.exists():
                            print(f"Error: Data directory not found at {new_data_dir}")
                            print(f"Skipping {new_vendor}, trying next...")
                            # Try next vendor in rotation
                            next_idx = (next_idx + 1) % len(vendor_rotation)
                            new_vendor = vendor_rotation[next_idx]
                            show_prompt()
                        else:
                            # Switch vendor
                            update_vendor_state(new_vendor)
                            print("-" * get_terminal_width())
                            print("\n" + "=" * get_terminal_width())
                            print(f"SWITCHED TO {vendor_state['vendor_name'].upper()}")
                            print("=" * get_terminal_width() + "\n")
                            result = print_stats()
                            terminal_too_small[0] = (result is None)
                            next_refresh_time = time.time() + monitor_state['interval']
                            show_prompt()
                    elif command == "a":
                        # Jump to vendor=all
                        if vendor_state['vendor'] == 'all':
                            print("Already monitoring all vendors.")
                            show_prompt()
                        else:
                            update_vendor_state('all')
                            print("-" * get_terminal_width())
                            print("\n" + "=" * get_terminal_width())
                            print(f"SWITCHED TO {vendor_state['vendor_name'].upper()}")
                            print("=" * get_terminal_width() + "\n")
                            result = print_stats()
                            terminal_too_small[0] = (result is None)
                            next_refresh_time = time.time() + monitor_state['interval']
                            show_prompt()
                    elif command.startswith("d ") or command.startswith("day ") or command.startswith("days "):
                        # Parse d, day, days command
                        parts = command.split()
                        if len(parts) == 2:
                            try:
                                new_days = int(parts[1])
                                if new_days < 1:
                                    print("Days must be at least 1.")
                                    show_prompt()
                                elif new_days == days_state['days']:
                                    print(f"Already showing {new_days} days.")
                                    show_prompt()
                                else:
                                    days_state['days'] = new_days
                                    print("-" * get_terminal_width())
                                    print("\n" + "=" * get_terminal_width())
                                    print(f"CHANGED TO {new_days} DAYS")
                                    print("=" * get_terminal_width() + "\n")
                                    result = print_stats()
                                    terminal_too_small[0] = (result is None)
                                    next_refresh_time = time.time() + monitor_state['interval']
                                    show_prompt()
                            except ValueError:
                                print(f"Invalid days value: '{parts[1]}'. Must be a positive integer.")
                                show_prompt()
                        else:
                            print("Usage: d <N>, day <N>, or days <N> (e.g., d 7)")
                            print(f"Current days: {days_state['days']}")
                            show_prompt()
                    elif command in ("d", "day", "days"):
                        # No argument provided: default to 1 day
                        new_days = 1
                        if new_days == days_state['days']:
                            print(f"Already showing {new_days} day.")
                            show_prompt()
                        else:
                            days_state['days'] = new_days
                            print("-" * get_terminal_width())
                            print("\n" + "=" * get_terminal_width())
                            print(f"CHANGED TO {new_days} DAY")
                            print("=" * get_terminal_width() + "\n")
                            result = print_stats()
                            terminal_too_small[0] = (result is None)
                            next_refresh_time = time.time() + monitor_state['interval']
                            show_prompt()
                    elif command in ("w", "week"):
                        # Week mode: 7 days
                        new_days = 7
                        if new_days == days_state['days']:
                            print(f"Already showing {new_days} days (week mode).")
                            show_prompt()
                        else:
                            days_state['days'] = new_days
                            print("-" * get_terminal_width())
                            print("\n" + "=" * get_terminal_width())
                            print(f"CHANGED TO {new_days} DAYS (WEEK MODE)")
                            print("=" * get_terminal_width() + "\n")
                            result = print_stats()
                            terminal_too_small[0] = (result is None)
                            next_refresh_time = time.time() + monitor_state['interval']
                            show_prompt()
                    elif command in ("m", "month"):
                        # Month mode: 30 days
                        new_days = 30
                        if new_days == days_state['days']:
                            print(f"Already showing {new_days} days (month mode).")
                            show_prompt()
                        else:
                            days_state['days'] = new_days
                            print("-" * get_terminal_width())
                            print("\n" + "=" * get_terminal_width())
                            print(f"CHANGED TO {new_days} DAYS (MONTH MODE)")
                            print("=" * get_terminal_width() + "\n")
                            result = print_stats()
                            terminal_too_small[0] = (result is None)
                            next_refresh_time = time.time() + monitor_state['interval']
                            show_prompt()
                    elif command.startswith("i ") or command.startswith("interval "):
                        # Parse i, interval command
                        parts = command.split()
                        if len(parts) == 2:
                            try:
                                new_interval = int(parts[1])
                                if new_interval < 1:
                                    print("Interval must be at least 1 second.")
                                    show_prompt()
                                elif new_interval == monitor_state['interval']:
                                    print(f"Already using {new_interval} second interval.")
                                    show_prompt()
                                else:
                                    monitor_state['interval'] = new_interval
                                    print(f"Refresh interval changed to {new_interval} seconds.")
                                    next_refresh_time = time.time() + monitor_state['interval']
                                    show_prompt()
                            except ValueError:
                                print(f"Invalid interval value: '{parts[1]}'. Must be a positive integer.")
                                show_prompt()
                        else:
                            print("Usage: i <N> or interval <N> (e.g., i 1800)")
                            print(f"Current interval: {monitor_state['interval']} seconds")
                            show_prompt()
                    elif command in ("i", "interval"):
                        # Show current interval if no argument provided
                        print(f"Current interval: {monitor_state['interval']} seconds")
                        print("Usage: i <N> or interval <N> (e.g., i 1800)")
                        show_prompt()
                    elif command in ("h", "help"):
                        # Print help info
                        print("-" * get_terminal_width())
                        print("Available Commands:")
                        print("  r, refresh       - Refresh statistics immediately")
                        print("  v, vendor [X]    - Switch vendor (claude|codex|gemini|all)")
                        print("  n                - Rotate to next vendor")
                        print("  a                - Jump to vendor=all")
                        print("  d, day, days [N] - Change days (default: 1 if no N)")
                        print("  w, week          - Week mode (7 days)")
                        print("  m, month         - Month mode (30 days)")
                        print("  i, interval <N>  - Change refresh interval (seconds)")
                        print("  h, help          - Show this help")
                        print("  e, exit          - Exit monitor mode")
                        print("  Ctrl+C, Ctrl+D   - Exit monitor mode")
                        print("-" * get_terminal_width())
                        print(f"Current: vendor={vendor_state['vendor']}, days={days_state['days']}, interval={monitor_state['interval']}s")
                        show_prompt()
                    elif command in ("e", "exit"):
                        print("-" * get_terminal_width())
                        print("\nExiting monitor mode...")
                        break
                    elif command == "":
                        # Empty command, just show prompt again
                        show_prompt()
                    elif command:
                        print(f"Unknown command: '{command}'. Type h for help.")
                        show_prompt()

        except KeyboardInterrupt:
            print("\n" + "-" * get_terminal_width())
            print("\nMonitoring stopped.")

        sys.exit(0)
    else:
        # One-time execution
        result = print_stats()
        if result is None:
            # Terminal too small - message already printed
            sys.exit(1)
        elif not result:
            # No data found
            sys.exit(0)


if __name__ == '__main__':
    main()
