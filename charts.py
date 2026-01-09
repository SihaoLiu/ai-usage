"""Chart and visualization functions for Claude Code / Codex usage analysis."""

from datetime import datetime, timedelta
from formatting import format_y_axis_value, format_total_value

# Model display configuration for Claude
MODEL_CONFIG = {
    'claude-opus-4-5-20251101': {'short': 'Opus 4.5', 'order': 0},
    'claude-opus-4-1-20250805': {'short': 'Opus 4.1', 'order': 1},
    'claude-sonnet-4-5-20250929': {'short': 'Sonnet 4.5', 'order': 2},
    'claude-sonnet-4-20250514': {'short': 'Sonnet 4', 'order': 3},
    'claude-haiku-4-5-20251001': {'short': 'Haiku 4.5', 'order': 4},
}

# Color configuration for lines (ANSI 256-color codes)
# Each model gets a distinct color, input/output differentiated by intensity
LINE_COLORS = {
    # Claude model colors
    'opus_input': '\033[38;5;196m',      # Bright Red
    'opus_output': '\033[38;5;203m',     # Light Red
    'sonnet_input': '\033[38;5;33m',     # Bright Blue
    'sonnet_output': '\033[38;5;75m',    # Light Blue
    'haiku_input': '\033[38;5;40m',      # Bright Green
    'haiku_output': '\033[38;5;120m',    # Light Green
    # Codex/OpenAI model colors (fallback colors by index)
    'model0_input': '\033[38;5;208m',    # Orange
    'model0_output': '\033[38;5;215m',   # Light Orange
    'model1_input': '\033[38;5;135m',    # Purple
    'model1_output': '\033[38;5;177m',   # Light Purple
    'model2_input': '\033[38;5;37m',     # Teal
    'model2_output': '\033[38;5;80m',    # Light Teal
    'model3_input': '\033[38;5;197m',    # Pink
    'model3_output': '\033[38;5;218m',   # Light Pink
    'model4_input': '\033[38;5;226m',    # Yellow
    'model4_output': '\033[38;5;228m',   # Light Yellow
    'model5_input': '\033[38;5;51m',     # Cyan
    'model5_output': '\033[38;5;87m',    # Light Cyan
}
RESET_COLOR = '\033[0m'


def get_short_model_name_for_chart(model):
    """Get short model name for chart display."""
    # Check if it's a known Claude model
    if model in MODEL_CONFIG:
        return MODEL_CONFIG[model]['short']

    # For Codex models with effort level like "gpt-5-codex (high)"
    if ' (' in model and model.endswith(')'):
        base_model, effort = model.rsplit(' (', 1)
        effort = effort.rstrip(')')
        # Abbreviate effort
        effort_short = {'low': 'L', 'medium': 'M', 'high': 'H', 'xhigh': 'XH'}.get(effort, effort[0].upper())
        short = f"{base_model}({effort_short})"
        return short

    # Generic fallback
    return model[:12] if len(model) > 12 else model


def print_multi_line_chart(time_series, height=29, days_back=7, chart_type='io', show_x_axis=True, target_width=None, interval_minutes=60, vendor='claude', included_models=None, show_legend=True):
    """Print a multi-line chart with multiple lines (models x token types).

    Args:
        time_series: Time series data with per-model token breakdown
        height: Height of the chart
        days_back: Number of days to show
        chart_type: 'io' (input+output) or 'cache' (cache_creation+cache_read, or reasoning for Codex)
        show_x_axis: Whether to show X-axis labels
        target_width: Target chart width in columns. If None, uses default x_scale=2.4
        interval_minutes: Interval in minutes for each data point (default: 60)
        vendor: 'claude' or 'codex' (affects chart labels for cache/reasoning)
        included_models: Set of model names to include (None = include all)
        show_legend: Whether to show the legend (default: True)
    """
    if not time_series:
        print("No time series data available.")
        return

    # Sort by time
    all_sorted_times = sorted(time_series.keys())

    if not all_sorted_times:
        print("No data available.")
        return

    # Calculate start time based on days_back parameter (from now, not from last data point)
    now = datetime.now().astimezone()
    start_time = now - timedelta(days=days_back)

    # Round start_time down to nearest interval
    # Calculate minutes since start of day and round down to interval boundary
    total_minutes = start_time.hour * 60 + start_time.minute
    interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
    start_time_rounded = start_time.replace(
        hour=interval_start_minutes // 60,
        minute=interval_start_minutes % 60,
        second=0,
        microsecond=0
    )

    # Create a complete continuous time series (every interval_minutes)
    sorted_times = []
    current_time = start_time_rounded
    while current_time <= now:
        sorted_times.append(current_time)
        current_time += timedelta(minutes=interval_minutes)

    if len(sorted_times) < 2:
        print("Not enough data points for chart.")
        return

    # Limit chart width to 500 columns
    if len(sorted_times) > 500:
        step = max(1, len(sorted_times) // 500)
        sorted_times = sorted_times[::step]

    # Reverse time axis: most recent on left, oldest on right
    sorted_times = sorted_times[::-1]

    # Determine which token types to show based on chart_type and vendor
    if chart_type == 'io':
        token_types = ['input', 'output']
        type_labels = {'input': 'Input', 'output': 'Output'}
        chart_title = "Models Input / Output Token Consumption"
    else:  # cache (or reasoning for Codex, or thinking for Gemini)
        token_types = ['cache_read', 'cache_creation']  # Order: cache_read first, then cache_creation/reasoning/thinking
        if vendor == 'codex':
            type_labels = {'cache_read': 'Cache Read In', 'cache_creation': 'Reasoning Out'}
            chart_title = "Models Cache Read Input / Reasoning Output Token Consumption"
        elif vendor == 'gemini':
            type_labels = {'cache_read': 'Cache Read In', 'cache_creation': 'Thinking Out'}
            chart_title = "Models Cache Read Input / Thinking Output Token Consumption"
        else:
            type_labels = {'cache_read': 'Cache Read In', 'cache_creation': 'Cache Create In'}
            chart_title = "Models Cache Read Input / Cache Creation Input Token Consumption"

    # Collect all models from the data
    all_models = set()
    for time in sorted_times:
        if time in time_series:
            all_models.update(time_series[time].keys())

    # Filter to only included models (if specified)
    if included_models is not None:
        all_models = all_models & included_models

    # Separate known Claude models from other models (Codex etc.)
    known_models = [m for m in all_models if m in MODEL_CONFIG]
    other_models = [m for m in all_models if m not in MODEL_CONFIG]

    # Sort known models by configured order, other models alphabetically
    known_models.sort(key=lambda m: MODEL_CONFIG[m]['order'])
    other_models.sort()

    # Combine: known models first, then other models
    all_models_sorted = known_models + other_models

    if not all_models_sorted:
        print("No models found in data.")
        return

    # Define line configurations: (model, token_type, color_key, line_char)
    lines = []
    line_chars = ['━', '─', '╌', '┄', '┈', '╍']  # Different line styles
    char_idx = 0
    for model_idx, model in enumerate(all_models_sorted):
        if model in MODEL_CONFIG:
            # Known Claude model - use configured name and colors
            model_short = MODEL_CONFIG[model]['short'].lower().split()[0]  # 'opus', 'sonnet', 'haiku'
            short_label = MODEL_CONFIG[model]['short']
            for token_type in token_types:
                # Map token_type to color category: input types use 'input' colors, output types use 'output' colors
                if token_type in ['input', 'cache_read']:
                    color_suffix = 'input'
                else:  # output, cache_creation (or reasoning for Codex)
                    color_suffix = 'output'
                color_key = f"{model_short}_{color_suffix}"
                lines.append({
                    'model': model,
                    'token_type': token_type,
                    'color': LINE_COLORS.get(color_key, ''),
                    'char': line_chars[char_idx % len(line_chars)],
                    'label': f"{short_label} {type_labels[token_type]}"
                })
                char_idx += 1
        else:
            # Unknown/Codex model - use dynamic name and indexed colors
            short_label = get_short_model_name_for_chart(model)
            color_idx = model_idx % 6  # Cycle through 6 color sets
            for token_type in token_types:
                # Map token_type to color category: input types use 'input' colors, output types use 'output' colors
                if token_type in ['input', 'cache_read']:
                    color_suffix = 'input'
                else:  # output, cache_creation (or reasoning for Codex)
                    color_suffix = 'output'
                color_key = f"model{color_idx}_{color_suffix}"
                lines.append({
                    'model': model,
                    'token_type': token_type,
                    'color': LINE_COLORS.get(color_key, ''),
                    'char': line_chars[char_idx % len(line_chars)],
                    'label': f"{short_label} {type_labels[token_type]}"
                })
                char_idx += 1

    # Calculate values for each line at each time point
    line_values = {i: [] for i in range(len(lines))}
    for time in sorted_times:
        for i, line in enumerate(lines):
            if time in time_series and line['model'] in time_series[time]:
                value = time_series[time][line['model']].get(line['token_type'], 0)
            else:
                value = 0
            line_values[i].append(value)

    # Find max value across all lines for Y-axis scaling
    all_values = []
    for values in line_values.values():
        all_values.extend(values)

    max_value_raw = max(all_values) if all_values else 1
    if max_value_raw == 0:
        max_value_raw = 1

    # Round max to nice value
    def round_to_nice(value, round_up=True):
        if value >= 5_000_000_000:
            unit = 5_000_000_000
        elif value >= 5_000_000:
            unit = 5_000_000
        elif value >= 5_000:
            unit = 5_000
        else:
            unit = 5
        if round_up:
            return ((int(value) + unit - 1) // unit) * unit
        return (int(value) // unit) * unit

    min_value = 0
    max_value = round_to_nice(max_value_raw, round_up=True)
    if max_value == min_value:
        max_value = min_value + 5_000

    chart_height = height
    num_data_points = len(sorted_times)

    # Scale values to chart height (row positions)
    def value_to_row(value):
        if max_value == min_value:
            return 0
        return int((value - min_value) / (max_value - min_value) * (chart_height - 1))

    line_rows = {i: [value_to_row(v) for v in values] for i, values in line_values.items()}

    # Build chart columns (data points and separators)
    # Count separators first (midnight boundaries)
    separator_count = sum(1 for i, t in enumerate(sorted_times) if t.hour == 0 and t.minute == 0 and i > 0)

    # Calculate x_scale based on target_width or use default
    y_axis_width = 7  # Width of Y-axis labels (e.g., "100M |")
    if target_width is not None:
        # Target width is total terminal width, chart content = target_width - y_axis_width
        available_width = target_width - y_axis_width - separator_count
        x_scale = max(1.0, available_width / num_data_points)
    else:
        # Default: each data point occupies ~2.4 columns
        x_scale = 2.4

    # Distribute extra columns evenly using accumulated fractions
    chart_columns = []
    data_to_col = {}  # Maps data_idx to first column
    data_col_count = {}  # Maps data_idx to number of columns it occupies
    col_idx = 0
    accumulated = 0.0

    for i in range(num_data_points):
        time = sorted_times[i]
        if time.hour == 0 and time.minute == 0 and i > 0:
            chart_columns.append(('separator', None))
            col_idx += 1

        # Calculate columns for this data point
        accumulated += x_scale
        cols_for_this = int(accumulated)
        accumulated -= cols_for_this

        # Ensure at least 1 column per data point
        if cols_for_this < 1:
            cols_for_this = 1

        data_to_col[i] = col_idx
        data_col_count[i] = cols_for_this

        for sub_col in range(cols_for_this):
            chart_columns.append(('data', i, sub_col, cols_for_this))
            col_idx += 1

    chart_width = len(chart_columns)

    # Print centered title above the chart
    total_width = chart_width + y_axis_width  # Y-axis label width (7 chars: "xxxxx |")
    # Only print blank line before title for the first chart (show_x_axis=False)
    # For the second chart (show_x_axis=True), legend comes directly before the title
    if not show_x_axis:
        print()
    print(chart_title.center(total_width))
    print("=" * total_width)

    # Calculate daily totals for display at top
    # Group by calendar date (not by separator) to handle reversed time axis
    from collections import OrderedDict
    daily_data = OrderedDict()  # date -> {'total': 0, 'cols': [], 'time': datetime}

    for col_idx, col_info in enumerate(chart_columns):
        if col_info[0] == 'separator':
            continue
        col_type, data_idx, sub_col, total_cols = col_info
        time = sorted_times[data_idx]
        date_key = time.date()

        if date_key not in daily_data:
            daily_data[date_key] = {'total': 0, 'cols': [], 'time': time}

        daily_data[date_key]['cols'].append(col_idx)

        # Only count on first column of each data point to avoid double counting
        if sub_col == 0:
            for i in range(len(lines)):
                daily_data[date_key]['total'] += line_values[i][data_idx]

    # Build daily_totals list with mid column position
    daily_totals = []
    for date_key, data in daily_data.items():
        if data['cols']:
            mid_col = (min(data['cols']) + max(data['cols'])) // 2
            daily_totals.append((mid_col, data['total'], data['time']))

    # Print daily totals header
    weekday_line = " " * 7
    date_line = " " * 7
    prev_end = 0
    weekday_abbr = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for day_idx, (mid_col, total, day_start) in enumerate(daily_totals):
        total_str = format_total_value(total)
        weekday = weekday_abbr[day_start.weekday()]
        weekday_total = f"{weekday} : {total_str}"
        date_str = day_start.strftime(' %m / %d')

        colon_idx = weekday_total.index(':')
        slash_idx = date_str.index('/')

        if colon_idx > slash_idx:
            date_str = ' ' * (colon_idx - slash_idx) + date_str
            slash_idx = colon_idx
        elif slash_idx > colon_idx:
            weekday_total = ' ' * (slash_idx - colon_idx) + weekday_total
            colon_idx = slash_idx

        max_len = max(len(weekday_total), len(date_str))
        weekday_total = weekday_total.ljust(max_len)
        date_str = date_str.ljust(max_len)

        start_pos = mid_col - colon_idx
        padding = start_pos - prev_end
        if padding > 0:
            weekday_line += " " * padding
            date_line += " " * padding

        weekday_line += weekday_total
        date_line += date_str
        prev_end = start_pos + max_len

    print(weekday_line)
    print(date_line)

    # Create a 2D grid for the chart
    # grid[row][col] = (line_idx, char_type) - only store one entry per cell
    grid = [[None for _ in range(len(chart_columns))] for _ in range(chart_height)]

    # For each line, determine what character to draw at each position
    # Process lines in reverse order so higher priority lines (earlier in list) overwrite
    for line_idx in range(len(lines) - 1, -1, -1):
        rows = line_rows[line_idx]

        for col_idx, col_info in enumerate(chart_columns):
            if col_info[0] == 'separator':
                continue

            col_type, data_idx, sub_col, total_cols = col_info
            curr_row = rows[data_idx]
            prev_row = rows[data_idx - 1] if data_idx > 0 else curr_row

            if sub_col > 0:
                # Non-first columns: horizontal line at current value
                grid[curr_row][col_idx] = (line_idx, 'flat')
            else:
                # First column (sub_col == 0): handle transition from previous value
                if prev_row == curr_row:
                    # No change - horizontal line
                    grid[curr_row][col_idx] = (line_idx, 'flat')
                elif prev_row < curr_row:
                    # Going up: draw ╯ at prev level, │ in between, ╭ at curr level
                    grid[prev_row][col_idx] = (line_idx, 'up_to_right')
                    for row in range(prev_row + 1, curr_row):
                        grid[row][col_idx] = (line_idx, 'vertical')
                    grid[curr_row][col_idx] = (line_idx, 'up_from_left')
                else:
                    # Going down: draw ╮ at prev level, │ in between, ╰ at curr level
                    grid[prev_row][col_idx] = (line_idx, 'down_to_right')
                    for row in range(curr_row + 1, prev_row):
                        grid[row][col_idx] = (line_idx, 'vertical')
                    grid[curr_row][col_idx] = (line_idx, 'down_from_left')

    # Character mapping for line drawing
    char_map = {
        'flat': '─',
        'peak': '╭',  # Will draw ╮ at the same position for visual effect
        'valley': '╰',
        'up_from_left': '╭',
        'down_from_left': '╰',
        'down_to_right': '╮',
        'up_to_right': '╯',
        'vertical': '│',
    }

    # Draw chart from top to bottom
    for row in range(chart_height - 1, -1, -1):
        # Y-axis label
        y_val = min_value + (max_value - min_value) * row / (chart_height - 1)
        y_label = f"{format_y_axis_value(y_val)} |"

        line = ""
        for col_idx, col_info in enumerate(chart_columns):
            if col_info[0] == 'separator':
                line += "|"
            else:
                cell = grid[row][col_idx]
                if cell:
                    line_idx, char_type = cell
                    char = char_map.get(char_type, '─')
                    color = lines[line_idx]['color']
                    line += f"{color}{char}{RESET_COLOR}"
                else:
                    line += " "

        print(y_label + line)

    # X-axis
    x_axis_line = ""
    for col_info in chart_columns:
        if col_info[0] == 'separator':
            x_axis_line += "\u2534"
        else:
            x_axis_line += "\u2500"
    print("      \u2514" + x_axis_line)

    # X-axis labels
    if show_x_axis:
        print()
        labels = []
        positions = []

        # Calculate time span in minutes
        first_time = sorted_times[-1]  # Oldest time (remember: reversed)
        # Use current time as the most recent point (leftmost on chart)
        last_time = datetime.now().astimezone()
        time_span_minutes = (last_time - first_time).total_seconds() / 60

        # Calculate how much time each tick should represent (~5% of total time span)
        target_tick_interval_minutes = time_span_minutes * 0.05

        # Round to a standard interval that divides 24h evenly for day consistency
        # Available intervals: 15min, 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 24h
        standard_intervals = [15, 30, 60, 120, 180, 240, 360, 480, 720, 1440]

        # Find the closest standard interval
        tick_interval_minutes = min(standard_intervals,
                                   key=lambda x: abs(x - target_tick_interval_minutes))

        # Generate ticks at consistent positions across all days
        # Start from midnight of the first day
        current_tick = first_time.replace(hour=0, minute=0, second=0, microsecond=0)

        # Advance to first tick boundary at or after first_time
        minutes_since_midnight = first_time.hour * 60 + first_time.minute
        ticks_since_midnight = (minutes_since_midnight + tick_interval_minutes - 1) // tick_interval_minutes
        current_tick += timedelta(minutes=ticks_since_midnight * tick_interval_minutes)

        # Track used positions to avoid placing multiple labels at the same position
        used_positions = set()

        # Generate ticks
        while current_tick <= last_time:
            # Find closest data point to this tick
            closest_idx = None
            min_diff = None

            for i, time in enumerate(sorted_times):
                diff = abs((time - current_tick).total_seconds())
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    closest_idx = i

            # Only add if we found a close match and this position hasn't been used
            if closest_idx is not None and closest_idx in data_to_col:
                position = data_to_col[closest_idx]
                if position not in used_positions:
                    # Format label based on interval
                    if tick_interval_minutes < 60:
                        label = current_tick.strftime('%H:%M')
                    else:
                        label = current_tick.strftime('%H')
                    labels.append(label)
                    positions.append(position)
                    used_positions.add(position)

            current_tick += timedelta(minutes=tick_interval_minutes)

        max_label_len = max(len(label) for label in labels) if labels else 0

        for char_idx in range(max_label_len):
            line = "       "
            for col_idx, col_info in enumerate(chart_columns):
                if col_info[0] == 'separator':
                    char_to_print = "|"
                else:
                    char_to_print = " "
                    for label_idx, pos in enumerate(positions):
                        if col_idx == pos and char_idx < len(labels[label_idx]):
                            char_to_print = labels[label_idx][char_idx]
                            break
                line += char_to_print
            print(line)

    # Legend (shown for both charts when show_legend=True)
    if show_legend:
        # Print legend with colored symbols
        legend_parts = []
        for line in lines:
            legend_parts.append(f"{line['color']}─{RESET_COLOR} {line['label']}")
        print("Legend: " + "  ".join(legend_parts))


def print_stacked_bar_chart(time_series, height=75, days_back=7, chart_type='all', show_x_axis=True):
    """Print a text-based stacked bar chart of token usage breakdown over time.

    Args:
        time_series: Time series data with token breakdown
        height: Height of the chart
        days_back: Number of days to show
        chart_type: 'all' (all 4 types), 'io' (input+output), or 'cache' (cache_creation+cache_read)
        show_x_axis: Whether to show X-axis labels
    """
    if not time_series:
        print("No time series data available.")
        return

    # Sort by time
    all_sorted_times = sorted(time_series.keys())

    if not all_sorted_times:
        print("No data available.")
        return

    # Calculate start time based on days_back parameter
    last_time = all_sorted_times[-1]
    start_time = last_time - timedelta(days=days_back)

    # Round start_time down to nearest hour
    start_time_rounded = start_time.replace(minute=0, second=0, microsecond=0)

    # Create a complete continuous time series (every hour)
    # This ensures uniform spacing even when there's no data
    sorted_times = []
    current_time = start_time_rounded
    while current_time <= last_time:
        sorted_times.append(current_time)
        current_time += timedelta(hours=1)

    if len(sorted_times) < 2:
        print("Not enough data points for chart.")
        return

    # Limit chart width to 500 columns
    if len(sorted_times) > 500:
        # Adjust interval to fit in 500 columns
        hours_per_interval = len(sorted_times) / 500
        print(f"Note: Adjusting interval to ~{hours_per_interval:.1f} hours to fit in 500 columns.")

        # Resample to fit in 500 columns
        step = max(1, len(sorted_times) // 500)
        sorted_times = sorted_times[::step]

    # Calculate breakdown per time interval
    breakdown_data = []
    totals = []
    for time in sorted_times:
        if time in time_series:
            input_val = time_series[time].get('input', 0)
            output_val = time_series[time].get('output', 0)
            cache_creation_val = time_series[time].get('cache_creation', 0)
            cache_read_val = time_series[time].get('cache_read', 0)
        else:
            input_val = output_val = cache_creation_val = cache_read_val = 0

        breakdown_data.append({
            'input': input_val,
            'output': output_val,
            'cache_creation': cache_creation_val,
            'cache_read': cache_read_val
        })

        # Calculate total based on chart_type
        if chart_type == 'io':
            total = input_val + output_val
        elif chart_type == 'cache':
            total = cache_creation_val + cache_read_val
        else:  # 'all'
            total = input_val + output_val + cache_creation_val + cache_read_val

        totals.append(total)

    # First pass: calculate Y-axis range from all data
    max_value_raw = max(totals) if totals else 1
    min_value_raw = min(totals) if totals else 0

    # Round min/max to nearest multiple of 5K or 5M or 5B
    def round_to_5_multiple(value, round_up=True):
        """Round value to nearest multiple of 5B/5M/5K."""
        if value >= 5_000_000_000:
            # Round to nearest 5B
            unit = 5_000_000_000
        elif value >= 5_000_000:
            # Round to nearest 5M
            unit = 5_000_000
        elif value >= 5_000:
            # Round to nearest 5K
            unit = 5_000
        else:
            # Round to nearest 5
            unit = 5

        if round_up:
            return ((int(value) + unit - 1) // unit) * unit
        else:
            return (int(value) // unit) * unit

    min_value = round_to_5_multiple(min_value_raw, round_up=False)
    max_value = round_to_5_multiple(max_value_raw, round_up=True)

    # Ensure max > min
    if max_value == min_value:
        max_value = min_value + 5_000

    num_data_points = len(totals)
    chart_height = height

    # Print chart title based on type
    if chart_type == 'io':
        print("\nInput + Output Tokens Over Time (1-hour intervals, Local Time)")
        print(f"Y-axis: Input and Output token consumption")
    elif chart_type == 'cache':
        print("\nCache Tokens Over Time (1-hour intervals, Local Time)")
        print(f"Y-axis: Cache Output and Cache Input token consumption")
    else:
        print("\nToken Usage Breakdown Over Time (1-hour intervals, Local Time)")
        print(f"Y-axis: Token consumption (all token types)")

    if show_x_axis:
        print(f"X-axis: Time (each day has 24 data points, ticks at 6-hour intervals)\n")
    else:
        print()

    # Scale breakdown values to chart height
    # For each data point, calculate the scaled heights of each segment
    scaled_breakdown = []
    for breakdown in breakdown_data:
        if max_value == min_value:
            scaled_breakdown.append({
                'input': 0,
                'output': 0,
                'cache_creation': 0,
                'cache_read': 0
            })
        else:
            # Scale each component individually
            scaled_breakdown.append({
                'input': int((breakdown['input'] - 0) / (max_value - min_value) * (chart_height - 1)),
                'output': int((breakdown['output'] - 0) / (max_value - min_value) * (chart_height - 1)),
                'cache_creation': int((breakdown['cache_creation'] - 0) / (max_value - min_value) * (chart_height - 1)),
                'cache_read': int((breakdown['cache_read'] - 0) / (max_value - min_value) * (chart_height - 1))
            })

    # Build chart:
    # First day: data points (no separator, Y-axis serves as the boundary)
    # Subsequent days: separator + data points
    chart_columns = []  # List of (type, value)
    data_to_col = {}  # Map data point index to column index

    col_idx = 0
    for i in range(num_data_points):
        time = sorted_times[i]

        # Add separator before 00:00 (except for the very first day)
        if time.hour == 0 and time.minute == 0 and i > 0:
            chart_columns.append(('separator', None))
            col_idx += 1

        # Add data point
        chart_columns.append(('data', i))
        data_to_col[i] = col_idx
        col_idx += 1

    chart_width = len(chart_columns)
    y_axis_width = 7  # "xxxxx |" format
    print("=" * (chart_width + y_axis_width))

    # Calculate daily totals for display at top of chart
    daily_totals = []
    current_day_start = None
    current_day_total = 0
    current_day_start_col = 0

    for col_idx, (col_type, col_data) in enumerate(chart_columns):
        if col_type == 'separator':
            # End of previous day
            if current_day_start is not None:
                mid_col = (current_day_start_col + col_idx) // 2
                daily_totals.append((mid_col, current_day_total, current_day_start))
            current_day_start = None
            current_day_total = 0
            current_day_start_col = col_idx + 1
        else:
            data_idx = col_data
            if current_day_start is None:
                current_day_start = sorted_times[data_idx]
                current_day_start_col = col_idx
            current_day_total += totals[data_idx]

    # Add last day if exists
    if current_day_start is not None:
        mid_col = (current_day_start_col + len(chart_columns)) // 2
        daily_totals.append((mid_col, current_day_total, current_day_start))

    # Print daily totals at top of chart (weekday + total tokens)
    weekday_line = " " * 7  # Align with Y-axis
    date_line = " " * 7  # Align with Y-axis
    prev_end = 0

    weekday_abbr = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for day_idx, (mid_col, total, day_start) in enumerate(daily_totals):
        total_str = format_total_value(total)
        weekday = weekday_abbr[day_start.weekday()]
        weekday_total = f"{weekday} : {total_str}"

        # Format date as " MM / DD"
        date_str = day_start.strftime(' %m / %d')

        # Find positions of : and /
        colon_idx = weekday_total.index(':')
        slash_idx = date_str.index('/')

        # Add padding to align : and / at the same relative position
        if colon_idx > slash_idx:
            # Add spaces before date_str
            date_str = ' ' * (colon_idx - slash_idx) + date_str
            slash_idx = colon_idx
        elif slash_idx > colon_idx:
            # Add spaces before weekday_total
            weekday_total = ' ' * (slash_idx - colon_idx) + weekday_total
            colon_idx = slash_idx

        # Make both strings the same length
        max_len = max(len(weekday_total), len(date_str))
        weekday_total = weekday_total.ljust(max_len)
        date_str = date_str.ljust(max_len)

        # Position them so : and / are at mid_col
        start_pos = mid_col - colon_idx

        # Add padding and content to both lines
        padding = start_pos - prev_end
        if padding > 0:
            weekday_line += " " * padding
            date_line += " " * padding

        weekday_line += weekday_total
        date_line += date_str
        prev_end = start_pos + max_len

    print(weekday_line)
    print(date_line)

    # Draw chart from top to bottom (stacked bar chart style)
    for row in range(chart_height - 1, -1, -1):
        # Y-axis label
        y_val = min_value + (max_value - min_value) * row / (chart_height - 1)
        y_label = f"{format_y_axis_value(y_val)} |"

        # Chart line
        line = ""
        for col_type, col_data in chart_columns:
            if col_type == 'separator':
                line += "|"
            else:
                data_idx = col_data
                breakdown = scaled_breakdown[data_idx]

                # Calculate cumulative heights for stacking (bottom to top)
                # Stack order: input (bottom) -> output -> cache_creation -> cache_read (top)
                input_height = breakdown['input']
                output_height = breakdown['output']
                cache_creation_height = breakdown['cache_creation']
                cache_read_height = breakdown['cache_read']

                cumulative_input = input_height
                cumulative_output = cumulative_input + output_height
                cumulative_cache_creation = cumulative_output + cache_creation_height
                cumulative_cache_read = cumulative_cache_creation + cache_read_height

                # Determine which character to draw based on current row and chart_type
                # ANSI 256-color codes: Cyan for input, Green for output, Orange for cache_output, Pink for cache_input
                if chart_type == 'io':
                    # Only show input and output
                    if row < cumulative_input:
                        line += "\033[38;5;51m\u2588\033[0m"  # Input tokens (Bright Cyan)
                    elif row < cumulative_output:
                        line += "\033[38;5;46m\u2593\033[0m"  # Output tokens (Bright Green)
                    else:
                        line += " "  # Empty space
                elif chart_type == 'cache':
                    # Only show cache_creation and cache_read, but calculate from 0
                    cache_only_cumulative_creation = cache_creation_height
                    cache_only_cumulative_read = cache_only_cumulative_creation + cache_read_height
                    if row < cache_only_cumulative_creation:
                        line += "\033[38;5;214m\u2592\033[0m"  # Cache output tokens (Bright Orange)
                    elif row < cache_only_cumulative_read:
                        line += "\u2588"  # Cache input tokens (default color)
                    else:
                        line += " "  # Empty space
                else:
                    # Show all 4 types
                    if row < cumulative_input:
                        line += "\033[38;5;51m\u2588\033[0m"  # Input tokens (Bright Cyan)
                    elif row < cumulative_output:
                        line += "\033[38;5;46m\u2593\033[0m"  # Output tokens (Bright Green)
                    elif row < cumulative_cache_creation:
                        line += "\033[38;5;214m\u2592\033[0m"  # Cache output tokens (Bright Orange)
                    elif row < cumulative_cache_read:
                        line += "\u2588\u2591"  # Cache input tokens (default color)
                    else:
                        line += " "  # Empty space

        print(y_label + line)

    # X-axis with day separators
    # Position: 6 spaces to align + with Y-axis |
    x_axis_line = ""
    for col_type, _ in chart_columns:
        if col_type == 'separator':
            x_axis_line += "\u2534"
        else:
            x_axis_line += "\u2500"
    print("      \u2514" + x_axis_line)  # 6 spaces + corner aligns with Y-axis position

    # X-axis labels (show only if show_x_axis is True)
    if show_x_axis:
        # X-axis labels (show only 6:00, 12:00, and 18:00) - rotated 90 degrees counter-clockwise
        print()

        # Create label for 6:00, 12:00, and 18:00
        labels = []
        positions = []

        for i, time in enumerate(sorted_times):
            # Only show labels for 6:00, 12:00, and 18:00
            if time.hour in [6, 12, 18]:
                # Position is the column index for this data point
                if i in data_to_col:
                    labels.append(time.strftime('%H'))
                    positions.append(data_to_col[i])

        # Find maximum label length
        max_label_len = max(len(label) for label in labels) if labels else 0

        # Print each character position vertically
        # Position: 6 spaces to align first character with Y-axis | position
        # Then add one more space so labels start at column 0 of chart content
        for char_idx in range(max_label_len):
            line = "       "  # 7 spaces: aligns with Y-axis format (5 chars + space + |)

            for col_idx, (col_type, col_data) in enumerate(chart_columns):
                if col_type == 'separator':
                    char_to_print = "|"
                else:
                    # Check if this column has a label
                    char_to_print = " "
                    for label_idx, pos in enumerate(positions):
                        if col_idx == pos and char_idx < len(labels[label_idx]):
                            char_to_print = labels[label_idx][char_idx]
                            break

                line += char_to_print

            print(line)

    # Show summary info only for the last chart (when show_x_axis is True)
    if show_x_axis:
        print("\n" + "=" * (chart_width + y_axis_width))
        print(f"Total time span: {sorted_times[0].strftime('%Y-%m-%d %H:%M')} to {sorted_times[-1].strftime('%Y-%m-%d %H:%M')} | Data points: {len(sorted_times)}")
        print(f"Legend: \033[38;5;51m\u2588\033[0m Input  \033[38;5;46m\u2593\033[0m Output  \u2588 Cache Input  \033[38;5;214m\u2592\033[0m Cache Output")


def print_model_chart(time_series, width=100, height=15):
    """Print a text-based chart showing each model's usage over time."""
    if not time_series:
        print("No time series data available.")
        return

    sorted_times = sorted(time_series.keys())

    if len(sorted_times) < 2:
        print("Not enough data points for chart.")
        return

    # Get all models and their colors
    all_models = set()
    for models in time_series.values():
        all_models.update(models.keys())

    all_models = sorted(all_models)
    model_symbols = {'claude-sonnet-4-5-20250929': '\u2588',
                     'claude-haiku-4-5-20251001': '\u2593',
                     'claude-opus-4-1-20250805': '\u2592'}

    print("\n\nToken Usage by Model Over Time")
    print("=" * width)

    for model in all_models:
        if model not in model_symbols:
            model_symbols[model] = '\u2591'

        # Get values for this model
        values = []
        for time in sorted_times:
            val = time_series[time].get(model, 0) / 1000  # KTok
            values.append(val)

        if all(v == 0 for v in values):
            continue

        max_value = max(values)

        # Print model name
        print(f"\n{model}:")
        print(f"Max: {max_value:.1f} KTok")

        # Simple bar chart
        chart_width = width - 25
        for i, val in enumerate(values):
            if i % 4 == 0:  # Show every 4th data point to avoid clutter
                bar_length = int((val / max_value * chart_width)) if max_value > 0 else 0
                time_str = sorted_times[i].strftime('%m/%d %H:%M')
                bar = model_symbols[model] * bar_length
                print(f"  {time_str} |{bar} {val:.1f}")


def print_vendor_comparison_chart(time_series, height=20, days_back=7, target_width=None, interval_minutes=60, show_legend=True):
    """Print a chart comparing total token usage across vendors.

    Uses the same style as print_multi_line_chart with vertical bars, x-axis ticks, etc.

    Args:
        time_series: Time series data with per-vendor totals:
            {interval_time: {'Claude': total, 'Codex': total, 'Gemini': total}, ...}
        height: Height of the chart
        days_back: Number of days to show
        target_width: Target chart width in columns
        interval_minutes: Interval in minutes for each data point
        show_legend: Whether to show the legend
    """
    from collections import OrderedDict

    if not time_series:
        print("No time series data available.")
        return

    # Sort by time
    all_sorted_times = sorted(time_series.keys())
    if not all_sorted_times:
        print("No time series data available.")
        return

    # Calculate start time based on days_back parameter (from now, not from last data point)
    now = datetime.now().astimezone()
    start_time = now - timedelta(days=days_back)

    # Round start_time down to nearest interval
    total_minutes = start_time.hour * 60 + start_time.minute
    interval_start_minutes = (total_minutes // interval_minutes) * interval_minutes
    start_time_rounded = start_time.replace(
        hour=interval_start_minutes // 60,
        minute=interval_start_minutes % 60,
        second=0,
        microsecond=0
    )

    # Create a complete continuous time series (every interval_minutes)
    sorted_times = []
    current_time = start_time_rounded
    while current_time <= now:
        sorted_times.append(current_time)
        current_time += timedelta(minutes=interval_minutes)

    if len(sorted_times) < 2:
        print("Not enough data points for chart.")
        return

    # Limit chart width to 500 columns
    if len(sorted_times) > 500:
        step = max(1, len(sorted_times) // 500)
        sorted_times = sorted_times[::step]

    # Reverse time axis: most recent on left, oldest on right
    sorted_times = sorted_times[::-1]

    # Collect all vendors from the data
    all_vendors = set()
    for time in sorted_times:
        if time in time_series:
            all_vendors.update(time_series[time].keys())

    # Sort vendors in a consistent order
    vendor_order = ['Claude', 'Codex', 'Gemini']
    all_vendors_sorted = [v for v in vendor_order if v in all_vendors]
    # Add any unexpected vendors
    all_vendors_sorted += [v for v in sorted(all_vendors) if v not in vendor_order]

    if not all_vendors_sorted:
        print("No vendor data available.")
        return

    # Vendor colors (distinct colors for each vendor)
    VENDOR_COLORS = {
        'Claude': '\033[38;5;173m',   # Brownish/Salmon
        'Codex': '\033[38;5;255m',    # White
        'Gemini': '\033[38;5;33m',    # Blue
        'All': '\033[38;5;226m',      # Yellow (for combined total)
    }
    DEFAULT_COLORS = ['\033[38;5;135m', '\033[38;5;197m', '\033[38;5;51m']  # Purple, Pink, Cyan

    chart_title = "Total Token Consumption by Vendor"
    chart_height = height
    num_data_points = len(sorted_times)

    # Count separators first (midnight boundaries)
    separator_count = sum(1 for i, t in enumerate(sorted_times) if t.hour == 0 and t.minute == 0 and i > 0)

    # Calculate x_scale based on target_width or use default
    y_axis_width = 7
    if target_width is not None:
        available_width = target_width - y_axis_width - separator_count
        x_scale = max(1.0, available_width / num_data_points)
    else:
        x_scale = 2.4

    # Build chart columns (data points and separators)
    chart_columns = []
    data_to_col = {}
    data_col_count = {}
    col_idx = 0
    accumulated = 0.0

    for i in range(num_data_points):
        time = sorted_times[i]
        if time.hour == 0 and time.minute == 0 and i > 0:
            chart_columns.append(('separator', None))
            col_idx += 1

        accumulated += x_scale
        cols_for_this = int(accumulated)
        accumulated -= cols_for_this
        if cols_for_this < 1:
            cols_for_this = 1

        data_to_col[i] = col_idx
        data_col_count[i] = cols_for_this

        for sub_col in range(cols_for_this):
            chart_columns.append(('data', i, sub_col, cols_for_this))
            col_idx += 1

    chart_width = len(chart_columns)

    # Add "All" to the list (at the end, so it's drawn first and appears behind other lines)
    all_vendors_sorted.append('All')

    # Build data points for each vendor
    vendor_data = {}
    for vendor in all_vendors_sorted:
        if vendor == 'All':
            continue  # Calculate All separately below
        vendor_data[vendor] = []
        for time in sorted_times:
            value = time_series.get(time, {}).get(vendor, 0)
            vendor_data[vendor].append(value)

    # Calculate "All" as sum of all vendors
    vendor_data['All'] = []
    for i, time in enumerate(sorted_times):
        total = sum(vendor_data[v][i] for v in all_vendors_sorted if v != 'All')
        vendor_data['All'].append(total)

    # Calculate maximum value across all vendors
    all_values = []
    for vendor in all_vendors_sorted:
        all_values.extend(vendor_data[vendor])

    max_value_raw = max(all_values) if all_values else 1
    if max_value_raw == 0:
        max_value_raw = 1

    # Round max to nice value
    def round_to_nice(value, round_up=True):
        if value >= 5_000_000_000:
            unit = 5_000_000_000
        elif value >= 5_000_000:
            unit = 5_000_000
        elif value >= 5_000:
            unit = 5_000
        else:
            unit = 5
        if round_up:
            return ((int(value) + unit - 1) // unit) * unit
        return (int(value) // unit) * unit

    min_value = 0
    max_value = round_to_nice(max_value_raw, round_up=True)
    if max_value == min_value:
        max_value = min_value + 5_000

    # Scale values to chart height (row positions)
    def value_to_row(value):
        if max_value == min_value:
            return 0
        return int((value - min_value) / (max_value - min_value) * (chart_height - 1))

    vendor_rows = {}
    for vendor in all_vendors_sorted:
        vendor_rows[vendor] = [value_to_row(v) for v in vendor_data[vendor]]

    # Print centered title above the chart
    total_width = chart_width + y_axis_width  # y_axis_width = 7 (defined earlier)
    print()
    print(chart_title.center(total_width))
    print("=" * total_width)

    # Calculate daily totals for display at top
    daily_data = OrderedDict()

    for col_idx_iter, col_info in enumerate(chart_columns):
        if col_info[0] == 'separator':
            continue
        col_type, data_idx, sub_col, total_cols = col_info
        time = sorted_times[data_idx]
        date_key = time.date()

        if date_key not in daily_data:
            daily_data[date_key] = {'total': 0, 'cols': [], 'time': time}

        daily_data[date_key]['cols'].append(col_idx_iter)

        # Only count on first column of each data point to avoid double counting
        # Use "All" vendor directly since it's already the sum
        if sub_col == 0:
            daily_data[date_key]['total'] += vendor_data['All'][data_idx]

    # Build daily_totals list with mid column position
    daily_totals = []
    for date_key, data in daily_data.items():
        if data['cols']:
            mid_col = (min(data['cols']) + max(data['cols'])) // 2
            daily_totals.append((mid_col, data['total'], data['time']))

    # Print daily totals header
    weekday_line = " " * 7
    date_line = " " * 7
    prev_end = 0
    weekday_abbr = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for day_idx, (mid_col, total, day_start) in enumerate(daily_totals):
        total_str = format_total_value(total)
        weekday = weekday_abbr[day_start.weekday()]
        weekday_total = f"{weekday} : {total_str}"
        date_str = day_start.strftime(' %m / %d')

        colon_idx = weekday_total.index(':')
        slash_idx = date_str.index('/')

        if colon_idx > slash_idx:
            date_str = ' ' * (colon_idx - slash_idx) + date_str
            slash_idx = colon_idx
        elif slash_idx > colon_idx:
            weekday_total = ' ' * (slash_idx - colon_idx) + weekday_total
            colon_idx = slash_idx

        max_len = max(len(weekday_total), len(date_str))
        weekday_total = weekday_total.ljust(max_len)
        date_str = date_str.ljust(max_len)

        start_pos = mid_col - colon_idx
        padding = start_pos - prev_end
        if padding > 0:
            weekday_line += " " * padding
            date_line += " " * padding

        weekday_line += weekday_total
        date_line += date_str
        prev_end = start_pos + max_len

    print(weekday_line)
    print(date_line)

    # Create a 2D grid for the chart
    grid = [[None for _ in range(len(chart_columns))] for _ in range(chart_height)]

    # Character mapping for line drawing (regular and bold for "All" line)
    char_map = {
        'flat': '─',
        'up_from_left': '╭',
        'down_from_left': '╰',
        'down_to_right': '╮',
        'up_to_right': '╯',
        'vertical': '│',
    }
    # Bold/heavy characters for the "All" line
    char_map_bold = {
        'flat': '━',
        'up_from_left': '┏',
        'down_from_left': '┗',
        'down_to_right': '┓',
        'up_to_right': '┛',
        'vertical': '┃',
    }

    # Process vendors in reverse order (so first vendor is drawn on top)
    for vendor_idx in range(len(all_vendors_sorted) - 1, -1, -1):
        vendor = all_vendors_sorted[vendor_idx]
        rows = vendor_rows[vendor]

        for col_idx_iter, col_info in enumerate(chart_columns):
            if col_info[0] == 'separator':
                continue

            col_type, data_idx, sub_col, total_cols = col_info
            curr_row = rows[data_idx]
            prev_row = rows[data_idx - 1] if data_idx > 0 else curr_row

            if sub_col > 0:
                # Non-first columns: horizontal line at current value
                grid[curr_row][col_idx_iter] = (vendor_idx, 'flat')
            else:
                # First column (sub_col == 0): handle transition from previous value
                if prev_row == curr_row:
                    grid[curr_row][col_idx_iter] = (vendor_idx, 'flat')
                elif prev_row < curr_row:
                    # Going up: draw ╯ at prev level, │ in between, ╭ at curr level
                    grid[prev_row][col_idx_iter] = (vendor_idx, 'up_to_right')
                    for row in range(prev_row + 1, curr_row):
                        grid[row][col_idx_iter] = (vendor_idx, 'vertical')
                    grid[curr_row][col_idx_iter] = (vendor_idx, 'up_from_left')
                else:
                    # Going down: draw ╮ at prev level, │ in between, ╰ at curr level
                    grid[prev_row][col_idx_iter] = (vendor_idx, 'down_to_right')
                    for row in range(curr_row + 1, prev_row):
                        grid[row][col_idx_iter] = (vendor_idx, 'vertical')
                    grid[curr_row][col_idx_iter] = (vendor_idx, 'down_from_left')

    # Draw chart from top to bottom
    for row in range(chart_height - 1, -1, -1):
        y_val = min_value + (max_value - min_value) * row / (chart_height - 1)
        y_label = f"{format_y_axis_value(y_val)} |"

        line = ""
        for col_idx_iter, col_info in enumerate(chart_columns):
            if col_info[0] == 'separator':
                line += "|"
            else:
                cell = grid[row][col_idx_iter]
                if cell:
                    vendor_idx, char_type = cell
                    vendor = all_vendors_sorted[vendor_idx]
                    # Use bold characters for "All" line
                    if vendor == 'All':
                        char = char_map_bold.get(char_type, '━')
                    else:
                        char = char_map.get(char_type, '─')
                    color = VENDOR_COLORS.get(vendor, DEFAULT_COLORS[vendor_idx % len(DEFAULT_COLORS)])
                    line += f"{color}{char}{RESET_COLOR}"
                else:
                    line += " "

        print(y_label + line)

    # X-axis
    x_axis_line = ""
    for col_info in chart_columns:
        if col_info[0] == 'separator':
            x_axis_line += "\u2534"
        else:
            x_axis_line += "\u2500"
    print("      \u2514" + x_axis_line)

    # X-axis labels
    print()
    labels = []
    positions = []

    # Calculate time span in minutes
    first_time = sorted_times[-1]  # Oldest time (remember: reversed)
    # Use current time as the most recent point (leftmost on chart)
    last_time_axis = datetime.now().astimezone()
    time_span_minutes = (last_time_axis - first_time).total_seconds() / 60

    # Calculate how much time each tick should represent (~5% of total time span)
    target_tick_interval_minutes = time_span_minutes * 0.05

    # Round to a standard interval that divides 24h evenly for day consistency
    standard_intervals = [15, 30, 60, 120, 180, 240, 360, 480, 720, 1440]

    # Find the closest standard interval
    tick_interval_minutes = min(standard_intervals,
                               key=lambda x: abs(x - target_tick_interval_minutes))

    # Generate ticks at consistent positions across all days
    current_tick = first_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Advance to first tick boundary at or after first_time
    minutes_since_midnight = first_time.hour * 60 + first_time.minute
    ticks_since_midnight = (minutes_since_midnight + tick_interval_minutes - 1) // tick_interval_minutes
    current_tick += timedelta(minutes=ticks_since_midnight * tick_interval_minutes)

    # Track used positions to avoid placing multiple labels at the same position
    used_positions = set()

    # Generate ticks
    while current_tick <= last_time_axis:
        # Find closest data point to this tick
        closest_idx = None
        min_diff = None

        for i, time in enumerate(sorted_times):
            diff = abs((time - current_tick).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest_idx = i

        # Only add if we found a close match and this position hasn't been used
        if closest_idx is not None and closest_idx in data_to_col:
            position = data_to_col[closest_idx]
            if position not in used_positions:
                # Format label based on interval
                if tick_interval_minutes < 60:
                    label = current_tick.strftime('%H:%M')
                else:
                    label = current_tick.strftime('%H')
                labels.append(label)
                positions.append(position)
                used_positions.add(position)

        current_tick += timedelta(minutes=tick_interval_minutes)

    max_label_len = max(len(label) for label in labels) if labels else 0

    for char_idx in range(max_label_len):
        line = "       "
        for col_idx_iter, col_info in enumerate(chart_columns):
            if col_info[0] == 'separator':
                char_to_print = "|"
            else:
                char_to_print = " "
                for label_idx, pos in enumerate(positions):
                    if col_idx_iter == pos and char_idx < len(labels[label_idx]):
                        char_to_print = labels[label_idx][char_idx]
                        break
            line += char_to_print
        print(line)

    # Legend with percentages
    if show_legend:
        # Calculate total tokens per vendor over the time period
        vendor_totals = {}
        for vendor in all_vendors_sorted:
            if vendor == 'All':
                continue
            vendor_totals[vendor] = sum(vendor_data[vendor])

        # Calculate grand total (sum of all individual vendors)
        grand_total = sum(vendor_totals.values())

        legend_items = []
        # Show individual vendors first, then "All"
        legend_order = [v for v in all_vendors_sorted if v != 'All'] + ['All']
        for vendor in legend_order:
            color = VENDOR_COLORS.get(vendor, DEFAULT_COLORS[0])
            # Use bold character for "All"
            char = '━' if vendor == 'All' else '─'
            # Calculate percentage
            if vendor == 'All':
                pct_str = "100%"
            elif grand_total > 0:
                pct = (vendor_totals[vendor] / grand_total) * 100
                pct_str = f"{pct:.1f}%"
            else:
                pct_str = "0.0%"
            legend_items.append(f"{color}{char}{RESET_COLOR} {vendor}({pct_str})")
        legend = "Legend: " + "  ".join(legend_items)
        print(legend)
