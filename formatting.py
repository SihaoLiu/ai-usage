"""Formatting utilities for Claude Code usage analysis."""

import math

from constants import (
    MODEL_PRICING, DEFAULT_PRICING, SUBSCRIPTION_PRICE,
    CODEX_MODEL_PRICING, CODEX_DEFAULT_PRICING, CODEX_SUBSCRIPTION_PRICE,
    GEMINI_MODEL_PRICING, GEMINI_DEFAULT_PRICING, GEMINI_SUBSCRIPTION_PRICE
)

# Short model name mapping for Claude
SHORT_MODEL_NAMES = {
    'claude-opus-4-5-20251101': 'Opus 4.5',
    'claude-opus-4-1-20250805': 'Opus 4.1',
    'claude-sonnet-4-5-20250929': 'Sonnet 4.5',
    'claude-sonnet-4-20250514': 'Sonnet 4',
    'claude-haiku-4-5-20251001': 'Haiku 4.5',
    '<synthetic>': 'synthetic',
}

# Short model name mapping for Codex (OpenAI)
# Note: Codex model names include effort level like "gpt-5-codex (high)"
CODEX_SHORT_MODEL_NAMES = {
    'gpt-5-codex': 'GPT-5 Codex',
    'gpt-5.1-codex': 'GPT-5.1 Codex',
    'gpt-5.1-codex-max': 'GPT-5.1 Max',
    'gpt-5.1-codex-mini': 'GPT-5.1 Mini',
    'codex-mini-latest': 'Codex Mini',
    'gpt-4.1': 'GPT-4.1',
    'gpt-4.1-mini': 'GPT-4.1 Mini',
    'gpt-4.1-nano': 'GPT-4.1 Nano',
    'o1': 'o1',
    'o3': 'o3',
    'o3-mini': 'o3-mini',
    'o4-mini': 'o4-mini',
}

# Short model name mapping for Gemini (Google)
GEMINI_SHORT_MODEL_NAMES = {
    'gemini-3-pro-preview': 'Gem 3 Pro',
    'gemini-3-pro-image-preview': 'Gem 3 Img',
    'gemini-2.5-pro': 'Gem 2.5 Pro',
    'gemini-2.5-flash': 'Gem 2.5 Fl',
    'gemini-2.5-flash-preview-09-2025': 'Gem 2.5 Fl',
    'gemini-2.5-flash-lite': 'Gem 2.5 Lt',
    'gemini-2.5-flash-lite-preview-09-2025': 'Gem 2.5 Lt',
    'gemini-2.0-flash': 'Gem 2.0 Fl',
    'gemini-2.0-flash-lite': 'Gem 2.0 Lt',
}


def get_short_model_name(model, vendor='claude'):
    """Get short display name for a model.

    Args:
        model: Model name (may include effort level for Codex like "gpt-5-codex (high)")
        vendor: 'claude', 'codex', or 'gemini'
    """
    if vendor == 'codex':
        # Codex model names may include effort level like "gpt-5-codex (high)"
        # Extract base model name and effort
        if ' (' in model and model.endswith(')'):
            base_model, effort = model.rsplit(' (', 1)
            effort = effort.rstrip(')')
            short_base = CODEX_SHORT_MODEL_NAMES.get(base_model, base_model[:10])
            # Abbreviate effort levels
            effort_short = {'low': 'L', 'medium': 'M', 'high': 'H', 'xhigh': 'XH'}.get(effort, effort[0].upper())
            return f"{short_base}({effort_short})"
        return CODEX_SHORT_MODEL_NAMES.get(model, model[:12] if len(model) > 12 else model)
    elif vendor == 'gemini':
        return GEMINI_SHORT_MODEL_NAMES.get(model, model[:12] if len(model) > 12 else model)
    else:
        return SHORT_MODEL_NAMES.get(model, model[:12] if len(model) > 12 else model)


def format_number(num):
    """Format number with thousand separators."""
    return f"{num:,}"


def format_number_compact(value):
    """Format number compactly with K/M/B suffixes."""
    if value >= 1_000_000_000:
        val_b = value / 1_000_000_000
        if val_b >= 100:
            return f"{int(val_b)}B"
        elif val_b >= 10:
            return f"{val_b:.1f}B"
        else:
            return f"{val_b:.2f}B"
    elif value >= 1_000_000:
        val_m = value / 1_000_000
        if val_m >= 100:
            return f"{int(val_m)}M"
        elif val_m >= 10:
            return f"{val_m:.1f}M"
        else:
            return f"{val_m:.2f}M"
    elif value >= 1_000:
        val_k = value / 1_000
        if val_k >= 100:
            return f"{int(val_k)}K"
        elif val_k >= 10:
            return f"{val_k:.1f}K"
        else:
            return f"{val_k:.2f}K"
    else:
        return f"{int(value)}"


def format_y_axis_value(value):
    """Format Y-axis value to always be 5 characters with K/M units."""
    if value >= 1_000_000:
        # Millions
        val_m = value / 1_000_000
        if val_m >= 100:
            return f"{int(val_m):3d} M"
        elif val_m >= 10:
            return f" {int(val_m):2d} M"
        else:
            return f"{val_m:3.1f} M"
    elif value >= 1000:
        # Thousands
        val_k = value / 1000
        if val_k >= 100:
            return f"{int(val_k):3d} K"
        elif val_k >= 10:
            return f" {int(val_k):2d} K"
        else:
            return f"{val_k:3.1f} K"
    else:
        # Less than 1000, show as integer
        return f"{int(value):5d}"


def format_total_value(value):
    """Format total value with B/M/K units."""
    if value >= 1_000_000_000:
        # Billions
        val_b = value / 1_000_000_000
        if val_b >= 100:
            return f"{int(val_b)}B"
        elif val_b >= 10:
            return f"{val_b:.1f}B"
        else:
            return f"{val_b:.2f}B"
    elif value >= 1_000_000:
        # Millions
        val_m = value / 1_000_000
        if val_m >= 100:
            return f"{int(val_m)}M"
        elif val_m >= 10:
            return f"{val_m:.1f}M"
        else:
            return f"{val_m:.2f}M"
    elif value >= 1_000:
        # Thousands
        val_k = value / 1_000
        if val_k >= 100:
            return f"{int(val_k)}K"
        elif val_k >= 10:
            return f"{val_k:.1f}K"
        else:
            return f"{val_k:.2f}K"
    else:
        # Less than 1000
        return f"{int(value)}"


def print_overall_stats(stats):
    """Print overall statistics."""
    print("Overall Usage Statistics")
    print("=" * 50)
    print()
    print(f"Total messages:        {format_number(stats['total_messages'])}")
    print()
    print(f"Input tokens:          {format_number(stats['input_tokens'])}")
    print(f"Output tokens:         {format_number(stats['output_tokens'])}")
    print(f"Cache output tokens:   {format_number(stats['cache_creation_tokens'])}")
    print(f"Cache input tokens:    {format_number(stats['cache_read_tokens'])}")
    print()
    print(f"Total tokens:          {format_number(stats['total_tokens'])}")


def format_with_pct(value, total, width):
    """Format a number with its percentage of total."""
    pct = (value / total * 100) if total > 0 else 0
    return f"{format_number(value)}({pct:4.1f}%)".rjust(width)


def format_with_pct_compact(value, total, width):
    """Format a compact number with its percentage of total."""
    pct = (value / total * 100) if total > 0 else 0
    return f"{format_number_compact(value)}({pct:4.1f}%)".rjust(width)


def format_no_pct(value, width):
    """Format a number without percentage."""
    return f"{format_number(value)}".rjust(width)


def format_no_pct_compact(value, width):
    """Format a compact number without percentage."""
    return f"{format_number_compact(value)}".rjust(width)


def format_with_100pct_up(value, width):
    """Format a number with (↑100%) indicator for TOTAL row."""
    return f"{format_number(value)}(↑100%)".rjust(width)


def format_with_100pct_left(value, width):
    """Format a value with (←100%) indicator for Cost total."""
    return f"${value:.2f}(←100%)".rjust(width)


def format_cost_per_mtok(value):
    """Format cost per MTok with appropriate precision.

    If value >= 0.01, show two decimal places (e.g., $0.01, $0.15)
    If value < 0.01, show at least one significant digit (e.g., $0.007, $0.0003)
    """
    if value >= 0.01:
        return f"${value:.2f}"
    elif value <= 0:
        return "$0.00"
    else:
        # Find the number of decimal places needed for one significant digit
        # Number of decimal places = ceil(-log10(value))
        decimal_places = int(math.ceil(-math.log10(value)))
        return f"${value:.{decimal_places}f}"


def get_table_display_mode(terminal_width, terminal_height, num_models):
    """Determine table display mode based on terminal dimensions.

    Returns:
        str: 'full', 'medium', 'compact', 'minimal', or 'hidden'

    Display modes (width thresholds include small margin):
    - full (width >= 205): Full model names, percentages, formatted numbers (~204 chars)
    - medium (width >= 137): Short model names, percentages, formatted numbers (~135 chars)
    - compact (width >= 84): Short model names, no percentages, compact numbers (~82 chars)
    - minimal (width >= 70): Short names, no pct, compact, fewer columns (~68 chars)
    - hidden: Terminal too narrow or too short
    """
    # Table requires ~10 lines for header/footer + 1 line per model
    min_table_height = 10 + num_models

    # If terminal is too short, hide the table
    if terminal_height < min_table_height + 20:  # Need room for charts too
        return 'hidden'

    if terminal_width >= 205:
        return 'full'
    elif terminal_width >= 137:
        return 'medium'
    elif terminal_width >= 84:
        return 'compact'
    elif terminal_width >= 70:
        return 'minimal'
    else:
        return 'hidden'


def print_model_breakdown(model_stats, days_in_data=7, terminal_width=None, terminal_height=None, vendor='claude'):
    """Print model breakdown table with responsive formatting.

    Args:
        model_stats: Model statistics to display
        days_in_data: Number of days the data covers (for cost projections)
        terminal_width: Terminal width (None for default full mode)
        terminal_height: Terminal height (None for default full mode)
        vendor: 'claude', 'codex', or 'gemini' (affects pricing and display)

    Returns:
        bool: True if table was printed, False if hidden due to space constraints
    """
    # Calculate sums first (needed for percentages)
    sum_messages = 0
    sum_input = 0
    sum_output = 0
    sum_total = 0
    sum_cache_creation = 0  # For Codex: reasoning; for Gemini: thinking
    sum_cache_read = 0
    sum_total_with_cache = 0

    for stats in model_stats:
        sum_messages += stats['count']
        sum_input += stats['input']
        sum_output += stats['output']
        sum_total += stats['total']
        # For Codex, sum 'reasoning' field; for Gemini, sum 'thinking'; for Claude, sum 'cache_creation'
        if vendor == 'codex':
            sum_cache_creation += stats.get('reasoning', 0)
        elif vendor == 'gemini':
            sum_cache_creation += stats.get('thinking', 0)
        else:
            sum_cache_creation += stats['cache_creation']
        sum_cache_read += stats['cache_read']
        sum_total_with_cache += stats['total_with_cache']

    # Determine display mode
    if terminal_width is None or terminal_height is None:
        mode = 'full'
    else:
        mode = get_table_display_mode(terminal_width, terminal_height, len(model_stats))

    if mode == 'hidden':
        return False

    # Calculate costs (needed for all modes)
    # Select pricing tables based on vendor
    if vendor == 'codex':
        pricing_table = CODEX_MODEL_PRICING
        default_pricing = CODEX_DEFAULT_PRICING
        subscription_price = CODEX_SUBSCRIPTION_PRICE
    elif vendor == 'gemini':
        pricing_table = GEMINI_MODEL_PRICING
        default_pricing = GEMINI_DEFAULT_PRICING
        subscription_price = GEMINI_SUBSCRIPTION_PRICE
    else:
        pricing_table = MODEL_PRICING
        default_pricing = DEFAULT_PRICING
        subscription_price = SUBSCRIPTION_PRICE

    input_cost = 0
    output_cost = 0
    cache_output_cost = 0  # For Codex: reasoning; for Gemini: thinking
    cache_input_cost = 0

    for stats in model_stats:
        model = stats['model']
        # For Codex, extract base model name (without effort level) for pricing lookup
        if vendor == 'codex' and ' (' in model and model.endswith(')'):
            base_model = model.rsplit(' (', 1)[0]
        else:
            base_model = model
        pricing = pricing_table.get(base_model, default_pricing)
        input_cost += stats['input'] * pricing['input'] / 1_000_000
        output_cost += stats['output'] * pricing['output'] / 1_000_000
        cache_input_cost += stats['cache_read'] * pricing['cache_input'] / 1_000_000
        # For Codex, reasoning tokens are billed at output rate
        # For Gemini, thinking tokens are billed at output rate
        # For Claude, cache_creation has its own pricing
        if vendor == 'codex':
            cache_output_cost += stats.get('reasoning', 0) * pricing['output'] / 1_000_000
        elif vendor == 'gemini':
            # Thinking tokens are billed at output rate for Gemini
            cache_output_cost += stats.get('thinking', 0) * pricing['output'] / 1_000_000
        else:
            cache_output_cost += stats['cache_creation'] * pricing['cache_output'] / 1_000_000

    io_total_cost = input_cost + output_cost
    cache_total_cost = cache_output_cost + cache_input_cost
    total_cost = io_total_cost + cache_total_cost

    # Print table based on mode
    if mode == 'full':
        _print_table_full(model_stats, sum_messages, sum_input, sum_output, sum_total,
                          sum_cache_creation, sum_cache_read, sum_total_with_cache,
                          input_cost, output_cost, io_total_cost,
                          cache_output_cost, cache_input_cost, total_cost, vendor)
    elif mode == 'medium':
        _print_table_medium(model_stats, sum_messages, sum_input, sum_output, sum_total,
                            sum_cache_creation, sum_cache_read, sum_total_with_cache,
                            input_cost, output_cost, io_total_cost,
                            cache_output_cost, cache_input_cost, total_cost, vendor)
    elif mode == 'compact':
        _print_table_compact(model_stats, sum_messages, sum_input, sum_output, sum_total,
                             sum_cache_creation, sum_cache_read, sum_total_with_cache,
                             input_cost, output_cost, io_total_cost,
                             cache_output_cost, cache_input_cost, total_cost, vendor)
    elif mode == 'minimal':
        _print_table_minimal(model_stats, sum_messages, sum_total, sum_total_with_cache, total_cost, vendor)

    # Print cost summary (all modes)
    daily_cost = total_cost / days_in_data if days_in_data > 0 else 0
    weekly_cost = daily_cost * 7
    monthly_cost = daily_cost * 30
    savings = monthly_cost - subscription_price
    monthly_tokens = (sum_total_with_cache / days_in_data) * 30 if days_in_data > 0 else 0
    cost_per_mtok = subscription_price / (monthly_tokens / 1_000_000) if monthly_tokens > 0 else 0

    if mode in ('full', 'medium'):
        print(f"Daily: ${daily_cost:.2f}, Weekly: ${weekly_cost:.2f}, Monthly(30d): ${monthly_cost:.2f}, Monthly Saving ${savings:.2f}, {format_cost_per_mtok(cost_per_mtok)} / MTok")
    else:
        # Shorter summary for compact/minimal modes
        print(f"Daily: ${daily_cost:.2f}, Monthly: ${monthly_cost:.2f}, Saving: ${savings:.2f}")

    return True


def _print_table_full(model_stats, sum_messages, sum_input, sum_output, sum_total,
                      sum_cache_creation, sum_cache_read, sum_total_with_cache,
                      input_cost, output_cost, io_total_cost,
                      cache_output_cost, cache_input_cost, total_cost, vendor='claude'):
    """Print full-width table (width ~204)."""
    print("Usage / Cost by Model")
    print("=" * 204)

    # Column headers based on vendor
    if vendor == 'codex':
        col2_name1 = 'Cache Read Input'
        col2_name2 = 'Reasoning Output'
    elif vendor == 'gemini':
        col2_name1 = 'Cache Read Input'
        col2_name2 = 'Thinking Output'
    else:
        col2_name1 = 'Cache Read Input'
        col2_name2 = 'Cache Creation Input'

    header = f"| {'Model':<35} {'Messages':>18} | {'Input':>22} {'Output':>22} {'Total':>22} | {col2_name1:>22} {col2_name2:>22} {'Total':>26} |"
    print(header)
    print("|" + "-" * 202 + "|")

    for stats in model_stats:
        msg_str = format_with_pct(stats['count'], sum_messages, 18)
        input_str = format_with_pct(stats['input'], sum_input, 22)
        output_str = format_with_pct(stats['output'], sum_output, 22)
        total_str = format_with_pct(stats['total'], sum_total, 22)
        cache_read_str = format_with_pct(stats['cache_read'], sum_cache_read, 22)
        # For Codex, use 'reasoning'; for Gemini, use 'thinking'; for Claude, use 'cache_creation'
        if vendor == 'codex':
            col2_val2 = stats.get('reasoning', 0)
        elif vendor == 'gemini':
            col2_val2 = stats.get('thinking', 0)
        else:
            col2_val2 = stats['cache_creation']
        col2_str2 = format_with_pct(col2_val2, sum_cache_creation, 22)
        total_with_cache_str = format_with_pct(stats['total_with_cache'], sum_total_with_cache, 26)

        row = (f"| {stats['model']:<35} "
               f"{msg_str} | "
               f"{input_str} "
               f"{output_str} "
               f"{total_str} | "
               f"{cache_read_str} "
               f"{col2_str2} "
               f"{total_with_cache_str} |")
        print(row)

    print("|" + "-" * 202 + "|")
    sum_row = (f"| {'TOTAL':<35} "
               f"{format_with_100pct_up(sum_messages, 18)} | "
               f"{format_with_100pct_up(sum_input, 22)} "
               f"{format_with_100pct_up(sum_output, 22)} "
               f"{format_with_100pct_up(sum_total, 22)} | "
               f"{format_with_100pct_up(sum_cache_read, 22)} "
               f"{format_with_100pct_up(sum_cache_creation, 22)} "
               f"{format_with_100pct_up(sum_total_with_cache, 26)} |")
    print(sum_row)

    def format_cost_with_pct(cost, total, width):
        pct = (cost / total * 100) if total > 0 else 0
        return f"${cost:.2f}({pct:4.1f}%)".rjust(width)

    cost_row = (f"| {'Cost(API)':<35} "
                f"{'':>18} | "
                f"{format_cost_with_pct(input_cost, total_cost, 22)} "
                f"{format_cost_with_pct(output_cost, total_cost, 22)} "
                f"{format_cost_with_pct(io_total_cost, total_cost, 22)} | "
                f"{format_cost_with_pct(cache_input_cost, total_cost, 22)} "
                f"{format_cost_with_pct(cache_output_cost, total_cost, 22)} "
                f"{format_with_100pct_left(total_cost, 26)} |")
    print(cost_row)
    print("=" * 204)


def _print_table_medium(model_stats, sum_messages, sum_input, sum_output, sum_total,
                        sum_cache_creation, sum_cache_read, sum_total_with_cache,
                        input_cost, output_cost, io_total_cost,
                        cache_output_cost, cache_input_cost, total_cost, vendor='claude'):
    """Print medium-width table with short names (width ~135)."""
    # Column widths sized for typical data with percentages
    w_model = 12
    w_msgs = 15
    w_io = 18      # Input/Output/Total columns
    w_cache = 20   # Cache columns (larger numbers)

    table_width = 135

    print("Usage / Cost by Model")
    print("=" * table_width)

    # Column headers based on vendor (shorter names for medium width)
    if vendor == 'codex':
        col2_name1 = 'CacheReadIn'
        col2_name2 = 'ReasonOut'
    elif vendor == 'gemini':
        col2_name1 = 'CacheReadIn'
        col2_name2 = 'ThinkingOut'
    else:
        col2_name1 = 'CacheReadIn'
        col2_name2 = 'CacheCreateIn'

    header = (f"| {'Model':<{w_model}} {'Msgs':>{w_msgs}} "
              f"| {'Input':>{w_io}} {'Output':>{w_io}} {'Total':>{w_io}} "
              f"| {col2_name1:>{w_cache}} {col2_name2:>{w_cache}} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        model_name = get_short_model_name(stats['model'], vendor)
        cache_read_str = format_with_pct(stats['cache_read'], sum_cache_read, w_cache)
        # For Codex, use 'reasoning'; for Gemini, use 'thinking'; for Claude, use 'cache_creation'
        if vendor == 'codex':
            col2_val2 = stats.get('reasoning', 0)
        elif vendor == 'gemini':
            col2_val2 = stats.get('thinking', 0)
        else:
            col2_val2 = stats['cache_creation']
        col2_str2 = format_with_pct(col2_val2, sum_cache_creation, w_cache)
        row = (f"| {model_name:<{w_model}} "
               f"{format_with_pct(stats['count'], sum_messages, w_msgs)} "
               f"| {format_with_pct(stats['input'], sum_input, w_io)} "
               f"{format_with_pct(stats['output'], sum_output, w_io)} "
               f"{format_with_pct(stats['total'], sum_total, w_io)} "
               f"| {cache_read_str} "
               f"{col2_str2} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<{w_model}} "
               f"{format_with_100pct_up(sum_messages, w_msgs)} "
               f"| {format_with_100pct_up(sum_input, w_io)} "
               f"{format_with_100pct_up(sum_output, w_io)} "
               f"{format_with_100pct_up(sum_total, w_io)} "
               f"| {format_with_100pct_up(sum_cache_read, w_cache)} "
               f"{format_with_100pct_up(sum_cache_creation, w_cache)} |")
    print(sum_row)

    def format_cost_with_pct(cost, total, width):
        pct = (cost / total * 100) if total > 0 else 0
        return f"${cost:.2f}({pct:4.1f}%)".rjust(width)

    # Medium mode has no total cost column with breakdown, so use simple format
    cost_row = (f"| {'Cost(API)':<{w_model}} "
                f"{'':>{w_msgs}} "
                f"| {format_cost_with_pct(input_cost, total_cost, w_io)} "
                f"{format_cost_with_pct(output_cost, total_cost, w_io)} "
                f"{format_cost_with_pct(io_total_cost, total_cost, w_io)} "
                f"| {format_cost_with_pct(cache_input_cost, total_cost, w_cache)} "
                f"{format_cost_with_pct(cache_output_cost, total_cost, w_cache)} |")
    print(cost_row)
    print("=" * table_width)


def _print_table_compact(model_stats, sum_messages, sum_input, sum_output, sum_total,
                         sum_cache_creation, sum_cache_read, sum_total_with_cache,
                         input_cost, output_cost, io_total_cost,
                         cache_output_cost, cache_input_cost, total_cost, vendor='claude'):
    """Print compact table with short names, no percentages (width ~82)."""
    # Column widths for compact numbers (K/M/B format)
    w_model = 12
    w_msgs = 7
    w_val = 8   # For compact numbers and costs

    table_width = 82

    print("Usage / Cost by Model")
    print("=" * table_width)

    # Column headers based on vendor (compact names)
    if vendor == 'codex':
        col2_name1 = 'CacheRIn'
        col2_name2 = 'ReasnOut'
    elif vendor == 'gemini':
        col2_name1 = 'CacheRIn'
        col2_name2 = 'ThinkOut'
    else:
        col2_name1 = 'CacheRIn'
        col2_name2 = 'CacheCIn'

    header = (f"| {'Model':<{w_model}} {'Msgs':>{w_msgs}} "
              f"| {'Input':>{w_val}} {'Output':>{w_val}} {'Total':>{w_val}} "
              f"| {col2_name1:>{w_val}} {col2_name2:>{w_val}} {'Total':>{w_val}} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        model_name = get_short_model_name(stats['model'], vendor)
        # For Codex, use 'reasoning'; for Gemini, use 'thinking'; for Claude, use 'cache_creation'
        if vendor == 'codex':
            col2_val2 = stats.get('reasoning', 0)
        elif vendor == 'gemini':
            col2_val2 = stats.get('thinking', 0)
        else:
            col2_val2 = stats['cache_creation']
        row = (f"| {model_name:<{w_model}} "
               f"{format_number_compact(stats['count']):>{w_msgs}} "
               f"| {format_number_compact(stats['input']):>{w_val}} "
               f"{format_number_compact(stats['output']):>{w_val}} "
               f"{format_number_compact(stats['total']):>{w_val}} "
               f"| {format_number_compact(stats['cache_read']):>{w_val}} "
               f"{format_number_compact(col2_val2):>{w_val}} "
               f"{format_number_compact(stats['total_with_cache']):>{w_val}} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<{w_model}} "
               f"{format_number_compact(sum_messages):>{w_msgs}} "
               f"| {format_number_compact(sum_input):>{w_val}} "
               f"{format_number_compact(sum_output):>{w_val}} "
               f"{format_number_compact(sum_total):>{w_val}} "
               f"| {format_number_compact(sum_cache_read):>{w_val}} "
               f"{format_number_compact(sum_cache_creation):>{w_val}} "
               f"{format_number_compact(sum_total_with_cache):>{w_val}} |")
    print(sum_row)

    cost_row = (f"| {'Cost':<{w_model}} "
                f"{'':>{w_msgs}} "
                f"| ${input_cost:>{w_val - 1}.2f} "
                f"${output_cost:>{w_val - 1}.2f} "
                f"${io_total_cost:>{w_val - 1}.2f} "
                f"| ${cache_input_cost:>{w_val - 1}.2f} "
                f"${cache_output_cost:>{w_val - 1}.2f} "
                f"${total_cost:>{w_val - 1}.2f} |")
    print(cost_row)
    print("=" * table_width)


def _print_table_minimal(model_stats, sum_messages, sum_total, sum_total_with_cache, total_cost, vendor='claude'):
    """Print minimal table - just model, messages, totals."""
    # Column widths
    w_model = 12
    w_msgs = 7
    w_io = 10
    w_cache = 12
    w_all = 12

    # Calculate actual width: | Model<12> Msgs<7> | I/O<10> Cache<12> All<12> |
    # = 2 + 12 + 1 + 7 + 1 + 2 + 10 + 1 + 12 + 1 + 12 + 2 = 63
    table_width = 2 + w_model + 1 + w_msgs + 1 + 2 + w_io + 1 + w_cache + 1 + w_all + 2

    print("Usage Summary")
    print("=" * table_width)

    header = (f"| {'Model':<{w_model}} {'Msgs':>{w_msgs}} "
              f"| {'I/O Total':>{w_io}} {'Cache Total':>{w_cache}} {'All Total':>{w_all}} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        model_name = get_short_model_name(stats['model'], vendor)
        row = (f"| {model_name:<{w_model}} "
               f"{format_number_compact(stats['count']):>{w_msgs}} "
               f"| {format_number_compact(stats['total']):>{w_io}} "
               f"{format_number_compact(stats['cache_creation'] + stats['cache_read']):>{w_cache}} "
               f"{format_number_compact(stats['total_with_cache']):>{w_all}} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<{w_model}} "
               f"{format_number_compact(sum_messages):>{w_msgs}} "
               f"| {format_number_compact(sum_total):>{w_io}} "
               f"{format_number_compact(sum_total_with_cache - sum_total):>{w_cache}} "
               f"{format_number_compact(sum_total_with_cache):>{w_all}} |")
    print(sum_row)

    cost_row = (f"| {'Cost':<{w_model}} "
                f"{'':>{w_msgs}} "
                f"| {'':>{w_io}} "
                f"{'':>{w_cache}} "
                f"${total_cost:>{w_all - 1}.2f} |")
    print(cost_row)
    print("=" * table_width)
