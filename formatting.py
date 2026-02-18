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


def _format_model_name_with_vendor_prefix(model, vendor='claude', show_vendor_prefix=False, use_short_name=True, prefix_width=6):
    """Format model name for table output, optionally with explicit vendor prefix."""
    name = get_short_model_name(model, vendor) if use_short_name else model
    if not show_vendor_prefix:
        return name
    # Keep vendor prefixes aligned across rows by normalizing width.
    aligned_vendor = f"{vendor}".ljust(prefix_width)
    return f"{aligned_vendor}: {name}"


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


def _get_strategy_totals(stats, vendor='claude'):
    """Return vendor-specific token buckets for the final summary table."""
    resolved_vendor = stats.get('vendor', vendor)
    cache_hit = stats['cache_read']

    if resolved_vendor == 'claude':
        prefill = stats['input'] + stats['cache_creation']
        decoding = stats['output']
    elif resolved_vendor == 'codex':
        prefill = stats['input']
        decoding = stats['output'] + stats.get('reasoning', 0)
    elif resolved_vendor == 'gemini':
        prefill = stats['input']
        decoding = stats['output'] + stats.get('thinking', 0)
    else:
        # Fallback to Claude-style decomposition
        prefill = stats['input']
        decoding = stats['output']

    return cache_hit, prefill, decoding


def _get_strategy_costs(input_cost, output_cost, cache_output_cost, cache_input_cost, vendor='claude'):
    """Return cost totals grouped by strategy buckets for display only."""
    cache_hit_cost = cache_input_cost
    if vendor == 'claude':
        prefill_cost = input_cost + cache_output_cost
        decoding_cost = output_cost
    else:
        prefill_cost = input_cost
        decoding_cost = output_cost + cache_output_cost
    return cache_hit_cost, prefill_cost, decoding_cost


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
    sum_cache_hit = 0
    sum_prefill = 0
    sum_decoding = 0
    sum_total_with_cache = 0

    for stats in model_stats:
        sum_messages += stats['count']
        sum_input += stats['input']
        sum_output += stats['output']
        sum_total += stats['total']
        cache_hit, prefill, decoding = _get_strategy_totals(stats, vendor=vendor)
        sum_cache_hit += cache_hit
        sum_prefill += prefill
        sum_decoding += decoding
        sum_total_with_cache += cache_hit + prefill + decoding

    # Determine display mode
    if terminal_width is None or terminal_height is None:
        mode = 'full'
    else:
        mode = get_table_display_mode(terminal_width, terminal_height, len(model_stats))

    if mode == 'hidden':
        return False

    # Calculate costs (needed for all modes)
    input_cost = 0
    output_cost = 0
    cache_output_cost = 0  # For Codex: reasoning; for Gemini: thinking
    cache_input_cost = 0
    cache_hit_cost = 0
    prefill_cost = 0
    decoding_cost = 0
    if vendor == 'codex':
        subscription_price = CODEX_SUBSCRIPTION_PRICE
    elif vendor == 'gemini':
        subscription_price = GEMINI_SUBSCRIPTION_PRICE
    else:
        subscription_price = SUBSCRIPTION_PRICE

    for stats in model_stats:
        resolved_vendor = stats.get('vendor', vendor)

        # Select pricing tables based on row vendor
        if resolved_vendor == 'codex':
            pricing_table = CODEX_MODEL_PRICING
            default_pricing = CODEX_DEFAULT_PRICING
        elif resolved_vendor == 'gemini':
            pricing_table = GEMINI_MODEL_PRICING
            default_pricing = GEMINI_DEFAULT_PRICING
        else:
            pricing_table = MODEL_PRICING
            default_pricing = DEFAULT_PRICING

        model = stats['model']
        # For Codex, extract base model name (without effort level) for pricing lookup
        if resolved_vendor == 'codex' and ' (' in model and model.endswith(')'):
            base_model = model.rsplit(' (', 1)[0]
        else:
            base_model = model
        pricing = pricing_table.get(base_model, default_pricing)
        row_input_cost = stats['input'] * pricing['input'] / 1_000_000
        row_output_cost = stats['output'] * pricing['output'] / 1_000_000
        row_cache_input_cost = stats['cache_read'] * pricing['cache_input'] / 1_000_000
        # For Codex, reasoning tokens are billed at output rate
        # For Gemini, thinking tokens are billed at output rate
        # For Claude, cache_creation has its own pricing
        if resolved_vendor == 'codex':
            row_cache_output_cost = stats.get('reasoning', 0) * pricing['output'] / 1_000_000
        elif resolved_vendor == 'gemini':
            # Thinking tokens are billed at output rate for Gemini
            row_cache_output_cost = stats.get('thinking', 0) * pricing['output'] / 1_000_000
        else:
            row_cache_output_cost = stats['cache_creation'] * pricing['cache_output'] / 1_000_000

        row_prefill_cost, row_decoding_cost = _get_strategy_costs(
            row_input_cost,
            row_output_cost,
            row_cache_output_cost,
            row_cache_input_cost,
            vendor=resolved_vendor
        )[1:]

        input_cost += row_input_cost
        output_cost += row_output_cost
        cache_output_cost += row_cache_output_cost
        cache_input_cost += row_cache_input_cost
        cache_hit_cost += row_cache_input_cost
        prefill_cost += row_prefill_cost
        decoding_cost += row_decoding_cost

    io_total_cost = input_cost + output_cost
    cache_total_cost = cache_output_cost + cache_input_cost
    total_cost = io_total_cost + cache_total_cost

    # Print table based on mode
    if mode == 'full':
        _print_table_full(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                          input_cost, output_cost, io_total_cost,
                          cache_output_cost, cache_input_cost, total_cost, vendor,
                          cache_hit_cost, prefill_cost, decoding_cost,
                          show_vendor_prefix=(vendor == 'all'))
    elif mode == 'medium':
        _print_table_medium(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                            input_cost, output_cost, io_total_cost,
                            cache_output_cost, cache_input_cost, total_cost, vendor,
                            cache_hit_cost, prefill_cost, decoding_cost,
                            show_vendor_prefix=(vendor == 'all'))
    elif mode == 'compact':
        _print_table_compact(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                             input_cost, output_cost, io_total_cost,
                             cache_output_cost, cache_input_cost, total_cost, vendor,
                             cache_hit_cost, prefill_cost, decoding_cost,
                             show_vendor_prefix=(vendor == 'all'))
    elif mode == 'minimal':
        _print_table_minimal(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                             total_cost, vendor,
                             cache_hit_cost, prefill_cost, decoding_cost,
                             show_vendor_prefix=(vendor == 'all'))

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


def _print_table_full(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                      input_cost, output_cost, io_total_cost,
                      cache_output_cost, cache_input_cost, total_cost, vendor='claude',
                      cache_hit_cost=None, prefill_cost=None, decoding_cost=None, show_vendor_prefix=False):
    """Print full-width table (width ~204)."""
    table_width = 152

    print("Usage / Cost by Model")
    print("=" * table_width)

    header = (f"| {'Model':<35} {'Messages':>18} | "
              f"{'Cache Hit':>22} {'Prefill':>22} {'Decoding':>22} {'Total':>22} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        msg_str = format_with_pct(stats['count'], sum_messages, 18)
        effective_vendor = stats.get('vendor', vendor)
        cache_hit, prefill, decoding = _get_strategy_totals(stats, vendor=effective_vendor)
        model_name = _format_model_name_with_vendor_prefix(stats['model'], effective_vendor, show_vendor_prefix=show_vendor_prefix, use_short_name=False)
        cache_hit_str = format_with_pct(cache_hit, sum_cache_hit, 22)
        prefill_str = format_with_pct(prefill, sum_prefill, 22)
        decoding_str = format_with_pct(decoding, sum_decoding, 22)
        total_with_cache_str = format_with_pct(cache_hit + prefill + decoding, sum_total_with_cache, 22)

        row = (f"| {model_name:<35} "
               f"{msg_str} | "
               f"{cache_hit_str} "
               f"{prefill_str} "
               f"{decoding_str} "
               f"{total_with_cache_str} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<35} "
               f"{format_with_100pct_up(sum_messages, 18)} | "
               f"{format_with_100pct_up(sum_cache_hit, 22)} "
               f"{format_with_100pct_up(sum_prefill, 22)} "
               f"{format_with_100pct_up(sum_decoding, 22)} "
               f"{format_with_100pct_up(sum_total_with_cache, 22)} |")
    print(sum_row)

    def format_cost_with_pct(cost, total, width):
        pct = (cost / total * 100) if total > 0 else 0
        return f"${cost:.2f}({pct:4.1f}%)".rjust(width)

    if cache_hit_cost is None or prefill_cost is None or decoding_cost is None:
        cache_hit_cost, prefill_cost, decoding_cost = _get_strategy_costs(
            input_cost, output_cost, cache_output_cost, cache_input_cost, vendor=vendor
        )

    cost_row = (f"| {'Cost(API)':<35} "
                f"{'':>18} | "
                f"{format_cost_with_pct(cache_hit_cost, total_cost, 22)} "
                f"{format_cost_with_pct(prefill_cost, total_cost, 22)} "
                f"{format_cost_with_pct(decoding_cost, total_cost, 22)} "
                f"{format_with_100pct_left(total_cost, 22)} |")
    print(cost_row)
    print("=" * table_width)


def _print_table_medium(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                        input_cost, output_cost, io_total_cost,
                        cache_output_cost, cache_input_cost, total_cost, vendor='claude',
                        cache_hit_cost=None, prefill_cost=None, decoding_cost=None, show_vendor_prefix=False):
    """Print medium-width table with short names (width ~135)."""
    # Column widths sized for typical data with percentages
    w_model = 12
    w_msgs = 15
    w_cache = 20   # Cache columns (larger numbers)

    table_width = 118

    print("Usage / Cost by Model")
    print("=" * table_width)

    header = (f"| {'Model':<{w_model}} {'Msgs':>{w_msgs}} "
              f"| {'CacheHit':>{w_cache}} {'Prefill':>{w_cache}} {'Decode':>{w_cache}} {'Total':>{w_cache}} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        effective_vendor = stats.get('vendor', vendor)
        model_name = _format_model_name_with_vendor_prefix(stats['model'], effective_vendor, show_vendor_prefix=show_vendor_prefix)
        cache_hit, prefill, decoding = _get_strategy_totals(stats, vendor=effective_vendor)
        cache_hit_str = format_with_pct(cache_hit, sum_cache_hit, w_cache)
        prefill_str = format_with_pct(prefill, sum_prefill, w_cache)
        decoding_str = format_with_pct(decoding, sum_decoding, w_cache)
        total_str = format_with_pct(cache_hit + prefill + decoding, sum_total_with_cache, w_cache)
        row = (f"| {model_name:<{w_model}} "
               f"{format_with_pct(stats['count'], sum_messages, w_msgs)} "
               f"| {cache_hit_str} "
               f"{prefill_str} "
               f"{decoding_str} "
               f"{total_str} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<{w_model}} "
               f"{format_with_100pct_up(sum_messages, w_msgs)} "
               f"| {format_with_100pct_up(sum_cache_hit, w_cache)} "
               f"{format_with_100pct_up(sum_prefill, w_cache)} "
               f"{format_with_100pct_up(sum_decoding, w_cache)} "
               f"{format_with_100pct_up(sum_total_with_cache, w_cache)} |")
    print(sum_row)

    def format_cost_with_pct(cost, total, width):
        pct = (cost / total * 100) if total > 0 else 0
        return f"${cost:.2f}({pct:4.1f}%)".rjust(width)

    if cache_hit_cost is None or prefill_cost is None or decoding_cost is None:
        cache_hit_cost, prefill_cost, decoding_cost = _get_strategy_costs(
            input_cost, output_cost, cache_output_cost, cache_input_cost, vendor=vendor
        )

    cost_row = (f"| {'Cost(API)':<{w_model}} "
                f"{'':>{w_msgs}} "
                f"{format_cost_with_pct(cache_hit_cost, total_cost, w_cache)} "
                f"{format_cost_with_pct(prefill_cost, total_cost, w_cache)} "
                f"{format_cost_with_pct(decoding_cost, total_cost, w_cache)} "
                f"{format_with_100pct_left(total_cost, w_cache)} |")
    print(cost_row)
    print("=" * table_width)


def _print_table_compact(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache,
                         input_cost, output_cost, io_total_cost,
                         cache_output_cost, cache_input_cost, total_cost, vendor='claude',
                         cache_hit_cost=None, prefill_cost=None, decoding_cost=None, show_vendor_prefix=False):
    """Print compact table with short names, no percentages (width ~82)."""
    # Column widths for compact numbers (K/M/B format)
    w_model = 12
    w_msgs = 7
    w_val = 8   # For compact numbers and costs

    table_width = 62

    print("Usage / Cost by Model")
    print("=" * table_width)

    header = (f"| {'Model':<{w_model}} {'Msgs':>{w_msgs}} "
              f"| {'CacheHit':>{w_val}} {'Prefill':>{w_val}} {'Decode':>{w_val}} {'Total':>{w_val}} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        effective_vendor = stats.get('vendor', vendor)
        model_name = _format_model_name_with_vendor_prefix(stats['model'], effective_vendor, show_vendor_prefix=show_vendor_prefix)
        cache_hit, prefill, decoding = _get_strategy_totals(stats, vendor=effective_vendor)
        row = (f"| {model_name:<{w_model}} "
               f"{format_number_compact(stats['count']):>{w_msgs}} "
               f"| {format_number_compact(cache_hit):>{w_val}} "
               f"{format_number_compact(prefill):>{w_val}} "
               f"{format_number_compact(decoding):>{w_val}} "
               f"{format_number_compact(cache_hit + prefill + decoding):>{w_val}} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<{w_model}} "
               f"{format_number_compact(sum_messages):>{w_msgs}} "
               f"| {format_number_compact(sum_cache_hit):>{w_val}} "
               f"{format_number_compact(sum_prefill):>{w_val}} "
               f"{format_number_compact(sum_decoding):>{w_val}} "
               f"{format_number_compact(sum_total_with_cache):>{w_val}} |")
    print(sum_row)

    if cache_hit_cost is None or prefill_cost is None or decoding_cost is None:
        cache_hit_cost, prefill_cost, decoding_cost = _get_strategy_costs(
            input_cost, output_cost, cache_output_cost, cache_input_cost, vendor=vendor
        )

    cost_row = (f"| {'Cost':<{w_model}} "
                f"{'':>{w_msgs}} "
                f"| ${cache_hit_cost:>{w_val - 1}.2f} "
                f"${prefill_cost:>{w_val - 1}.2f} "
                f"${decoding_cost:>{w_val - 1}.2f} "
                f"${total_cost:>{w_val - 1}.2f} |")
    print(cost_row)
    print("=" * table_width)


def _print_table_minimal(model_stats, sum_messages, sum_cache_hit, sum_prefill, sum_decoding, sum_total_with_cache, total_cost, vendor='claude',
                        cache_hit_cost=None, prefill_cost=None, decoding_cost=None, show_vendor_prefix=False):
    """Print minimal table - just model, messages, and strategy totals."""
    # Column widths
    w_model = 12
    w_msgs = 7
    w_strategy = 10

    # Calculate actual width: | Model<12> Msgs<7> | Cache Hit/Prefill/Decode/Total |
    table_width = 2 + w_model + 1 + w_msgs + 1 + 2 + w_strategy * 4 + 3 + 2

    print("Usage Summary")
    print("=" * table_width)

    header = (f"| {'Model':<{w_model}} {'Msgs':>{w_msgs}} "
              f"| {'Cache Hit':>{w_strategy}} {'Prefill':>{w_strategy}} {'Decode':>{w_strategy}} {'All':>{w_strategy}} |")
    print(header)
    print("|" + "-" * (table_width - 2) + "|")

    for stats in model_stats:
        effective_vendor = stats.get('vendor', vendor)
        model_name = _format_model_name_with_vendor_prefix(stats['model'], effective_vendor, show_vendor_prefix=show_vendor_prefix)
        cache_hit, prefill, decoding = _get_strategy_totals(stats, vendor=effective_vendor)
        row = (f"| {model_name:<{w_model}} "
               f"{format_number_compact(stats['count']):>{w_msgs}} "
               f"| {format_number_compact(cache_hit):>{w_strategy}} "
               f"{format_number_compact(prefill):>{w_strategy}} "
               f"{format_number_compact(decoding):>{w_strategy}} "
               f"{format_number_compact(cache_hit + prefill + decoding):>{w_strategy}} |")
        print(row)

    print("|" + "-" * (table_width - 2) + "|")
    sum_row = (f"| {'TOTAL':<{w_model}} "
               f"{format_number_compact(sum_messages):>{w_msgs}} "
               f"| {format_number_compact(sum_cache_hit):>{w_strategy}} "
               f"{format_number_compact(sum_prefill):>{w_strategy}} "
               f"{format_number_compact(sum_decoding):>{w_strategy}} "
               f"{format_number_compact(sum_total_with_cache):>{w_strategy}} |")
    print(sum_row)

    cost_row = (f"| {'Cost':<{w_model}} "
                f"{'':>{w_msgs}} "
                f"| {'':>{w_strategy}} "
                f"{'':>{w_strategy}} "
                f"{'':>{w_strategy}} "
                f"${total_cost:>{w_strategy - 1}.2f} |")
    print(cost_row)
    print("=" * table_width)
