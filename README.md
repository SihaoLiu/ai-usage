# AI Usage Monitor

A high-performance terminal utility for monitoring token usage and costs across multiple AI coding assistants: **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI**. Written in Rust.

## Features

- **Multi-vendor support**: Track usage from Claude, Codex, and Gemini in one place
- **Cost analysis**: Calculate costs based on current API pricing with daily/weekly/monthly projections
- **Token breakdown**: Input, output, cache read, and cache creation tokens tracked separately
- **Visualizations**: ASCII-based multi-line charts and stacked bar charts
- **Responsive UI**: Adapts to terminal width with 4 display modes
- **Monitor mode**: Auto-refresh with interactive keyboard controls
- **Fast**: Processes 30 days of data in ~1 second

## Build

```bash
cargo build --release
```

The binary is at `target/release/vibe-usage`.

## Quick Start

```bash
# All vendors, last 7 days (default monitor mode)
./target/release/vibe-usage

# Single snapshot, no monitor
./target/release/vibe-usage --once

# Last 30 days
./target/release/vibe-usage --days 30

# Specific vendor only
./target/release/vibe-usage --vendor claude
./target/release/vibe-usage --vendor codex
./target/release/vibe-usage --vendor gemini
```

### Monitor Mode Controls

- `q` / `Esc`: Quit
- `n`: Cycle to next vendor
- `v claude` / `v codex` / `v gemini` / `v all`: Switch vendor
- `d <N>`: Change days range
- `r`: Refresh now

## Data Sources

The tool parses local usage data from each vendor's CLI:

| Vendor | Directory | File Pattern |
|--------|-----------|--------------|
| Claude | `~/.claude/projects/` | `**/*.jsonl` |
| Codex | `~/.codex/sessions/` | `YYYY/MM/DD/*.jsonl` |
| Gemini | `~/.gemini/tmp/` | `<hash>/chats/session-*.json` |

Environment variables can override default paths:
- `CLAUDE_CONFIG_DIR` (default: `~/.claude`)
- `CODEX_CONFIG_DIR` (default: `~/.codex`)
- `GEMINI_CONFIG_DIR` (default: `~/.gemini`)

## Architecture

```
src/
  main.rs            # Entry point, CLI args, monitor loop, vendor aggregation
  constants.rs       # Pricing configuration loader (pricing.json + .fee.env)
  formatting.rs      # Output formatting and responsive tables
  charts.rs          # ASCII chart visualization
  time_utils.rs      # Timezone and time formatting utilities
  data/
    mod.rs           # Common data types (UsageEntry, TokenUsage)
    claude.rs        # Claude data reader
    codex.rs         # Codex data reader
    gemini.rs        # Gemini data reader
  stats/
    mod.rs           # Generic model breakdown and time series calculation
    claude.rs        # Claude statistics wrappers
    codex.rs         # Codex statistics wrappers
    gemini.rs        # Gemini statistics wrappers
pricing.json         # API pricing data for all vendors
```

## Output

The tool generates:

1. **Usage statistics table** - Per-model breakdown with token counts and costs
2. **Time series charts** - Input/output tokens over time by model
3. **Cache usage charts** - Cache read/creation patterns
4. **Vendor comparison** - Side-by-side usage across vendors
5. **Cost summary** - Daily, weekly, and monthly projections

Display modes adapt to terminal width:
- **Full** (>=205 chars): Complete details with percentages
- **Medium** (>=137 chars): Shortened model names
- **Compact** (>=84 chars): Essential metrics only
- **Minimal** (>=70 chars): Basic info

## Testing

```bash
# Run all tests (requires release build)
cargo build --release && cargo test

# Run snapshot tests only
cargo test --test snapshot_test

# Capture snapshots for comparison
./scripts/capture-snapshots.sh

# Run performance benchmarks
./scripts/bench.sh
```

## Supported Models

**Claude**: Opus 4.6, Opus 4.5, Sonnet 4.6, Sonnet 4.5, Sonnet 4, Haiku 4.5

**Codex/OpenAI**: GPT-5, GPT-5.1, GPT-5.3, GPT-4.1, o1, o3, o3-mini, o4-mini

**Gemini**: 3 Pro Preview, 2.5 Pro/Flash/Flash-Lite, 2.0 Flash/Flash-Lite

## Legacy Python

The original Python implementation is preserved in the `legacy/` directory. The Rust version produces equivalent output and is used as the primary tool.

## License

MIT License - see [LICENSE](LICENSE) file.
