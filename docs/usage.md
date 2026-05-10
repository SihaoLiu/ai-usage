# Usage

## CLI Options

```bash
vibe-usage [OPTIONS]

Options:
  --once              Show stats once and exit (no monitor loop)
  --days <N>          Number of days to analyze (default: 3)
  --vendor <VENDOR>   Filter to a single vendor: claude, codex, or gemini
```

Default behavior (no flags) enters monitor mode showing all vendors for the last 3 days.

## Monitor Mode Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `n` | Cycle to next vendor |
| `r` | Refresh now |
| `v claude` / `v codex` / `v gemini` / `v all` | Switch vendor |
| `d <N>` | Change days range |
| `a` | Switch to all vendors |
| `d` / `w` / `m` | Switch to 1 day / 7 days / 30 days |
| `date YYYY-MM-DD` | Show one complete local day |
| `range YYYY-MM-DD YYYY-MM-DD` | Show an inclusive local date span |
| `range YYYY-MM-DDTHH:MM YYYY-MM-DDTHH:MM` | Show an explicit local date-time span |
| `latest` / `last` | Return to the current rolling days range |

Monitor mode date and date-time inputs are interpreted in the running machine's local timezone. For date-only ranges, the ending date is included for the full local day. For example, `range 2026-05-01 2026-05-07` covers local time from `2026-05-01 00:00:00` through the end of `2026-05-07`.

## Data Sources

The tool reads local usage data written by each vendor's CLI:

| Vendor | Directory | File Pattern |
|--------|-----------|--------------|
| Claude | `~/.claude/projects/` | `**/*.jsonl` |
| Codex | `~/.codex/sessions/` | `YYYY/MM/DD/*.jsonl` |
| Gemini | `~/.gemini/tmp/` | `<hash>/chats/session-*.json` |

### Environment Variable Overrides

Override default config directories:

- `CLAUDE_CONFIG_DIR` (default: `~/.claude`)
- `CODEX_CONFIG_DIR` (default: `~/.codex`)
- `GEMINI_CONFIG_DIR` (default: `~/.gemini`)

## Display Modes

The table layout adapts to terminal width:

| Mode | Min Width | Description |
|------|-----------|-------------|
| Full | 205 cols | All columns with percentages |
| Medium | 137 cols | Shortened model names |
| Compact | 84 cols | Essential metrics only |
| Minimal | 70 cols | Basic summary |

## Subscription Fees

On first run, you will be prompted to enter your monthly subscription fees for each vendor. These are saved to `.fee.env` in the project directory and used to calculate monthly savings estimates.

## Supported Models

**Claude**: Opus 4.6, Opus 4.5, Sonnet 4.6, Sonnet 4.5, Sonnet 4, Haiku 4.5

**Codex/OpenAI**: GPT-5, GPT-5.1, GPT-5.3, GPT-4.1, o1, o3, o3-mini, o4-mini

**Gemini**: 3 Pro Preview, 2.5 Pro/Flash/Flash-Lite, 2.0 Flash/Flash-Lite

Model pricing is defined in `pricing.json` and can be updated manually.

## Architecture

```
src/
  main.rs            # Entry point, CLI args, monitor loop
  constants.rs       # Pricing loader (pricing.json + .fee.env)
  formatting.rs      # Responsive table formatting
  charts.rs          # ASCII chart rendering
  time_utils.rs      # Timezone and time utilities
  data/
    mod.rs           # Common types (UsageEntry, TokenUsage)
    claude.rs        # Claude JSONL reader
    codex.rs         # Codex JSONL reader
    gemini.rs        # Gemini JSON reader
  stats/
    mod.rs           # Model breakdown and time series calculation
    claude.rs        # Claude statistics
    codex.rs         # Codex statistics
    gemini.rs        # Gemini statistics
pricing.json         # API pricing data
```

## Testing

```bash
# Run all tests (requires release build)
cargo build --release && cargo test

# Snapshot tests only
cargo test --test snapshot_test

# Performance benchmarks
./scripts/bench.sh

# Capture Python/Rust comparison snapshots
./scripts/capture-snapshots.sh
```
