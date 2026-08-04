# Usage

## CLI Options

```bash
ai-usage [OPTIONS]

Options:
  --once              Show stats once and exit (no monitor loop)
  --days <N>          Number of days to analyze (default: 3)
  --tool <TOOL>       Filter to a single tool: claude, codex, gemini, kimi, or omp
  --view <VIEW>       Table shape: flat or vendor (default: flat)
  --sort <KEY>        Descending sort key: msgs, cache, prefill, decode, total,
                      cost, or rate (default: msgs)
  --host <HOST>       Filter to a single machine id
  --auto-update       Periodically check GitHub Releases in monitor mode
  --auto-update-interval-seconds <N>
                      Seconds between automatic release checks (default: 3600)
```

Default behavior (no flags) enters monitor mode showing all harnesses for the last 3 days.

## JSON Snapshots

Use the `snapshot` command for local dashboards and other machine consumers:

```bash
ai-usage snapshot --days 7 --tool all
```

The command writes one versioned JSON document to stdout. Diagnostics are written
to stderr. `--tool` accepts `claude`, `codex`, `gemini`, `kimi`, `omp`, or `all`;
`--host` and `--session` apply the same filters as the interactive dashboard.

The document contains `schema_version`, `generated_at`, `timezone`, `range`,
`window`, `daily`, and `top_models`. Token totals use inference-oriented names:
`cache_hit`, `prefill`, and `decoding`. API costs are estimates in US dollars.

## Monitor Mode Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `n` | Cycle to the next harness |
| `r` | Refresh now |
| `t claude` / `t codex` / `t gemini` / `t kimi` / `t omp` / `t all` | Switch tool (also `tool <name>`) |
| `host all` / `host <HOST>` | Switch between all machines and one machine |
| `d <N>` | Change days range |
| `a` | Switch to all harnesses |
| `v` / `view flat` / `view vendor` | Toggle or select the table view |
| `s` | Select the next descending sort key |
| `sort <KEY>` | Sort by `msgs`, `cache`, `prefill`, `decode`, `total`, `cost`, or `rate` |
| `d` / `w` / `m` | Switch to 1 day / 7 days / 30 days |
| `date YYYY-MM-DD` | Show one complete local day |
| `range YYYY-MM-DD YYYY-MM-DD` | Show an inclusive local date span |
| `range YYYY-MM-DDTHH:MM YYYY-MM-DDTHH:MM` | Show an explicit local date-time span |
| `latest` / `last` | Return to the current rolling days range |
| `cost` / `cost all` / `cost <HARNESS>` | Show all monthly fixed costs or one harness's cost |
| `cost <HARNESS> <AMOUNT>` | Save a non-negative monthly fixed cost and redraw |
| Up / Down arrows | Recall previous / next typed command (shell-style history) |
| Left / Right arrows | Empty prompt: slide newer / older by one chart interval; text prompt: move cursor |
| PgUp / PgDn | Slide the time window backward / forward by its width (PgDn snaps to the present) |
| + / - | Empty prompt: zoom the time window in / out |
| `update` | Download the latest GitHub release for this platform and restart in place |
| `e` / `exit` | Exit monitor mode |

Monitor mode date and date-time inputs are interpreted in the running machine's local timezone. For date-only ranges, the ending date is included for the full local day. For example, `range 2026-05-01 2026-05-07` covers local time from `2026-05-01 00:00:00` through the end of `2026-05-07`. The two arguments to `range` may be supplied in any order; the earlier instant is treated as the start and the later one as the end.

Automatic updates are disabled unless `--auto-update` is provided. When enabled, monitor mode checks the latest GitHub release on the configured interval, downloads the matching `ai-usage-<target>` asset when a newer version exists, replaces the current executable, and restarts in place.

## Data Sources

The tool reads local usage data written by each harness:

| Harness | Directory | File Pattern |
|--------|-----------|--------------|
| Claude | `~/.claude/projects/` | `**/*.jsonl` |
| Codex | `~/.codex/sessions/` | `YYYY/MM/DD/*.jsonl` |
| Gemini | `~/.gemini/tmp/` | `<hash>/chats/session-*.json` |
| Kimi Code | `~/.kimi-code/sessions/` | `**/agents/*/wire.jsonl` |
| Oh My Pi | `~/.omp/agent/sessions/` | `**/*.jsonl` |

### Environment Variable Overrides

Override default config directories:

- `CLAUDE_CONFIG_DIR` (default: `~/.claude`)
- `CODEX_CONFIG_DIR` (default: `~/.codex`)
- `GEMINI_CONFIG_DIR` (default: `~/.gemini`)
- `KIMI_CONFIG_DIR` (default: `~/.kimi-code`)
- `OMP_CONFIG_DIR` (default: `~/.omp`)

## Display Modes

The table layout adapts to terminal width:

| Mode | Min Width | Description |
|------|-----------|-------------|
| Full | 205 cols | All columns with percentages |
| Medium | 137 cols | Shortened model names |
| Compact | 84 cols | Essential metrics only |
| Minimal | 70 cols | Basic summary |

## Subscription Fees

On first run, you will be prompted to enter your monthly subscription fees for each harness. These are saved to `.fee.env` and used to calculate monthly savings estimates.

In monitor mode, `cost` or `cost all` shows all current values, while `cost claude`, `cost codex`, `cost gemini`, or `cost kimi` shows one value. Set a fee with a command such as `cost claude 200` or `cost gemini 19.99`. The amount must be a non-negative integer or decimal. A valid change is written to the active `.fee.env` file before the dashboard updates; if the write fails, the in-memory value remains unchanged.

## Supported Models

Model display names and pricing are resolved **dynamically**, so newly released
models (e.g. a future `claude-opus-4-9`, `gpt-5.6`, `gemini-3.2-pro`) render and
price correctly with no upgrade required.

**Display names** are derived algorithmically from the model id:

- `claude-opus-4-8` -> `Opus 4.8`
- `gpt-5.5-codex` -> `GPT-5.5 Cdx`, `gpt-5.5:xhigh` -> `GPT-5.5(XH)`
- `gemini-3.2-pro-preview` -> `Gem 3.2 Pro`
- `k3` -> `K3`, `kimi-for-coding` -> `Kimi Coding`

**Pricing** is layered, newest wins:

1. Embedded `pricing.json` baseline.
2. Cached LiteLLM snapshot at `~/.cache/ai-usage/pricing-cache.json`.
3. Live LiteLLM price list (fetched at most once every 24h).

For a model not yet in any table, pricing falls back to the newest known model
of the same provider, family, and size class (so a brand-new `claude-opus-4-8`
is priced like the latest Opus, never like the vendor default), and `-mini` /
`-nano` variants never inherit a base model's rate.

### Model Overrides (`models.toml`)

To pin a display name or price for any model -- including private vendors and
the day-one gap before LiteLLM lists a new model -- create
`~/.config/ai-usage/models.toml` (honors `$XDG_CONFIG_HOME`). Overrides win over
both the algorithm and live pricing:

```toml
# Force a display name for any model id.
[display."anthropic/claude-opus-4-8"]
short = "Opus 4.8"

# Pin pricing in $/MTok; cache_read / cache_write default to the input rate.
[pricing."some-private-model"]
input = 3.0
output = 15.0
cache_read = 0.30
cache_write = 3.75
```

## Architecture

```
src/
  main.rs            # Entry point, CLI args, monitor loop
  constants.rs       # Pricing tables + class-aware lookup (pricing.json + .fee.env)
  pricing.rs         # Layered pricing loader (embedded -> cache -> LiteLLM)
  model_id.rs        # Algorithmic model-id parser (label / sort / color)
  model_overrides.rs # User overrides loader (~/.config/ai-usage/models.toml)
  formatting.rs      # Responsive table formatting
  charts.rs          # ASCII chart rendering
  time_utils.rs      # Timezone and time utilities
  data/
    mod.rs           # Common types (UsageEntry, TokenUsage)
    claude.rs        # Claude JSONL reader
    codex.rs         # Codex JSONL reader
    gemini.rs        # Gemini JSON reader
    kimi.rs          # Kimi Code wire JSONL reader
    omp.rs           # Oh My Pi JSONL reader
  stats/
    mod.rs           # Model breakdown and time series calculation
    claude.rs        # Claude statistics
    codex.rs         # Codex statistics
    gemini.rs        # Gemini statistics
    kimi.rs          # Kimi Code statistics
    omp.rs           # Oh My Pi statistics
pricing.json         # API pricing data
```

## Testing

```bash
# Run all tests (requires release build for the sync end-to-end tests)
cargo build --release && cargo test --workspace
```
