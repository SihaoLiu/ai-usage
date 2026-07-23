# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust utility for monitoring token usage and costs across multiple AI coding assistants: **Claude Code**, **OpenAI Codex**, **Google Gemini CLI**, **Kimi Code**, and **Oh My Pi**.

## Building and Running

```bash
# Build
cargo build --release

# All tools, last 3 days (monitor mode)
./target/release/ai-usage

# Single snapshot
./target/release/ai-usage --once

# Last 30 days
./target/release/ai-usage --days 30

# Specific tool only
./target/release/ai-usage --tool claude
./target/release/ai-usage --tool codex
./target/release/ai-usage --tool gemini
./target/release/ai-usage --tool kimi
./target/release/ai-usage --tool omp

# Breakdown table shape (also cycled with `v` in monitor mode)
./target/release/ai-usage --view flat    # Vendor / Model / Harness columns
./target/release/ai-usage --view vendor  # grouped by vendor with subtotals
./target/release/ai-usage --view model   # one row per model, harnesses merged
```

## Data Sources

| Vendor | Directory | File Pattern |
|--------|-----------|--------------|
| Claude | `~/.claude/projects/` | `**/*.jsonl` |
| Codex | `~/.codex/sessions/` | `YYYY/MM/DD/*.jsonl` |
| Gemini | `~/.gemini/tmp/` | `<hash>/chats/session-*.json` |
| Kimi Code | `~/.kimi-code/sessions/` | `**/agents/*/wire.jsonl` |
| Oh My Pi | `~/.omp/agent/sessions/` | `**/*.jsonl` |

Environment variables can override default paths:
- `CLAUDE_CONFIG_DIR` (default: `~/.claude`)
- `CODEX_CONFIG_DIR` (default: `~/.codex`)
- `GEMINI_CONFIG_DIR` (default: `~/.gemini`)
- `KIMI_CONFIG_DIR` (default: `~/.kimi-code`)
- `OMP_CONFIG_DIR` (default: `~/.omp`)

## Architecture

```
src/
  main.rs            # Entry point, CLI args, once-mode printing, AppState
  tool.rs            # Tool enum = the Harness axis (keys, names, rotation)
  model_id.rs        # Algorithmic model-id parser; Vendor enum = model maker
                     # (Anthropic/OpenAI/Google/Moonshot/Zhipu) inferred from id
  table_view.rs      # Renderer-agnostic breakdown table views: flat / vendor /
                     # model (three shapes over the Vendor x Harness x Model axes)
  raw_data.rs        # Raw entry cache, host filtering, cross-tool aggregation
  window_nav.rs      # Time-window navigation and chart interval sizing
  sync_status.rs     # Sync worker polling and status-line formatting
  constants.rs       # Pricing tables, class-aware lookup (pricing.json + .fee.env)
  pricing.rs         # Layered pricing loader (embedded -> cache -> LiteLLM)
  model_overrides.rs # User overrides loader (~/.config/ai-usage/models.toml)
  formatting.rs      # Plain-text tables for --once output
  charts.rs          # Plain ASCII charts for --once output + shared series specs
  time_utils.rs      # Timezone and time formatting utilities
  updater.rs         # Self-update from GitHub releases
  tui/
    mod.rs           # Ratatui monitor mode: terminal lifecycle + event loop
    commands.rs      # Prompt command parsing/execution (testable, no terminal)
    data.rs          # Dashboard assembly (table rows, cost summary, chart series)
    input.rs         # Prompt input line and command history
    render.rs        # Frame rendering: header tabs, table, charts, footer, help
  data/
    mod.rs           # Common types (UsageEntry, TokenUsage)
    cache.rs         # Normalized per-vendor record cache
    claude.rs        # Claude data reader (dedup by message ID)
    codex.rs         # Codex data reader (session_meta + token_count)
    gemini.rs        # Gemini data reader (JSON sessions)
    kimi.rs          # Kimi Code data reader (wire.jsonl usage.record events)
    omp.rs           # Oh My Pi data reader (message usage lines)
  stats/
    mod.rs           # Generic model breakdown and time series calculation
    claude.rs        # Claude statistics wrappers
    codex.rs         # Codex statistics wrappers
    gemini.rs        # Gemini statistics wrappers
    kimi.rs          # Kimi statistics wrappers
    omp.rs           # Oh My Pi statistics wrappers
  sync/              # Optional cross-machine sync client (see docs/sync.md)
crates/
  ai-usage-proto     # Shared wire types and validation for sync
  ai-usage-server    # Sync server binary
pricing.json         # API pricing data for all vendors
```

**Key implementation notes:**
- Model display names are derived algorithmically from the id (`model_id.rs`); no per-model table to maintain. New models render and price automatically; unknown-but-priced models fall back to the newest same-class model, and `models.toml` overrides win over everything.
- Token values are displayed in thousands (KTok) in charts
- Cache tokens (creation and read) are tracked separately
- Time series data is bucketed into 8-hour intervals for trend analysis
- All times are displayed in the system's local timezone
- Display adapts to terminal width (Full/Medium/Compact/Minimal modes)
- The display model has three axes: Vendor (model maker, derived from the model
  id by `model_id::Vendor`), Harness (the CLI tool, `tool::Tool`), and Model.
  `table_view.rs` is the single source of truth for grouping/merging them.
  Note: the data/sync layer historically says "vendor" where it means harness
  (`data/<vendor>.rs`, `is_valid_vendor`); do not confuse the two.
- Monitor mode is a ratatui alternate-screen app (`src/tui/`); `--once` prints
  plain text via `formatting.rs`/`charts.rs` and stays pipe-friendly
- Adding a harness touches: `data/<vendor>.rs`, `stats/<vendor>.rs`, `tool.rs`, `main.rs` wiring, `pricing.json` + `constants.rs`/`pricing.rs`, `sync/engine/mod.rs`, and `ai-usage-proto::is_valid_vendor`

## Testing

```bash
cargo build --release && cargo test --workspace
```

Unit tests live in-module next to the code they cover; `tests/sync_e2e.rs` runs the built CLI against a local sync server.
