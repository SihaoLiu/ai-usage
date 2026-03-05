# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust utility for monitoring token usage and costs across multiple AI coding assistants: **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI**.

## Building and Running

```bash
# Build
cargo build --release

# All vendors, last 7 days (monitor mode)
./target/release/vibe-usage

# Single snapshot
./target/release/vibe-usage --once

# Last 30 days
./target/release/vibe-usage --days 30

# Specific vendor only
./target/release/vibe-usage --vendor claude
./target/release/vibe-usage --vendor codex
./target/release/vibe-usage --vendor gemini
```

## Data Sources

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
  constants.rs       # Pricing loader (pricing.json + .fee.env)
  formatting.rs      # Output formatting and responsive tables
  charts.rs          # ASCII chart visualization
  time_utils.rs      # Timezone and time formatting utilities
  data/
    mod.rs           # Common types (UsageEntry, TokenUsage)
    claude.rs        # Claude data reader (dedup by message ID)
    codex.rs         # Codex data reader (session_meta + token_count)
    gemini.rs        # Gemini data reader (JSON sessions)
  stats/
    mod.rs           # Generic model breakdown and time series calculation
    claude.rs        # Claude statistics wrappers
    codex.rs         # Codex statistics wrappers
    gemini.rs        # Gemini statistics wrappers
pricing.json         # API pricing data for all vendors
```

**Key implementation notes:**
- Token values are displayed in thousands (KTok) in charts
- Cache tokens (creation and read) are tracked separately
- Time series data is bucketed into 8-hour intervals for trend analysis
- All times are displayed in the system's local timezone
- Display adapts to terminal width (Full/Medium/Compact/Minimal modes)
- Monitor mode uses crossterm raw mode; disable before printing, re-enable after

## Testing

```bash
cargo build --release && cargo test
```

Snapshot tests compare Rust output against Python reference snapshots using fixture data in `tests/fixtures/`.

## Legacy Python

The original Python implementation is preserved in the `legacy/` directory for reference and snapshot comparison.
