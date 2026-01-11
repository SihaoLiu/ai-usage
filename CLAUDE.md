# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python utility for monitoring token usage and costs across multiple AI coding assistants: **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI**. Uses only Python standard library (zero dependencies).

## Running the Script

```bash
# All vendors, last 7 days
python3 vibe-usage.py

# Last 30 days
python3 vibe-usage.py --days 30

# Specific vendor only
python3 vibe-usage.py --vendor claude
python3 vibe-usage.py --vendor codex
python3 vibe-usage.py --vendor gemini

# Monitor mode (auto-refresh every hour)
python3 vibe-usage.py --monitor
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
vibe-usage.py          # Main entry point with CLI argument parsing
├── data.py            # Claude data reader
├── data_codex.py      # Codex data reader
├── data_gemini.py     # Gemini data reader
├── stats.py           # Claude statistics calculation
├── stats_codex.py     # Codex statistics calculation
├── stats_gemini.py    # Gemini statistics calculation
├── formatting.py      # Output formatting and responsive tables
├── charts.py          # ASCII chart visualization
├── constants.py       # Pricing configuration loader
└── pricing.json       # API pricing data for all vendors
```

**Key implementation notes:**
- Token values are displayed in thousands (KTok) in charts
- Cache tokens (creation and read) are tracked separately
- Time series data is bucketed into 8-hour intervals for trend analysis
- All times are displayed in the system's local timezone
- Display adapts to terminal width (Full/Medium/Compact/Minimal modes)
