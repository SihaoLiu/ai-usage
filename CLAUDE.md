# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a single Python utility (`claude-usage.py`) that analyzes Claude Code usage statistics by parsing JSONL files from the `~/.claude/projects` directory. It generates comprehensive reports including token usage statistics, model breakdowns, and time-series visualizations.

## Running the Script

```bash
# No additional dependencies required - uses only Python standard library
python3 claude-usage.py
# or
./claude-usage.py
```

The script reads from `~/.claude/projects` by default, or from the path specified in the `CLAUDE_CONFIG_DIR` environment variable.

## Architecture

**Single-file utility structure:**
- Data parsing functions (`read_jsonl_files`, `get_claude_dir`) - Read JSONL logs from Claude projects
- Statistics calculation (`calculate_overall_stats`, `calculate_model_breakdown`, `calculate_time_series`) - Aggregate token usage data
- Visualization functions (`print_line_chart`, `print_model_chart`) - ASCII-based charts for 8-hour interval analysis
- All times are displayed in the system's local timezone

**Data flow:**
1. Scans `~/.claude/projects/**/*.jsonl` for usage data
2. Filters entries containing message.usage fields
3. Aggregates by model and time (8-hour intervals)
4. Outputs formatted tables and ASCII charts

**Key implementation notes:**
- Token values are displayed in thousands (KTok) in charts
- Cache tokens (creation and read) are tracked separately
- Time series data is bucketed into 8-hour intervals for trend analysis
- Model-specific symbols used in charts: █ (Sonnet 4.5), ▓ (Haiku 4.5), ▒ (Opus 4.1)
