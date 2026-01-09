# AI Usage Monitor

A Python utility for monitoring token usage and costs across multiple AI coding assistants: **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI**.

## Features

- **Multi-vendor support**: Track usage from Claude, Codex, and Gemini in one place
- **Cost analysis**: Calculate costs based on current API pricing with daily/weekly/monthly projections
- **Token breakdown**: Input, output, cache read, and cache creation tokens tracked separately
- **Visualizations**: ASCII-based multi-line charts and stacked bar charts
- **Responsive UI**: Adapts to terminal width with 4 display modes
- **Monitor mode**: Auto-refresh for continuous tracking
- **Zero dependencies**: Uses only Python standard library

## Quick Start

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

## Output

The tool generates:

1. **Usage statistics table** - Per-model breakdown with token counts and costs
2. **Time series charts** - Input/output tokens over time by model
3. **Cache usage charts** - Cache read/creation patterns
4. **Vendor comparison** - Side-by-side usage across vendors
5. **Cost summary** - Daily, weekly, and monthly projections

Display modes adapt to terminal width:
- **Full** (≥205 chars): Complete details with percentages
- **Medium** (≥137 chars): Shortened model names
- **Compact** (≥84 chars): Essential metrics only
- **Minimal** (≥70 chars): Basic info

## Supported Models

**Claude**: Opus 4.5, Opus 4.1, Sonnet 4.5, Sonnet 4, Haiku 4.5

**Codex/OpenAI**: GPT-5, GPT-5.1, GPT-4.1, o1, o3, o3-mini, o4-mini

**Gemini**: Pro Preview, 2.5 Pro/Flash/Flash-Lite, 2.0 Flash/Flash-Lite

## License

MIT License - see [LICENSE](LICENSE) file.
