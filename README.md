# AI Usage Monitor

A fast terminal dashboard for tracking token usage and costs across **Claude Code**, **OpenAI Codex**, **Google Gemini CLI**, and **Oh My Pi**. Written in Rust.

<img src="docs/assets/ai-usage.gif" alt="AI Usage Monitor demo" width="100%">

## Features

- Multi-vendor usage tracking in one place
- Per-model token breakdown with cost analysis and daily/weekly/monthly projections
- ASCII time series charts with ANSI colors
- Responsive display adapting to terminal width
- Live monitor mode with interactive keyboard controls

## Install

### Option 1: Download binary

Download the pre-built binary for your platform from [Releases](https://github.com/SihaoLiu/ai-usage/releases):

| Platform | Asset |
|----------|-------|
| Linux x86_64 (musl, fully static) | `vibe-usage-x86_64-linux-musl` |
| Linux aarch64 (musl, fully static) | `vibe-usage-aarch64-linux-musl` |
| Linux x86_64 (glibc) | `vibe-usage-x86_64-linux-gnu` |
| Linux aarch64 (glibc) | `vibe-usage-aarch64-linux-gnu` |
| macOS Apple Silicon | `vibe-usage-aarch64-apple-darwin` |

Place it anywhere in your `$PATH` and `chmod +x` it. Once installed, you can keep it up to date by typing `update` in monitor mode (it downloads the matching asset from the latest release and re-execs in place).

### Option 2: Build from source

```bash
# Standard build
cargo build --release

# Static binary (portable across x86_64 Linux)
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Quick Start

```bash
# All vendors, last 3 days (monitor mode)
vibe-usage

# Single snapshot, then exit
vibe-usage --once

# Last 30 days, specific vendor
vibe-usage --days 30 --vendor claude
```

See [docs/usage.md](docs/usage.md) for full usage details, data sources, monitor mode controls, and configuration. For optional cross-machine sync, see [docs/sync.md](docs/sync.md).

## License

MIT License - see [LICENSE](LICENSE) file.
