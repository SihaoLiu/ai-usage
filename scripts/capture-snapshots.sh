#!/usr/bin/env bash
# Capture reference snapshots from both Python and Rust for comparison.
# Uses fixture data with controlled env vars and terminal sizes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURES="$ROOT_DIR/tests/fixtures"
SNAPSHOTS="$ROOT_DIR/tests/snapshots"
RUST_BIN="${1:-$ROOT_DIR/target/release/vibe-usage}"

mkdir -p "$SNAPSHOTS/rust" "$SNAPSHOTS/python"

export CLAUDE_CONFIG_DIR="$FIXTURES/claude"
export CODEX_CONFIG_DIR="$FIXTURES/codex"
export GEMINI_CONFIG_DIR="$FIXTURES/gemini"

# Touch fixture files so mtime filter passes
find "$FIXTURES" -type f \( -name '*.jsonl' -o -name '*.json' \) -exec touch {} +

capture() {
    local name="$1" cols="$2" lines="$3"
    shift 3
    local args=("$@")

    echo "Capturing: $name (${cols}x${lines})"
    COLUMNS="$cols" LINES="$lines" "$RUST_BIN" --once "${args[@]}" \
        > "$SNAPSHOTS/rust/${name}.txt" 2>&1 || true
    COLUMNS="$cols" LINES="$lines" python3 "$ROOT_DIR/vibe-usage.py" --once "${args[@]}" \
        > "$SNAPSHOTS/python/${name}.txt" 2>&1 || true
}

# Full mode (200 columns)
capture "all-full"      200 60 --days 3
capture "claude-full"   200 60 --days 3 --vendor claude
capture "codex-full"    200 60 --days 3 --vendor codex
capture "gemini-full"   200 60 --days 3 --vendor gemini

# Compact mode (100 columns)
capture "all-compact"    100 40 --days 3
capture "claude-compact" 100 40 --days 3 --vendor claude

echo
echo "Snapshots saved to $SNAPSHOTS/"
echo "Compare: diff $SNAPSHOTS/python/ $SNAPSHOTS/rust/"
