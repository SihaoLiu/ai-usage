#!/usr/bin/env bash
# Performance benchmark script for vibe-usage
# Runs warm-cache measurements and checks against tiered targets.
#
# Targets:
#   --days < 10: < 0.5s wall time
#   --days 30:   < 2.0s wall time

set -euo pipefail

BINARY="${1:-./target/release/vibe-usage}"
RUNS=3

if [ ! -x "$BINARY" ]; then
    echo "Error: binary not found at $BINARY"
    echo "Build first: cargo build --release"
    exit 1
fi

# Measure wall time for a command, return seconds as float
measure() {
    local start end
    start=$(date +%s%N)
    COLUMNS=200 LINES=60 "$BINARY" --once "$@" > /dev/null 2>&1
    end=$(date +%s%N)
    echo "scale=3; ($end - $start) / 1000000000" | bc
}

pass_count=0
fail_count=0

run_bench() {
    local label="$1"
    local target="$2"
    shift 2
    local args=("$@")

    echo "=== $label (target: < ${target}s) ==="
    local max_time=0
    for i in $(seq 1 $RUNS); do
        local t
        t=$(measure "${args[@]}")
        echo "  Run $i: ${t}s"
        if [ "$(echo "$t > $max_time" | bc -l)" -eq 1 ]; then
            max_time=$t
        fi
    done

    if [ "$(echo "$max_time < $target" | bc -l)" -eq 1 ]; then
        echo "  PASS (max: ${max_time}s < ${target}s)"
        pass_count=$((pass_count + 1))
    else
        echo "  FAIL (max: ${max_time}s >= ${target}s)"
        fail_count=$((fail_count + 1))
    fi
    echo
}

echo "Benchmark: vibe-usage performance"
echo "Binary: $BINARY"
echo "Runs per config: $RUNS"
echo

run_bench "3-day all vendors" 0.5 --days 3
run_bench "7-day all vendors" 0.5 --days 7
run_bench "3-day claude only" 0.5 --days 3 --vendor claude
run_bench "30-day all vendors" 2.0 --days 30

echo "================================"
echo "Results: $pass_count passed, $fail_count failed"

if [ "$fail_count" -gt 0 ]; then
    exit 1
fi
