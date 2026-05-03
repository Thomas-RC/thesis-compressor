#!/usr/bin/env bash
# Wait until any running thesis-compressor training finishes, then run
# deterministic non-overlapping BPC evaluation on test (and optionally val/train).
#
# Usage:
#   scripts/eval_after_run.sh <checkpoint> [preset] [--include-train]
#
# Examples:
#   scripts/eval_after_run.sh checkpoints/large_40k_b16_best.safetensors large
#   scripts/eval_after_run.sh checkpoints/xlarge_best.safetensors xlarge --include-train

set -euo pipefail

CKPT="${1:?missing checkpoint path}"
PRESET="${2:-large}"
INCLUDE_TRAIN="no"
[[ "${3:-}" == "--include-train" ]] && INCLUDE_TRAIN="yes"

EVAL_BIN="./target/release/eval"
CSV="results/eval_summary.csv"

cd "$(dirname "$0")/.."

if [[ ! -x "$EVAL_BIN" ]]; then
    echo "[!] $EVAL_BIN not found — build it first: cargo build --release --features cuda --bin eval" >&2
    exit 1
fi

echo "[+] waiting for any running thesis-compressor training to finish..."
while pgrep -x thesis-compressor > /dev/null; do
    sleep 30
done
echo "[+] training process not running"

if [[ ! -f "$CKPT" ]]; then
    echo "[!] checkpoint not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$CSV")"

SPLITS=(test val)
[[ "$INCLUDE_TRAIN" == "yes" ]] && SPLITS+=(train)

for split in "${SPLITS[@]}"; do
    echo ""
    echo "=== eval --split $split ==="
    "$EVAL_BIN" \
        --checkpoint "$CKPT" \
        --preset "$PRESET" \
        --split "$split" \
        --csv "$CSV"
done

echo ""
echo "[+] done. summary:"
cat "$CSV"
