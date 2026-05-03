#!/usr/bin/env bash
# Wait until any running thesis-compressor training finishes, then run
# deterministic non-overlapping BPC evaluation on test (and optionally val/train).
#
# Usage:
#   scripts/eval_after_run.sh <checkpoint> [preset] [--include-train]
#
# Env vars:
#   EVAL_BATCH=N — eval batch size (default 8, safe for all presets on 16 GB).
#                  For 'large' you can set EVAL_BATCH=16 to halve eval wall time.
#                  'xlarge' MUST stay at 8 (b=16 OOMs on 16 GB).
#
# Examples:
#   scripts/eval_after_run.sh checkpoints/large_40k_b16_best.safetensors large
#   EVAL_BATCH=16 scripts/eval_after_run.sh checkpoints/large_best.safetensors large
#   scripts/eval_after_run.sh checkpoints/xlarge_best.safetensors xlarge --include-train

set -euo pipefail

CKPT="${1:?missing checkpoint path}"
PRESET="${2:-large}"
INCLUDE_TRAIN="no"
[[ "${3:-}" == "--include-train" ]] && INCLUDE_TRAIN="yes"

EVAL_BATCH="${EVAL_BATCH:-8}"
EVAL_BIN="./target/release/eval"
CSV="results/eval_summary.csv"

cd "$(dirname "$0")/.."

if [[ ! -x "$EVAL_BIN" ]]; then
    echo "[!] $EVAL_BIN not found — build it first: cargo build --release --features cuda --bin eval" >&2
    exit 1
fi

echo "[+] waiting for any running thesis-compressor training to finish..."
# -f matches full command line (the binary name 'thesis-compressor' exceeds
# pgrep's default 15-char comm field, so -x silently misses it).
while pgrep -f 'target/release/thesis-compressor' > /dev/null; do
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
        --batch-size "$EVAL_BATCH" \
        --split "$split" \
        --csv "$CSV"
done

echo ""
echo "[+] done. summary:"
cat "$CSV"
