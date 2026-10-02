#!/usr/bin/env bash
# Sequential single-component Llama ablations on xlarge preset @20k b=8.
# Each run: training (~37 min) + test eval + val eval. Total ~2 h on RTX 4080S.
#
# Architectures (each is Baseline + ONE Llama-style change):
#   llama-rms     — only LayerNorm replaced by RmsNorm
#   llama-rope    — only learned absolute pos replaced by RoPE
#   llama-swiglu  — only GELU MLP replaced by SwiGLU (Llama hidden = ⌈8d/3⌉₃₂)
#
# Outputs append to results/eval_summary.csv for direct comparison with
# previously-trained Baseline (3.987) and full Llama (3.920) at same setup.

set -euo pipefail
cd "$(dirname "$0")/.."

# Wait for any other thesis-compressor training/eval to finish first.
echo "=== $(date '+%F %T') waiting for any active thesis-compressor process ==="
while pgrep -f 'target/release/thesis-compressor' > /dev/null \
   || pgrep -f 'target/release/eval' > /dev/null; do
    sleep 15
done
echo "=== $(date '+%F %T') GPU free, starting ablations ==="

ARCHES=("llama-rms" "llama-rope" "llama-swiglu")

for ARCH in "${ARCHES[@]}"; do
    NAME=${ARCH//-/_}
    LOG=results/long_xlarge_${NAME}_20k_b8.log
    CSV=results/long_xlarge_${NAME}_20k_b8.csv
    BEST=checkpoints/xlarge_${NAME}_20k_b8_best.safetensors
    LAST=checkpoints/xlarge_${NAME}_20k_b8_last.safetensors

    echo "=== $(date '+%F %T') TRAIN arch=${ARCH} ==="
    ./target/release/thesis-compressor \
        --arch "$ARCH" --preset xlarge \
        --steps 20000 --batch-size 8 --warmup-steps 200 \
        --eval-every 200 --log-every 100 \
        --csv-path "$CSV" \
        --best-checkpoint "$BEST" \
        --last-checkpoint "$LAST" \
        > "$LOG" 2>&1

    echo "=== $(date '+%F %T') EVAL test arch=${ARCH} ==="
    ./target/release/eval \
        --arch "$ARCH" --preset xlarge --batch-size 8 \
        --checkpoint "$BEST" --split test \
        --csv results/eval_summary.csv

    echo "=== $(date '+%F %T') EVAL val arch=${ARCH} ==="
    ./target/release/eval \
        --arch "$ARCH" --preset xlarge --batch-size 8 \
        --checkpoint "$BEST" --split val \
        --csv results/eval_summary.csv
done

echo "=== $(date '+%F %T') ALL ABLATIONS DONE ==="
cat results/eval_summary.csv
