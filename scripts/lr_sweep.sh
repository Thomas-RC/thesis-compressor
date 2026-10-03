#!/usr/bin/env bash
# Learning-rate sweep: short runs (warmup + stable + 20% linear decay) per preset,
# each followed by deterministic full-split validation BPC (bin/eval).
#
# Usage:
#   scripts/lr_sweep.sh                         # medium + large, default LR grids
#   PRESETS="xlarge" LRS_xlarge="3e-4 6e-4" scripts/lr_sweep.sh
#
# Output:
#   results/lr_sweep.csv                       one summary row per run
#   results/lr_sweep/<preset>_<arch>_lr<lr>.{csv,log}, checkpoints/lr_sweep/...
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH=${ARCH:-llama}
STEPS=${STEPS:-2000}
WARMUP=${WARMUP:-100}
LR_MIN=${LR_MIN:-1e-5}
PRESETS=${PRESETS:-"medium large"}

# Effective batch = 16k tokens/step for every preset.
BATCH_medium=32; ACCUM_medium=1
BATCH_large=4;   ACCUM_large=4
BATCH_xlarge=2;  ACCUM_xlarge=8
LRS_medium=${LRS_medium:-"1e-3 2e-3 4e-3 8e-3"}
LRS_large=${LRS_large:-"5e-4 1e-3 2e-3 4e-3"}
LRS_xlarge=${LRS_xlarge:-"3e-4 6e-4 1.2e-3 2.4e-3"}

BIN=./target/release/thesis-compressor
EVAL=./target/release/eval
SUMMARY=results/lr_sweep.csv
mkdir -p results/lr_sweep checkpoints/lr_sweep

cargo build --release --features cuda --bins
[ -f "$SUMMARY" ] || echo "preset,arch,lr_max,steps,eff_batch,tokens,sampled_val_bpc,full_val_bpc,train_wall_s" > "$SUMMARY"

for preset in $PRESETS; do
    bvar=BATCH_$preset; avar=ACCUM_$preset; lvar=LRS_$preset
    batch=${!bvar}; accum=${!avar}; lrs=${!lvar}
    for lr in $lrs; do
        tag="${preset}_${ARCH}_lr${lr}"
        echo "=== $tag (batch ${batch}x${accum}, $STEPS steps) ==="
        $BIN --preset "$preset" --arch "$ARCH" --steps "$STEPS" \
            --batch-size "$batch" --grad-accum "$accum" \
            --lr-max "$lr" --lr-min "$LR_MIN" --warmup-steps "$WARMUP" \
            --eval-every 250 --eval-batches 10 --log-every 50 \
            --csv-path "results/lr_sweep/$tag.csv" \
            --last-checkpoint "checkpoints/lr_sweep/${tag}_last.safetensors" \
            --best-checkpoint "checkpoints/lr_sweep/${tag}_best.safetensors" \
            > "results/lr_sweep/$tag.log" 2>&1 || { echo "    FAILED (see log)"; continue; }

        final=$(grep '^[0-9]*,final,' "results/lr_sweep/$tag.csv" | cut -d, -f4)
        wall=$(grep '^[0-9]*,final,' "results/lr_sweep/$tag.csv" | cut -d, -f6)
        rm -f "results/lr_sweep/${tag}_eval.csv"
        $EVAL --checkpoint "checkpoints/lr_sweep/${tag}_last.safetensors" \
            --preset "$preset" --arch "$ARCH" --split val --batch-size 8 \
            --csv "results/lr_sweep/${tag}_eval.csv" >> "results/lr_sweep/$tag.log" 2>&1
        full=$(tail -1 "results/lr_sweep/${tag}_eval.csv" | cut -d, -f5)
        seq_len=$(grep -oE 'seq_len=[0-9]+' "results/lr_sweep/$tag.log" | head -1 | cut -d= -f2)
        tokens=$((STEPS * batch * accum * seq_len))
        echo "$preset,$ARCH,$lr,$STEPS,$((batch * accum)),$tokens,$final,$full,$wall" >> "$SUMMARY"
        echo "    sampled val=$final | full val=$full | ${wall}s"
    done
done
