#!/usr/bin/env bash
# Long training run followed by deterministic full val + test evaluation.
#
# LR defaults to the best full_val_bpc for $PRESET in results/lr_sweep.csv.
# WAIT_PID makes the script wait until another process (e.g. the sweep) exits.
#
# Usage:
#   WAIT_PID=12345 setsid nohup scripts/long_run.sh >> results/long_run.log 2>&1 < /dev/null &
#   PRESET=xlarge BATCH=2 ACCUM=8 LR=4e-4 scripts/long_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET=${PRESET:-large}
ARCH=${ARCH:-llama}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
WARMUP=${WARMUP:-400}
LR_MIN=${LR_MIN:-1e-5}
FALLBACK_LR=${FALLBACK_LR:-5e-4}

if [ -n "${WAIT_PID:-}" ]; then
    [[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { echo "[!] WAIT_PID musi być pojedynczym PID, jest: '$WAIT_PID'"; exit 1; }
    kill -0 "$WAIT_PID" 2>/dev/null || { echo "[!] PID $WAIT_PID nie istnieje"; exit 1; }
    echo "[$(date '+%F %T')] czekam na zakończenie PID $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

if [ -z "${LR:-}" ]; then
    LR=$(awk -F, -v p="$PRESET" -v a="$ARCH" \
        'NR > 1 && $1 == p && $2 == a && $8 != "" { if (best == "" || $8 < best) { best = $8; lr = $3 } }
         END { print lr }' results/lr_sweep.csv 2>/dev/null || true)
    if [ -z "$LR" ]; then
        echo "[!] brak wyników sweepu dla $PRESET/$ARCH, używam LR=$FALLBACK_LR"
        LR=$FALLBACK_LR
    else
        echo "[+] LR z sweepu (najlepsze full_val_bpc): $LR"
    fi
fi

TAG="long_${PRESET}_${ARCH}_${STEPS}_lr${LR}"
echo "[$(date '+%F %T')] start $TAG (batch ${BATCH}x${ACCUM}, warmup $WARMUP)"
cargo build --release --features cuda --bins

./target/release/thesis-compressor --preset "$PRESET" --arch "$ARCH" --steps "$STEPS" \
    --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lr-max "$LR" --lr-min "$LR_MIN" --warmup-steps "$WARMUP" \
    --eval-every 1000 --eval-batches 20 --log-every 100 \
    --csv-path "results/$TAG.csv" \
    --last-checkpoint "checkpoints/${TAG}_last.safetensors" \
    --best-checkpoint "checkpoints/${TAG}_best.safetensors" \
    > "results/$TAG.log" 2>&1

for split in val test; do
    ./target/release/eval --checkpoint "checkpoints/${TAG}_last.safetensors" \
        --preset "$PRESET" --arch "$ARCH" --split "$split" --batch-size 8 \
        --csv results/eval_summary.csv >> "results/$TAG.log" 2>&1
    echo "[+] $split BPC: $(tail -1 results/eval_summary.csv | cut -d, -f5)"
done
echo "[$(date '+%F %T')] koniec $TAG"
