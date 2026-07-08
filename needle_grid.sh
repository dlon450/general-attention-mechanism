#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/needle_results"
mkdir -p "$OUT"

STEPS="${STEPS:-3000}"
BS="${BS:-128}"
COMMON=(--K 1 --noise 1.2 --dim 64 --heads 4 --redundant --steps "$STEPS" --batch-size "$BS" --lr 5e-4 --device cuda)

LS=(128 512 2048)
ARMS=(mha gated gated_rep)
SEEDS=(0 1 2)

declare -A GPUCMDS
idx=0
for L in "${LS[@]}"; do
  for a in "${ARMS[@]}"; do
    for s in "${SEEDS[@]}"; do
      gpu=$(( idx % 8 ))
      name="${a}_L${L}_s${s}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/synthetic_needle.py --attn $a --L $L --seed $s ${COMMON[*]} > $OUT/$name.json 2> $OUT/$name.err"
      GPUCMDS[$gpu]="${GPUCMDS[$gpu]:-}${cmd} ; "
      idx=$(( idx + 1 ))
    done
  done
done

for gpu in "${!GPUCMDS[@]}"; do
  ( eval "${GPUCMDS[$gpu]}" ; echo "gpu $gpu done" ) &
done
echo "launched $idx runs across 8 GPUs; waiting..."
wait
echo "NEEDLE_GRID_DONE"
