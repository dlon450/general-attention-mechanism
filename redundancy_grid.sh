#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/redundancy_results"; mkdir -p "$OUT"

STEPS="${STEPS:-4000}"
COMMON=(--max-sig 5 --min-count 2 --n-bg 30 --noise-sig 0.3 --dim 64 --heads 4 --steps "$STEPS" --batch-size 256 --lr 5e-4 --device cuda)

MAXDEC=(15 30 60)          # redundancy strength (clique size)
ARMS=(mha gated gated_rep)
SEEDS=(0 1 2 3 4)

declare -A GPUCMDS
idx=0
for md in "${MAXDEC[@]}"; do
  for a in "${ARMS[@]}"; do
    for s in "${SEEDS[@]}"; do
      gpu=$(( idx % 8 )); name="${a}_md${md}_s${s}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/synthetic_redundancy.py --attn $a --max-dec $md --seed $s ${COMMON[*]} > $OUT/$name.json 2> $OUT/$name.err"
      GPUCMDS[$gpu]="${GPUCMDS[$gpu]:-}${cmd} ; "
      idx=$(( idx + 1 ))
    done
  done
done
for gpu in "${!GPUCMDS[@]}"; do ( eval "${GPUCMDS[$gpu]}" ; echo "gpu $gpu done" ) & done
echo "launched $idx runs across 8 GPUs; waiting..."
wait
echo "REDUNDANCY_GRID_DONE"
