#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/novelty_results"; mkdir -p "$OUT"
MODES=(abmil dedup tome dpp rep)
SEEDS=(0 1 2)
declare -A Q; idx=0
for m in "${MODES[@]}"; do
  for s in "${SEEDS[@]}"; do
    gpu=$(( idx % 8 )); name="redundancy_${m}_s${s}"
    Q[$gpu]="${Q[$gpu]:-}CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/mil_abmil.py --mode $m --task redundancy --N 32 --steps 2500 --seed $s > $OUT/$name.json 2> $OUT/$name.err ; "
    idx=$(( idx + 1 ))
  done
done
for g in "${!Q[@]}"; do ( eval "${Q[$g]}" ; echo "gpu $g done" ) & done
echo "launched $idx runs"; wait; echo "NOVELTY_GRID_DONE"
