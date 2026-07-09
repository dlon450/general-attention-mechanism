#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/variance_results"; mkdir -p "$OUT"
# name|extra flags
VARIANTS=("learned_mf1|" "fixed5_mf1|--fixed-lambda 5.0" "fixed5_mf3|--fixed-lambda 5.0 --mf-iters 3" "learned_mf3|--mf-iters 3")
SEEDS=(0 1 2 3 4)
declare -A Q; idx=0
for spec in "${VARIANTS[@]}"; do
  name="${spec%%|*}"; flags="${spec#*|}"
  for s in "${SEEDS[@]}"; do
    gpu=$(( idx % 8 ))
    Q[$gpu]="${Q[$gpu]:-}CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/mil_abmil.py --mode rep --task redundancy --N 32 --steps 2500 --seed $s $flags > $OUT/${name}_s${s}.json 2> $OUT/${name}_s${s}.err ; "
    idx=$(( idx + 1 ))
  done
done
for g in "${!Q[@]}"; do ( eval "${Q[$g]}" ; echo "gpu $g done" ) & done
echo "launched $idx runs"; wait; echo "VARIANCE_GRID_DONE"
