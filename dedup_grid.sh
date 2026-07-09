#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/dedup_results"; mkdir -p "$OUT"
MODES=(abmil aem sparsemax entmax15 dedup countnorm rep)
SEEDS=(0 1 2 3 4)
declare -A Q; idx=0
for m in "${MODES[@]}"; do
  for s in "${SEEDS[@]}"; do
    gpu=$(( idx % 8 )); name="redundancy_${m}_s${s}"
    Q[$gpu]="${Q[$gpu]:-}CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/mil_abmil.py --mode $m --task redundancy --N 32 --steps 2500 --seed $s --aem-coef 0.02 > $OUT/$name.json 2> $OUT/$name.err ; "
    idx=$(( idx + 1 ))
  done
done
for g in "${!Q[@]}"; do ( eval "${Q[$g]}" ; echo "gpu $g done" ) & done
echo "launched $idx runs"; wait; echo "DEDUP_GRID_DONE"
