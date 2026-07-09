#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/overfire_results"; mkdir -p "$OUT"
declare -A Q; idx=0
for m in abmil rep; do for s in 0 1 2; do
  gpu=$(( idx % 8 ))
  Q[$gpu]="${Q[$gpu]:-}CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/mil_abmil.py --mode $m --task majority --steps 2500 --seed $s > $OUT/majority_${m}_s${s}.json 2> $OUT/majority_${m}_s${s}.err ; "
  idx=$(( idx+1 ))
done; done
for g in "${!Q[@]}"; do ( eval "${Q[$g]}" ; echo "gpu $g done" ) & done
echo "launched $idx"; wait; echo "OVERFIRE_DONE"
