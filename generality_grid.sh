#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/generality_results"; mkdir -p "$OUT"
DIMS=(32 64 128 256); MODES=(mha rep_key rep_val); SEEDS=(0 1 2)
declare -A Q; idx=0
for d in "${DIMS[@]}"; do for m in "${MODES[@]}"; do for s in "${SEEDS[@]}"; do
  gpu=$(( idx % 8 )); name="d${d}_${m}_s${s}"
  Q[$gpu]="${Q[$gpu]:-}CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/f2_sweep.py --task redundancy --f2 $m --dim $d --heads 4 --steps 3000 --seed $s > $OUT/$name.json 2> $OUT/$name.err ; "
  idx=$(( idx+1 ))
done; done; done
for g in "${!Q[@]}"; do ( eval "${Q[$g]}" ; echo "gpu $g done" ) & done
echo "launched $idx runs (4 dims x 3 modes x 3 seeds)"; wait; echo "GENERALITY_DONE"
