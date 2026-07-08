#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/f2_needle_power"; mkdir -p "$OUT"
STEPS="${STEPS:-4000}"
F2S=(mha modular rep_key rep_val)
SEEDS=(0 1 2 3 4 5 6 7)
LS=(512 1024)
declare -A GPUCMDS; idx=0
for L in "${LS[@]}"; do
  for f in "${F2S[@]}"; do
    for s in "${SEEDS[@]}"; do
      gpu=$(( idx % 8 )); name="needle_L${L}_${f}_s${s}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/f2_sweep.py --task needle --f2 $f --L $L --steps $STEPS --seed $s > $OUT/$name.json 2> $OUT/$name.err"
      GPUCMDS[$gpu]="${GPUCMDS[$gpu]:-}${cmd} ; "
      idx=$(( idx + 1 ))
    done
  done
done
for gpu in "${!GPUCMDS[@]}"; do ( eval "${GPUCMDS[$gpu]}" ; echo "gpu $gpu done" ) & done
echo "launched $idx runs across 8 GPUs; waiting..."
wait
echo "F2_NEEDLE_POWER_DONE"
