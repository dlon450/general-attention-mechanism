#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/mil_results"; mkdir -p "$OUT"
STEPS="${STEPS:-2500}"; N="${N:-32}"
F2S=(mha sparsemax entmax15 modular rep_key rep_val)
SEEDS=(0 1 2)
declare -A GPUCMDS; idx=0
for task in needle redundancy; do
  for f in "${F2S[@]}"; do
    for s in "${SEEDS[@]}"; do
      gpu=$(( idx % 8 )); name="${task}_${f}_s${s}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/mil_mnist.py --task $task --attn $f --N $N --steps $STEPS --seed $s > $OUT/$name.json 2> $OUT/$name.err"
      GPUCMDS[$gpu]="${GPUCMDS[$gpu]:-}${cmd} ; "
      idx=$(( idx + 1 ))
    done
  done
done
for gpu in "${!GPUCMDS[@]}"; do ( eval "${GPUCMDS[$gpu]}" ; echo "gpu $gpu done" ) & done
echo "launched $idx runs across 8 GPUs; waiting..."
wait
echo "MIL_GRID_DONE"
