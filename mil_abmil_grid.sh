#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
OUT="$ROOT/mil_abmil_results5"; mkdir -p "$OUT"
STEPS="${STEPS:-2500}"; N="${N:-32}"; AEM_COEF="${AEM_COEF:-0.02}"
MODES=(abmil aem sparsemax entmax15 rep)
SEEDS=(0 1 2 3 4)
declare -A GPUCMDS; idx=0
for task in redundancy needle; do
  for m in "${MODES[@]}"; do
    for s in "${SEEDS[@]}"; do
      gpu=$(( idx % 8 )); name="${task}_${m}_s${s}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $V $ROOT/mil_abmil.py --mode $m --task $task --N $N --steps $STEPS --seed $s --aem-coef $AEM_COEF > $OUT/$name.json 2> $OUT/$name.err"
      GPUCMDS[$gpu]="${GPUCMDS[$gpu]:-}${cmd} ; "
      idx=$(( idx + 1 ))
    done
  done
done
for gpu in "${!GPUCMDS[@]}"; do ( eval "${GPUCMDS[$gpu]}" ; echo "gpu $gpu done" ) & done
echo "launched $idx runs across 8 GPUs; waiting..."
wait
echo "GRID5_DONE"
