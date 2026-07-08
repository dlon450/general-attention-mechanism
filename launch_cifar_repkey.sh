#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/users/dereklong/scratch/general-attention-mechanism
V=/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/.venv/bin/python
D="$ROOT/data"; SAVE="$ROOT/results_cifar_repkey"; LOGS="$SAVE/logs"; mkdir -p "$LOGS"
COMMON=(--dataset cifar10 --device cuda --amp --data-dir "$D" --epochs 100 --batch-size 256
  --eval-batch-size 512 --lr 5e-4 --weight-decay 0.05 --dropout 0.1 --attn-dropout 0.1
  --num-workers 6 --log-every-batches 0 --checkpoint-every 0 --save-dir "$SAVE")

declare -A Q; idx=0
add(){ local g=$(( idx % 8 )); Q[$g]="${Q[$g]:-}$1 ; "; idx=$(( idx+1 )); }
for s in 0 1 2 3 4; do
  add "CUDA_VISIBLE_DEVICES=$(( idx%8 )) $V -u $ROOT/train_vit_cifar.py ${COMMON[*]} --attention mha --seed $s --run-name mha_s$s > $LOGS/mha_s$s.log 2>&1"
  add "CUDA_VISIBLE_DEVICES=$(( idx%8 )) $V -u $ROOT/train_vit_cifar.py ${COMMON[*]} --attention gated --gated-beta-init 0.5 --gated-repulsion --gated-lambda-init 1.0 --seed $s --run-name repkey_s$s > $LOGS/repkey_s$s.log 2>&1"
done
for g in "${!Q[@]}"; do ( eval "${Q[$g]}" ; echo "gpu $g done" ) & done
echo "launched $idx runs across 8 GPUs; waiting..."
wait
echo "CIFAR_REPKEY_DONE"
