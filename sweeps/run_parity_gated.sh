#!/usr/bin/env bash
set -euo pipefail

# Phase-A parity protocol: deterministic mean-field gated-softmax vs dense MHA,
# PARAMETER-MATCHED and REGULARIZATION-MATCHED, over multiple seeds.
#
# Success criterion (parity): gated (beta_init=0) final val-acc within +/-1 std of
# tuned dense MHA over the seeds. Then flip GATED_BETA_INIT to a small positive value
# (e.g. 0.5) to let the gate depart from softmax and look for a >seed-std improvement.
#
# NOTE: CIFAR-10 (L=65) is near-worst-case for a win (little softmax dilution). Expect
# parity here; the win experiment belongs on a large-L / distractor-heavy benchmark.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-cifar10}"
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-results_parity}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DROPOUT="${DROPOUT:-0.1}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.1}"   # regularization-matched baseline: dense MHA WITH dropout
NUM_WORKERS="${NUM_WORKERS:-8}"
SEEDS="${SEEDS:-0 1 2 3 4}"

GATED_BETA_INIT="${GATED_BETA_INIT:-0.0}"   # 0.0 = parity proof; 0.5 = live-gate win run
GATED_REPULSION="${GATED_REPULSION:-0}"     # 1 to enable the Phase-B DPP term

run_one() {
  local attn="$1"; local seed="$2"
  local run_name="parity_${DATASET}_${attn}_seed${seed}"
  local -a extra=()
  if [[ "${attn}" == "gated" ]]; then
    extra+=(--gated-beta-init "${GATED_BETA_INIT}")
    [[ "${GATED_REPULSION}" == "1" ]] && extra+=(--gated-repulsion)
    run_name="parity_${DATASET}_gated_b$(echo "${GATED_BETA_INIT}" | tr '.' 'p')_seed${seed}"
  fi
  echo "===== ${run_name} ====="
  "${PYTHON_BIN}" -u train_vit_cifar.py \
    --dataset "${DATASET}" --attention "${attn}" --device "${DEVICE}" --amp --download \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" --attn-dropout "${ATTN_DROPOUT}" \
    --num-workers "${NUM_WORKERS}" --seed "${seed}" \
    --save-dir "${SAVE_DIR}" --run-name "${run_name}" \
    "${extra[@]}"
}

for seed in ${SEEDS}; do
  run_one mha "${seed}"
  run_one gated "${seed}"
done

echo "Done. Summaries in ${SAVE_DIR}/*.json (best_val_acc field). Compare mean +/- std across seeds."
