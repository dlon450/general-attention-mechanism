#!/usr/bin/env bash
set -euo pipefail

# Dedicated sweep for query-conditioned within-support attention weights.
# Exact ordinary attention is recovered with:
#   --f1-type restricted_softmax --f2-type full_set --f1-query-mode none

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-cifar10}"
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-results}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"

GIBBS_STEPS="${GIBBS_STEPS:-24}"
GIBBS_RUNS="${GIBBS_RUNS:-6}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-128}"

F1_MAX_SET_SIZE="${F1_MAX_SET_SIZE:-8}"
F2_HIDDEN="${F2_HIDDEN:-128}"
DOT_REP_LAMBDA="${DOT_REP_LAMBDA:--0.1}"
SINGLETON_TAU_INIT="${SINGLETON_TAU_INIT:-2.0}"
VERIFY_FIRST="${VERIFY_FIRST:-1}"

F2S=(full_set modular_dot modular_dot_hard_singleton modular_dot_first_free dot_repulsion neural_mlp)

if [[ -n "${ONLY_F2:-}" ]]; then
  read -r -a F2S <<< "${ONLY_F2}"
fi

sanitize_token() {
  local s="$1"
  s="${s//-/m}"
  s="${s//./p}"
  echo "$s"
}

run_one() {
  local f2="$1"
  local tau_init="0.0"
  local run_name="sweep_${DATASET}_restrictedsoftmax_f2${f2}"
  local -a extra_args

  extra_args=(
    --f1-concat-max-set-size "${F1_MAX_SET_SIZE}"
  )

  if [[ "${f2}" == "neural_mlp" ]]; then
    extra_args+=(--f2-neural-hidden "${F2_HIDDEN}")
  fi

  if [[ "${f2}" == "dot_repulsion" ]]; then
    extra_args+=(--gibbs-repulsion-lambda "${DOT_REP_LAMBDA}")
    run_name="${run_name}_lam$(sanitize_token "${DOT_REP_LAMBDA}")"
  fi

  if [[ "${f2}" == "modular_dot_hard_singleton" || "${f2}" == "modular_dot_first_free" ]]; then
    tau_init="${SINGLETON_TAU_INIT}"
    run_name="${run_name}_tau$(sanitize_token "${tau_init}")"
  fi

  if [[ "${f2}" == "full_set" ]]; then
    run_name="${run_name}_exact"
  else
    run_name="${run_name}_s${GIBBS_STEPS}_r${GIBBS_RUNS}"
  fi

  echo "===== ${run_name} ====="
  "${PYTHON_BIN}" -u train_vit_cifar.py \
    --dataset "${DATASET}" \
    --attention general \
    --device "${DEVICE}" \
    --amp \
    --download \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --gibbs-steps "${GIBBS_STEPS}" \
    --gibbs-runs "${GIBBS_RUNS}" \
    --query-chunk-size "${QUERY_CHUNK_SIZE}" \
    --st-gradient-mode consistent \
    --f1-type restricted_softmax \
    --f2-type "${f2}" \
    --f1-query-mode none \
    --tau-init "${tau_init}" \
    --save-dir "${SAVE_DIR}" \
    --run-name "${run_name}" \
    "${extra_args[@]}"
}

if [[ "${VERIFY_FIRST}" == "1" ]]; then
  echo "===== verify_exact_attention ====="
  "${PYTHON_BIN}" verify_exact_attention.py
fi

for f2 in "${F2S[@]}"; do
  run_one "${f2}"
done
