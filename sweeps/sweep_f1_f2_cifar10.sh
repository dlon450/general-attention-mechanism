#!/usr/bin/env bash
set -euo pipefail

# Meaningful F1 sweeps should keep query-conditioned replacement disabled,
# otherwise F1 is bypassed in the final output path.

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
F1_HIDDEN="${F1_HIDDEN:-128}"
F2_HIDDEN="${F2_HIDDEN:-128}"
DOT_REP_LAMBDA="${DOT_REP_LAMBDA:--0.1}"
SINGLETON_TAU_INIT="${SINGLETON_TAU_INIT:-2.0}"

F1S=(mean mlp_mean mlp_concat transformer)
F2S=(modular_dot modular_dot_hard_singleton modular_dot_first_free dot_repulsion neural_mlp)

if [[ -n "${ONLY_F1:-}" ]]; then
  read -r -a F1S <<< "${ONLY_F1}"
fi
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
  local f1="$1"
  local f2="$2"
  local tau_init="0.0"
  local run_name="sweep_${DATASET}_f1${f1}_f2${f2}_s${GIBBS_STEPS}_r${GIBBS_RUNS}"
  local -a extra_args

  extra_args=(
    --f1-concat-max-set-size "${F1_MAX_SET_SIZE}"
  )

  if [[ "${f1}" == "mlp_concat" || "${f1}" == "transformer" ]]; then
    extra_args+=(--f1-concat-hidden "${F1_HIDDEN}")
  fi

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
    --f1-type "${f1}" \
    --f2-type "${f2}" \
    --f1-query-mode none \
    --tau-init "${tau_init}" \
    --save-dir "${SAVE_DIR}" \
    --run-name "${run_name}" \
    "${extra_args[@]}"
}

for f1 in "${F1S[@]}"; do
  for f2 in "${F2S[@]}"; do
    run_one "${f1}" "${f2}"
  done
done
