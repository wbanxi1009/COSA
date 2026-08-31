#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export SEED="${SEED:-0}"
export FORCE="${FORCE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INPUT_LEN=96
ENV_NAME="${ENV_NAME:-cosa}"

run_stage() {
  local stage="$1"
  shift

  printf '\n[%s] START time=%s input_len=%s seed=%s\n' \
    "${stage}" "$(date -Is)" "${INPUT_LEN}" "${SEED}"
  "$@"
  printf '[%s] COMPLETE time=%s\n' "${stage}" "$(date -Is)"
}

run_stage "TRAIN" mamba run -n "${ENV_NAME}" bash scripts/train.sh
run_stage "COSA-F" env PAAS=False mamba run -n "${ENV_NAME}" bash scripts/cosa.sh
run_stage "COSA-P" env PAAS=True mamba run -n "${ENV_NAME}" bash scripts/cosa.sh
run_stage "TAFAS" mamba run -n "${ENV_NAME}" bash scripts/tafas.sh
run_stage "PETSA" mamba run -n "${ENV_NAME}" bash scripts/petsa.sh

printf '\nExperiment 1 complete: input_len=%s seed=%s\n' "${INPUT_LEN}" "${SEED}"
