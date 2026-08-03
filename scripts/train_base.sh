#!/usr/bin/env bash

# Train one base forecasting model on one dataset and prediction horizon.
# Example: bash scripts/train_base.sh iTransformer ETTh1 96

set -euo pipefail

if [[ $# -ne 3 ]]; then
  cat <<'EOF'
Usage: bash scripts/train_base.sh <model> <dataset> <pred_len>

Supported models:
  iTransformer PatchTST DLinear OLS FreTS MICN

Supported datasets:
  weather illness electricity traffic exchange_rate ETTh1 ETTh2 ETTm1 ETTm2

Examples:
  bash scripts/train_base.sh iTransformer ETTh1 96
  bash scripts/train_base.sh DLinear weather 192
EOF
  exit 1
fi

MODEL=$1
DATASET=$2
PRED_LEN=$3

SUPPORTED_MODELS=(iTransformer PatchTST DLinear OLS FreTS MICN)
SUPPORTED_DATASETS=(weather illness electricity traffic exchange_rate ETTh1 ETTh2 ETTm1 ETTm2)

if [[ ! " ${SUPPORTED_MODELS[*]} " =~ " ${MODEL} " ]]; then
  echo "Unsupported model: ${MODEL}" >&2
  exit 1
fi

if [[ ! " ${SUPPORTED_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  echo "Unsupported dataset: ${DATASET}" >&2
  exit 1
fi

if [[ ! ${PRED_LEN} =~ ^[0-9]+$ ]] || [[ ${PRED_LEN} -le 0 ]]; then
  echo "Prediction length must be a positive integer: ${PRED_LEN}" >&2
  exit 1
fi

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
mkdir -p "${CHECKPOINT_DIR}"

echo "Training ${MODEL} on ${DATASET} with prediction length ${PRED_LEN}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

python main.py \
  DATA.NAME "${DATASET}" \
  DATA.PRED_LEN "${PRED_LEN}" \
  MODEL.NAME "${MODEL}" \
  MODEL.pred_len "${PRED_LEN}" \
  TRAIN.ENABLE True \
  TEST.ENABLE False \
  TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}"
