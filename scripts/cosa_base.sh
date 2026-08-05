#!/usr/bin/env bash

# Run COSA test-time adaptation for one trained base model.
# Example: bash scripts/cosa_base.sh iTransformer ETTh1 96

set -euo pipefail

if [[ $# -ne 3 ]]; then
  cat <<'EOF'
Usage: bash scripts/cosa_base.sh <model> <dataset> <pred_len>

Supported models:
  iTransformer PatchTST DLinear OLS FreTS MICN

Supported datasets:
  weather illness electricity traffic exchange_rate ETTh1 ETTh2 ETTm1 ETTm2

Examples:
  bash scripts/cosa_base.sh iTransformer ETTh1 96
  bash scripts/cosa_base.sh DLinear weather 192
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
CHECKPOINT_PATH="${CHECKPOINT_DIR}checkpoint_best.pth"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Train the base model first, for example:" >&2
  echo "  bash scripts/train_base.sh ${MODEL} ${DATASET} ${PRED_LEN}" >&2
  exit 1
fi

echo "Running COSA for ${MODEL} on ${DATASET} with prediction length ${PRED_LEN}"
echo "Loading checkpoint: ${CHECKPOINT_PATH}"

# SIMPLE Adapter Basic Configuration
BUFFER_CONTEXT_SIZE=10
STEPS=3
BATCH_SIZE=48

# PAAS  Configuration
PAAS=False
PERIOD_N=1

if [[ "${PAAS}" == "True" ]]; then
  RESULT_DIR="./results/SIMPLE/COSA_P/"
else
  RESULT_DIR="./results/SIMPLE/COSA_F/"
fi

# Fast Adaptation Optimization Settings
FAST_ADAPTATION=True
ADAPTIVE_LR=True
PER_BATCH_LR_RESET=True
MAX_LR=0.005
MIN_LR=0.0001

python main.py \
  DATA.NAME "${DATASET}" \
  DATA.PRED_LEN "${PRED_LEN}" \
  MODEL.NAME "${MODEL}" \
  MODEL.pred_len "${PRED_LEN}" \
  TRAIN.ENABLE False \
  TEST.ENABLE False \
  TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
  RESULT_DIR "${RESULT_DIR}" \
  TTA.ENABLE True \
  TTA.SOLVER.BASE_LR 0.001 \
  TTA.SOLVER.WEIGHT_DECAY 0.0001 \
  TTA.COSA.BATCH_SIZE ${BATCH_SIZE} \
  TTA.COSA.STEPS ${STEPS} \
  TTA.COSA.BUFFER_CONTEXT_SIZE ${BUFFER_CONTEXT_SIZE} \
  TTA.COSA.FAST_ADAPTATION ${FAST_ADAPTATION} \
  TTA.COSA.PER_BATCH_LR_RESET ${PER_BATCH_LR_RESET} \
  TTA.COSA.ADAPTIVE_LR ${ADAPTIVE_LR} \
  TTA.COSA.PAAS ${PAAS} \
  TTA.COSA.PERIOD_N ${PERIOD_N}
