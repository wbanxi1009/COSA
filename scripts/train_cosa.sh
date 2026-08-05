#!/usr/bin/env bash

# Train each model/dataset/horizon combination, then run COSA TTA from its checkpoint.
# Edit the arrays below to select the experiments to run.

set -euo pipefail

MODELS=(DLinear FreTS iTransformer MICN OLS PatchTST)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather)
PRED_LENS=(96 192 336 720)

# COSA configuration
BUFFER_CONTEXT_SIZE=10
STEPS=3
BATCH_SIZE=48
PAAS=False
PERIOD_N=1
FAST_ADAPTATION=True
ADAPTIVE_LR=True
PER_BATCH_LR_RESET=True
BASE_LR=0.001
WEIGHT_DECAY=0.0001

# "SIMPLE" is required by main.py to dispatch to tta.cosa.
if [[ "${PAAS}" == "True" ]]; then
  RESULT_DIR="./results/SIMPLE/COSA_P/"
else
  RESULT_DIR="./results/SIMPLE/COSA_F/"
fi

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"

      echo "Training: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"
      python main.py \
        DATA.NAME "${DATASET}" \
        DATA.PRED_LEN "${PRED_LEN}" \
        MODEL.NAME "${MODEL}" \
        MODEL.pred_len "${PRED_LEN}" \
        TRAIN.ENABLE True \
        TEST.ENABLE False \
        TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}"

      CHECKPOINT_PATH="${CHECKPOINT_DIR}checkpoint_best.pth"
      if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        echo "Training did not produce checkpoint: ${CHECKPOINT_PATH}" >&2
        exit 1
      fi

      echo "Running COSA: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"
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
        TTA.SOLVER.BASE_LR "${BASE_LR}" \
        TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
        TTA.COSA.BATCH_SIZE "${BATCH_SIZE}" \
        TTA.COSA.STEPS "${STEPS}" \
        TTA.COSA.BUFFER_CONTEXT_SIZE "${BUFFER_CONTEXT_SIZE}" \
        TTA.COSA.FAST_ADAPTATION "${FAST_ADAPTATION}" \
        TTA.COSA.PER_BATCH_LR_RESET "${PER_BATCH_LR_RESET}" \
        TTA.COSA.ADAPTIVE_LR "${ADAPTIVE_LR}" \
        TTA.COSA.PAAS "${PAAS}" \
        TTA.COSA.PERIOD_N "${PERIOD_N}"
    done
  done
done
