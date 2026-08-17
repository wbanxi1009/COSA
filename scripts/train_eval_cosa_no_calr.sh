#!/usr/bin/env bash

# Train/evaluate base models, then run COSA with CALR disabled. This preserves
# COSA adaptation while using a fixed TTA.SOLVER.BASE_LR.

set -euo pipefail

MODELS=(DLinear FreTS iTransformer MICN OLS PatchTST)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather)
PRED_LENS=(96 192 336 720)

# Keep these values identical to the CALR-enabled experiment.
BUFFER_CONTEXT_SIZE=10
STEPS=3
BATCH_SIZE=48
PAAS=False
PERIOD_N=1
FAST_ADAPTATION=True
ADAPTIVE_LR=False
PER_BATCH_LR_RESET=True
BASE_LR=0.001
WEIGHT_DECAY=0.0001

BASELINE_RESULT_DIR="./results/baseline"
if [[ "${PAAS}" == "True" ]]; then
  COSA_RESULT_DIR="./results/SIMPLE/COSA_P_no_CALR"
else
  COSA_RESULT_DIR="./results/SIMPLE/COSA_F_no_CALR"
fi

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
      CHECKPOINT_PATH="${CHECKPOINT_DIR}checkpoint_best.pth"
      EXPERIMENT_DIR="${MODEL}/${DATASET}_${PRED_LEN}"

      if [[ -f "${CHECKPOINT_PATH}" ]]; then
        echo "Skipping training; reusing checkpoint: ${CHECKPOINT_PATH}"
      else
        echo "Training: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"
        python main.py \
          DATA.NAME "${DATASET}" \
          DATA.PRED_LEN "${PRED_LEN}" \
          MODEL.NAME "${MODEL}" \
          MODEL.pred_len "${PRED_LEN}" \
          TRAIN.ENABLE True \
          TEST.ENABLE False \
          TTA.ENABLE False \
          TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}"

        if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
          echo "Training did not produce checkpoint: ${CHECKPOINT_PATH}" >&2
          exit 1
        fi
      fi

      echo "Evaluating base model: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"
      python main.py \
        DATA.NAME "${DATASET}" \
        DATA.PRED_LEN "${PRED_LEN}" \
        MODEL.NAME "${MODEL}" \
        MODEL.pred_len "${PRED_LEN}" \
        TRAIN.ENABLE False \
        TEST.ENABLE True \
        TTA.ENABLE False \
        TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
        RESULT_DIR "${BASELINE_RESULT_DIR}"

      COSA_EXPERIMENT_DIR="${COSA_RESULT_DIR}/${EXPERIMENT_DIR}"
      mkdir -p "${COSA_EXPERIMENT_DIR}"

      echo "Running COSA without CALR: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"
      python main.py \
        DATA.NAME "${DATASET}" \
        DATA.PRED_LEN "${PRED_LEN}" \
        MODEL.NAME "${MODEL}" \
        MODEL.pred_len "${PRED_LEN}" \
        TRAIN.ENABLE False \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
        RESULT_DIR "${COSA_RESULT_DIR}" \
        TTA.SOLVER.BASE_LR "${BASE_LR}" \
        TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
        TTA.COSA.BATCH_SIZE "${BATCH_SIZE}" \
        TTA.COSA.STEPS "${STEPS}" \
        TTA.COSA.BUFFER_CONTEXT_SIZE "${BUFFER_CONTEXT_SIZE}" \
        TTA.COSA.FAST_ADAPTATION "${FAST_ADAPTATION}" \
        TTA.COSA.PER_BATCH_LR_RESET "${PER_BATCH_LR_RESET}" \
        TTA.COSA.ADAPTIVE_LR "${ADAPTIVE_LR}" \
        TTA.COSA.PAAS "${PAAS}" \
        TTA.COSA.PERIOD_N "${PERIOD_N}" \
        TTA.COSA.SAVE_CSV True | tee "${COSA_EXPERIMENT_DIR}/cosa_output.log"
    done
  done
done
