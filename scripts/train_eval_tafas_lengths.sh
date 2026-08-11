#!/usr/bin/env bash

# Run a resumable TAFAS sweep over models, datasets, input lengths, and
# prediction lengths. Each combination has an independent checkpoint and result
# directory so runs with different lengths never overwrite one another.

set -euo pipefail

DEFAULT_MODELS=(DLinear FreTS iTransformer MICN OLS PatchTST)
DEFAULT_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather)
DEFAULT_INPUT_LENS=(96 192 336 720)
DEFAULT_PRED_LENS=(96 192 336 720)

MODELS=("${DEFAULT_MODELS[@]}")
DATASETS=("${DEFAULT_DATASETS[@]}")
INPUT_LENS=("${DEFAULT_INPUT_LENS[@]}")
PRED_LENS=("${DEFAULT_PRED_LENS[@]}")

# The decoder label length is independent of the input length. Override it with
# LABEL_LEN=<positive integer> when a model/dataset setup requires another value.
LABEL_LEN="${LABEL_LEN:-48}"

# TAFAS configuration. PAAS dynamically sets each adaptation batch size.
BASE_LR=0.001
WEIGHT_DECAY=0.0001
GATING_INIT=0.01
PAAS=True
PERIOD_N=1
STEPS=1
ADJUST_PRED=True
CALI_MODULE=True

CHECKPOINT_ROOT="./checkpoints"
BASELINE_RESULT_ROOT="./results/baseline"
# "TAFAS" is required by main.py to dispatch to tta.tafas.
TAFAS_RESULT_ROOT="./results/TAFAS"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_eval_tafas_lengths.sh [options]

Options accept comma-separated values and form their Cartesian product.
  --models <names>         Models to run (default: DLinear,FreTS,iTransformer,MICN,OLS,PatchTST)
  --datasets <names>       Datasets to run (default: ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather)
  --input-lens <lengths>   Input window lengths (default: 96,192,336,720)
  --pred-lens <lengths>    Prediction lengths (default: 96,192,336,720)
  -h, --help               Show this help text

Set LABEL_LEN in the environment to change the decoder label length (default: 48).

Examples:
  bash scripts/train_eval_tafas_lengths.sh --models iTransformer --datasets ETTh1 --input-lens 96,192 --pred-lens 96,192
  LABEL_LEN=96 bash scripts/train_eval_tafas_lengths.sh --models PatchTST --datasets weather --input-lens 336 --pred-lens 720
EOF
}

parse_csv() {
  local value=$1
  local -n target=$2
  IFS=',' read -r -a target <<< "${value}"
  if [[ ${#target[@]} -eq 0 ]] || [[ -z ${target[0]} ]]; then
    echo "Expected at least one comma-separated value" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      [[ $# -ge 2 ]] || { echo "Missing value for --models" >&2; exit 1; }
      parse_csv "$2" MODELS
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "Missing value for --datasets" >&2; exit 1; }
      parse_csv "$2" DATASETS
      shift 2
      ;;
    --input-lens)
      [[ $# -ge 2 ]] || { echo "Missing value for --input-lens" >&2; exit 1; }
      parse_csv "$2" INPUT_LENS
      shift 2
      ;;
    --pred-lens)
      [[ $# -ge 2 ]] || { echo "Missing value for --pred-lens" >&2; exit 1; }
      parse_csv "$2" PRED_LENS
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

is_positive_integer() {
  [[ $1 =~ ^[0-9]+$ ]] && [[ $1 -gt 0 ]]
}

if ! is_positive_integer "${LABEL_LEN}"; then
  echo "LABEL_LEN must be a positive integer: ${LABEL_LEN}" >&2
  exit 1
fi

for INPUT_LEN in "${INPUT_LENS[@]}"; do
  if ! is_positive_integer "${INPUT_LEN}"; then
    echo "Input length must be a positive integer: ${INPUT_LEN}" >&2
    exit 1
  fi
  if [[ ${LABEL_LEN} -gt ${INPUT_LEN} ]]; then
    echo "LABEL_LEN (${LABEL_LEN}) cannot exceed input length (${INPUT_LEN})" >&2
    exit 1
  fi
done

for PRED_LEN in "${PRED_LENS[@]}"; do
  if ! is_positive_integer "${PRED_LEN}"; then
    echo "Prediction length must be a positive integer: ${PRED_LEN}" >&2
    exit 1
  fi
done

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for INPUT_LEN in "${INPUT_LENS[@]}"; do
      for PRED_LEN in "${PRED_LENS[@]}"; do
        RUN_ID="input_${INPUT_LEN}_pred_${PRED_LEN}"
        CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/${DATASET}/${RUN_ID}/"
        BASELINE_EXPERIMENT_DIR="${BASELINE_RESULT_ROOT}/${MODEL}/${DATASET}/${RUN_ID}"
        TAFAS_EXPERIMENT_DIR="${TAFAS_RESULT_ROOT}/${MODEL}/${DATASET}/${RUN_ID}"
        CHECKPOINT_PATH="${CHECKPOINT_DIR}checkpoint_best.pth"
        BASELINE_RESULT_PATH="${BASELINE_EXPERIMENT_DIR}/test.txt"
        TAFAS_COMPLETE_PATH="${TAFAS_EXPERIMENT_DIR}/tafas_complete"

        if [[ -f "${CHECKPOINT_PATH}" ]]; then
          echo "Skipping training; checkpoint exists: ${CHECKPOINT_PATH}"
        else
          mkdir -p "${CHECKPOINT_DIR}"
          echo "Training: model=${MODEL}, dataset=${DATASET}, input_len=${INPUT_LEN}, pred_len=${PRED_LEN}"
          python main.py \
            DATA.NAME "${DATASET}" \
            DATA.SEQ_LEN "${INPUT_LEN}" \
            DATA.LABEL_LEN "${LABEL_LEN}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            MODEL.NAME "${MODEL}" \
            MODEL.seq_len "${INPUT_LEN}" \
            MODEL.label_len "${LABEL_LEN}" \
            MODEL.pred_len "${PRED_LEN}" \
            TRAIN.ENABLE True \
            TEST.ENABLE False \
            TTA.ENABLE False \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" 2>&1 | tee "${CHECKPOINT_DIR}/training_output.log"

          if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
            echo "Training did not produce checkpoint: ${CHECKPOINT_PATH}" >&2
            exit 1
          fi
        fi

        if [[ -s "${BASELINE_RESULT_PATH}" ]]; then
          echo "Skipping baseline evaluation; result exists: ${BASELINE_RESULT_PATH}"
        else
          mkdir -p "${BASELINE_EXPERIMENT_DIR}"
          echo "Evaluating baseline: model=${MODEL}, dataset=${DATASET}, input_len=${INPUT_LEN}, pred_len=${PRED_LEN}"
          python main.py \
            DATA.NAME "${DATASET}" \
            DATA.SEQ_LEN "${INPUT_LEN}" \
            DATA.LABEL_LEN "${LABEL_LEN}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            MODEL.NAME "${MODEL}" \
            MODEL.seq_len "${INPUT_LEN}" \
            MODEL.label_len "${LABEL_LEN}" \
            MODEL.pred_len "${PRED_LEN}" \
            TRAIN.ENABLE False \
            TEST.ENABLE True \
            TTA.ENABLE False \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
            RESULT_DIR "${BASELINE_RESULT_ROOT}" 2>&1 | tee "${BASELINE_EXPERIMENT_DIR}/baseline_output.log"

          if [[ ! -s "${BASELINE_RESULT_PATH}" ]]; then
            echo "Baseline evaluation did not produce result: ${BASELINE_RESULT_PATH}" >&2
            exit 1
          fi
        fi

        if [[ -f "${TAFAS_COMPLETE_PATH}" ]]; then
          echo "Skipping TAFAS; completed result exists: ${TAFAS_COMPLETE_PATH}"
        else
          mkdir -p "${TAFAS_EXPERIMENT_DIR}"
          echo "Running TAFAS: model=${MODEL}, dataset=${DATASET}, input_len=${INPUT_LEN}, pred_len=${PRED_LEN}"
          python main.py \
            DATA.NAME "${DATASET}" \
            DATA.SEQ_LEN "${INPUT_LEN}" \
            DATA.LABEL_LEN "${LABEL_LEN}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            MODEL.NAME "${MODEL}" \
            MODEL.seq_len "${INPUT_LEN}" \
            MODEL.label_len "${LABEL_LEN}" \
            MODEL.pred_len "${PRED_LEN}" \
            TRAIN.ENABLE False \
            TEST.ENABLE False \
            TTA.ENABLE True \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
            RESULT_DIR "${TAFAS_RESULT_ROOT}" \
            TTA.SOLVER.BASE_LR "${BASE_LR}" \
            TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
            TTA.TAFAS.GATING_INIT "${GATING_INIT}" \
            TTA.TAFAS.PAAS "${PAAS}" \
            TTA.TAFAS.PERIOD_N "${PERIOD_N}" \
            TTA.TAFAS.STEPS "${STEPS}" \
            TTA.TAFAS.ADJUST_PRED "${ADJUST_PRED}" \
            TTA.TAFAS.CALI_MODULE "${CALI_MODULE}" 2>&1 | tee "${TAFAS_EXPERIMENT_DIR}/tafas_output.log"
          touch "${TAFAS_COMPLETE_PATH}"
        fi
      done
    done
  done
done
