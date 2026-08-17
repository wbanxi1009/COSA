#!/usr/bin/env bash

# Compare COSA-F with each implemented normalization module. NST, the default
# normalization path, is deliberately excluded: this is a normalization-only
# ablation over the non-default methods.

set -euo pipefail

DEFAULT_MODELS=(DLinear FreTS iTransformer MICN OLS PatchTST)
DEFAULT_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather)
DEFAULT_INPUT_LENS=(96 192 336 720)
DEFAULT_PRED_LENS=(96 192 336 720)
NORMALIZATION_METHODS=(SAN RevIN)

MODELS=("${DEFAULT_MODELS[@]}")
DATASETS=("${DEFAULT_DATASETS[@]}")
INPUT_LENS=("${DEFAULT_INPUT_LENS[@]}")
PRED_LENS=("${DEFAULT_PRED_LENS[@]}")

LABEL_LEN="${LABEL_LEN:-48}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/normalization}"
# main.py appends the checkpoint-relative path, which already begins with
# normalization/<method>, to each of these roots.
BASELINE_RESULT_ROOT="${BASELINE_RESULT_ROOT:-./results/baseline}"
COSA_RESULT_ROOT="${COSA_RESULT_ROOT:-./results/SIMPLE}"
BUFFER_CONTEXT_SIZE="${BUFFER_CONTEXT_SIZE:-10}"
STEPS="${STEPS:-3}"
BATCH_SIZE="${BATCH_SIZE:-48}"
BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_eval_cosa_normalization.sh [options]

Run COSA-F with every implemented non-default normalizer: SAN and RevIN.
NST is the default normalization path and is intentionally not included.

Options accept comma-separated values and form their Cartesian product.
  --models <names>         Models to run
  --datasets <names>       Datasets to run
  --input-lens <lengths>   Input window lengths
  --pred-lens <lengths>    Prediction lengths
  -h, --help               Show this help text

Environment variables:
  LABEL_LEN, CHECKPOINT_ROOT, BASELINE_RESULT_ROOT, COSA_RESULT_ROOT,
  BUFFER_CONTEXT_SIZE, STEPS, BATCH_SIZE, BASE_LR, WEIGHT_DECAY

Results are written to:
  BASELINE_RESULT_ROOT/normalization/<normalizer>/<model>/<dataset>/input_<input_len>_pred_<pred_len>/
  COSA_RESULT_ROOT/normalization/<normalizer>/<model>/<dataset>/input_<input_len>_pred_<pred_len>/

Example:
  bash scripts/train_eval_cosa_normalization.sh --models iTransformer --datasets ETTh1 --input-lens 96 --pred-lens 96
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

validate_values() {
  local name=$1
  shift
  local value
  for value in "$@"; do
    if ! is_positive_integer "${value}"; then
      echo "${name} must contain positive integers: ${value}" >&2
      exit 1
    fi
  done
}

validate_values "LABEL_LEN" "${LABEL_LEN}"
validate_values "input lengths" "${INPUT_LENS[@]}"
validate_values "prediction lengths" "${PRED_LENS[@]}"
validate_values "COSA steps" "${STEPS}"
validate_values "COSA batch size" "${BATCH_SIZE}"

for INPUT_LEN in "${INPUT_LENS[@]}"; do
  if [[ ${LABEL_LEN} -gt ${INPUT_LEN} ]]; then
    echo "LABEL_LEN (${LABEL_LEN}) cannot exceed input length (${INPUT_LEN})" >&2
    exit 1
  fi
done

for NORMALIZATION_METHOD in "${NORMALIZATION_METHODS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      for INPUT_LEN in "${INPUT_LENS[@]}"; do
        for PRED_LEN in "${PRED_LENS[@]}"; do
          RUN_ID="input_${INPUT_LEN}_pred_${PRED_LEN}"
          CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${NORMALIZATION_METHOD}/${MODEL}/${DATASET}/${RUN_ID}"
          CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_best.pth"
          NORM_CHECKPOINT_DIR="${CHECKPOINT_DIR}/normalizer"
          NORM_CHECKPOINT_PATH="${NORM_CHECKPOINT_DIR}/checkpoint_best.pth"
          BASELINE_EXPERIMENT_DIR="${BASELINE_RESULT_ROOT}/normalization/${NORMALIZATION_METHOD}/${MODEL}/${DATASET}/${RUN_ID}"
          BASELINE_RESULT_PATH="${BASELINE_EXPERIMENT_DIR}/test.txt"
          EXPERIMENT_DIR="${COSA_RESULT_ROOT}/normalization/${NORMALIZATION_METHOD}/${MODEL}/${DATASET}/${RUN_ID}"
          COMPLETE_PATH="${EXPERIMENT_DIR}/cosa_complete"

          # OLS does not optimize RevIN parameters, so it intentionally does
          # not write a RevIN checkpoint; its initialized affine transform is
          # the one used during both fitting and COSA-F evaluation.
          REQUIRE_NORM_CHECKPOINT=True
          if [[ "${MODEL}" == "OLS" && "${NORMALIZATION_METHOD}" == "RevIN" ]]; then
            REQUIRE_NORM_CHECKPOINT=False
          fi

          if [[ ! -f "${CHECKPOINT_PATH}" ]] || { [[ "${REQUIRE_NORM_CHECKPOINT}" == "True" ]] && [[ ! -f "${NORM_CHECKPOINT_PATH}" ]]; }; then
            mkdir -p "${CHECKPOINT_DIR}" "${NORM_CHECKPOINT_DIR}"
            echo "Training: normalizer=${NORMALIZATION_METHOD}, model=${MODEL}, dataset=${DATASET}, ${RUN_ID}"
            python main.py \
              DATA.NAME "${DATASET}" \
              DATA.SEQ_LEN "${INPUT_LEN}" \
              DATA.LABEL_LEN "${LABEL_LEN}" \
              DATA.PRED_LEN "${PRED_LEN}" \
              MODEL.NAME "${MODEL}" \
              MODEL.seq_len "${INPUT_LEN}" \
              MODEL.label_len "${LABEL_LEN}" \
              MODEL.pred_len "${PRED_LEN}" \
              NORM_MODULE.ENABLE True \
              NORM_MODULE.NAME "${NORMALIZATION_METHOD}" \
              SAN.TRAIN.CHECKPOINT_DIR "${NORM_CHECKPOINT_DIR}" \
              SAN.RESULT_DIR "${NORM_CHECKPOINT_DIR}" \
              REVIN.TRAIN.CHECKPOINT_DIR "${NORM_CHECKPOINT_DIR}" \
              REVIN.RESULT_DIR "${NORM_CHECKPOINT_DIR}" \
              TRAIN.ENABLE True \
              TEST.ENABLE False \
              TTA.ENABLE False \
              TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" | tee "${CHECKPOINT_DIR}/training_output.log"

            if [[ ! -f "${CHECKPOINT_PATH}" ]] || { [[ "${REQUIRE_NORM_CHECKPOINT}" == "True" ]] && [[ ! -f "${NORM_CHECKPOINT_PATH}" ]]; }; then
              echo "Training did not produce model and normalizer checkpoints in: ${CHECKPOINT_DIR}" >&2
              exit 1
            fi
          else
            echo "Skipping training; checkpoints exist: ${CHECKPOINT_DIR}"
          fi

          if [[ -s "${BASELINE_RESULT_PATH}" ]]; then
            echo "Skipping baseline evaluation; result exists: ${BASELINE_RESULT_PATH}"
          else
            mkdir -p "${BASELINE_EXPERIMENT_DIR}"
            echo "Evaluating baseline: normalizer=${NORMALIZATION_METHOD}, model=${MODEL}, dataset=${DATASET}, ${RUN_ID}"
            python main.py \
              DATA.NAME "${DATASET}" \
              DATA.SEQ_LEN "${INPUT_LEN}" \
              DATA.LABEL_LEN "${LABEL_LEN}" \
              DATA.PRED_LEN "${PRED_LEN}" \
              MODEL.NAME "${MODEL}" \
              MODEL.seq_len "${INPUT_LEN}" \
              MODEL.label_len "${LABEL_LEN}" \
              MODEL.pred_len "${PRED_LEN}" \
              NORM_MODULE.ENABLE True \
              NORM_MODULE.NAME "${NORMALIZATION_METHOD}" \
              SAN.TRAIN.CHECKPOINT_DIR "${NORM_CHECKPOINT_DIR}" \
              REVIN.TRAIN.CHECKPOINT_DIR "${NORM_CHECKPOINT_DIR}" \
              TRAIN.ENABLE False \
              TEST.ENABLE True \
              TTA.ENABLE False \
              TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
              RESULT_DIR "${BASELINE_RESULT_ROOT}" | tee "${BASELINE_EXPERIMENT_DIR}/baseline_output.log"

            if [[ ! -s "${BASELINE_RESULT_PATH}" ]]; then
              echo "Baseline evaluation did not produce result: ${BASELINE_RESULT_PATH}" >&2
              exit 1
            fi
          fi

          if [[ -f "${COMPLETE_PATH}" ]]; then
            echo "Skipping completed COSA-F run: normalizer=${NORMALIZATION_METHOD}, model=${MODEL}, dataset=${DATASET}, ${RUN_ID}"
            continue
          fi

          mkdir -p "${EXPERIMENT_DIR}"
          echo "Running COSA-F: normalizer=${NORMALIZATION_METHOD}, model=${MODEL}, dataset=${DATASET}, ${RUN_ID}"
          python main.py \
            DATA.NAME "${DATASET}" \
            DATA.SEQ_LEN "${INPUT_LEN}" \
            DATA.LABEL_LEN "${LABEL_LEN}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            MODEL.NAME "${MODEL}" \
            MODEL.seq_len "${INPUT_LEN}" \
            MODEL.label_len "${LABEL_LEN}" \
            MODEL.pred_len "${PRED_LEN}" \
            NORM_MODULE.ENABLE True \
            NORM_MODULE.NAME "${NORMALIZATION_METHOD}" \
            SAN.TRAIN.CHECKPOINT_DIR "${NORM_CHECKPOINT_DIR}" \
            REVIN.TRAIN.CHECKPOINT_DIR "${NORM_CHECKPOINT_DIR}" \
            TRAIN.ENABLE False \
            TEST.ENABLE False \
            TTA.ENABLE True \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
            RESULT_DIR "${COSA_RESULT_ROOT}" \
            TTA.SOLVER.BASE_LR "${BASE_LR}" \
            TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
            TTA.COSA.BATCH_SIZE "${BATCH_SIZE}" \
            TTA.COSA.STEPS "${STEPS}" \
            TTA.COSA.BUFFER_CONTEXT_SIZE "${BUFFER_CONTEXT_SIZE}" \
            TTA.COSA.FAST_ADAPTATION True \
            TTA.COSA.ADAPTIVE_LR True \
            TTA.COSA.PER_BATCH_LR_RESET True \
            TTA.COSA.PAAS False \
            TTA.COSA.SAVE_CSV False | tee "${EXPERIMENT_DIR}/cosa_output.log"
          touch "${COMPLETE_PATH}"
        done
      done
    done
  done
done
