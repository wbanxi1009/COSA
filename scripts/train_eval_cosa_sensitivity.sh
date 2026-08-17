#!/usr/bin/env bash

# Run resumable one-factor-at-a-time COSA sensitivity sweeps over adaptation
# steps (S), context length (K), and adaptation batch size (B). The two
# parameters not under test stay at their config.py defaults.

set -euo pipefail

DEFAULT_MODELS=(DLinear FreTS iTransformer MICN OLS PatchTST)
DEFAULT_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather)
DEFAULT_INPUT_LENS=(96 192 336 720)
DEFAULT_PRED_LENS=(96 192 336 720)
DEFAULT_STEPS=(1 3 5 10 20)
DEFAULT_CONTEXT_SIZES=(1 3 5 10 20)
DEFAULT_BATCH_SIZES=(12 24 48 96)
DEFAULT_SEEDS=(0)

# These values must match config.py's TTA.COSA defaults.
DEFAULT_STEP_COUNT=20
DEFAULT_CONTEXT_SIZE=5
DEFAULT_BATCH_SIZE=25

MODELS=("${DEFAULT_MODELS[@]}")
DATASETS=("${DEFAULT_DATASETS[@]}")
INPUT_LENS=("${DEFAULT_INPUT_LENS[@]}")
PRED_LENS=("${DEFAULT_PRED_LENS[@]}")
STEPS=("${DEFAULT_STEPS[@]}")
CONTEXT_SIZES=("${DEFAULT_CONTEXT_SIZES[@]}")
BATCH_SIZES=("${DEFAULT_BATCH_SIZES[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")

LABEL_LEN="${LABEL_LEN:-48}"
BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-./results/SIMPLE/sensitivity}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_eval_cosa_sensitivity.sh [options]

Options accept comma-separated values. Each parameter is swept independently;
the other two COSA parameters remain at their config.py defaults (S=20, K=5,
B=25).
  --models <names>          Models to evaluate (default: iTransformer)
  --datasets <names>        Datasets to evaluate (default: ETTh1)
  --input-lens <lengths>    Input lengths used to train checkpoints (default: 96)
  --pred-lens <lengths>     Prediction lengths used to train checkpoints (default: 96)
  --steps <values>          COSA adaptation steps S (default: 1,3,5,10,20)
  --context-sizes <values>  COSA context sizes K (default: 1,3,5,10,20)
  --batch-sizes <values>    COSA adaptation batch sizes B (default: 12,24,48,96)
  --seeds <values>          Random seeds (default: 0)
  -h, --help                Show this help text

Environment variables:
  LABEL_LEN, BASE_LR, WEIGHT_DECAY, CHECKPOINT_ROOT, RESULT_ROOT

The expected checkpoint path is:
  CHECKPOINT_ROOT/<model>/<dataset>/input_<input_len>_pred_<pred_len>/checkpoint_best.pth

PAAS and fast adaptation are disabled so BATCH_SIZE and STEPS are applied
exactly. Each S/K/B/seed combination gets a separate result directory.

Example:
  bash scripts/train_eval_cosa_sensitivity.sh \
    --models iTransformer --datasets ETTh1 \
    --input-lens 96 --pred-lens 96 \
    --steps 1,3,5,10 --context-sizes 1,5,10 --batch-sizes 24,48
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
    --steps)
      [[ $# -ge 2 ]] || { echo "Missing value for --steps" >&2; exit 1; }
      parse_csv "$2" STEPS
      shift 2
      ;;
    --context-sizes)
      [[ $# -ge 2 ]] || { echo "Missing value for --context-sizes" >&2; exit 1; }
      parse_csv "$2" CONTEXT_SIZES
      shift 2
      ;;
    --batch-sizes)
      [[ $# -ge 2 ]] || { echo "Missing value for --batch-sizes" >&2; exit 1; }
      parse_csv "$2" BATCH_SIZES
      shift 2
      ;;
    --seeds)
      [[ $# -ge 2 ]] || { echo "Missing value for --seeds" >&2; exit 1; }
      parse_csv "$2" SEEDS
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
validate_values "steps" "${STEPS[@]}"
validate_values "context sizes" "${CONTEXT_SIZES[@]}"
validate_values "batch sizes" "${BATCH_SIZES[@]}"
validate_values "seeds" "${SEEDS[@]}"

for INPUT_LEN in "${INPUT_LENS[@]}"; do
  if [[ ${LABEL_LEN} -gt ${INPUT_LEN} ]]; then
    echo "LABEL_LEN (${LABEL_LEN}) cannot exceed input length (${INPUT_LEN})" >&2
    exit 1
  fi
done

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for INPUT_LEN in "${INPUT_LENS[@]}"; do
      for PRED_LEN in "${PRED_LENS[@]}"; do
        RUN_ID="input_${INPUT_LEN}_pred_${PRED_LEN}"
        CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/${DATASET}/${RUN_ID}"
        CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_best.pth"

        if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
          echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
          exit 1
        fi

        run_experiment() {
          local step_count=$1
          local context_size=$2
          local batch_size=$3
          local seed=$4
          local experiment_id="S${step_count}_K${context_size}_B${batch_size}_seed${seed}"
          local experiment_root="${RESULT_ROOT}/${experiment_id}"
          local experiment_dir="${experiment_root}/${MODEL}/${DATASET}/${RUN_ID}"
          local complete_path="${experiment_dir}/cosa_complete"

          if [[ -f "${complete_path}" ]]; then
            echo "Skipping completed COSA sensitivity run: ${experiment_id} ${MODEL} ${DATASET} ${RUN_ID}"
            return
          fi

          mkdir -p "${experiment_dir}"
          echo "Running COSA: model=${MODEL}, dataset=${DATASET}, ${RUN_ID}, ${experiment_id}"
          python main.py \
            SEED "${seed}" \
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
            RESULT_DIR "${experiment_root}" \
            TTA.SOLVER.BASE_LR "${BASE_LR}" \
            TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
            TTA.COSA.STEPS "${step_count}" \
            TTA.COSA.BUFFER_CONTEXT_SIZE "${context_size}" \
            TTA.COSA.BATCH_SIZE "${batch_size}" \
            TTA.COSA.PAAS False \
            TTA.COSA.FAST_ADAPTATION False \
            TTA.COSA.SAVE_CSV False | tee "${experiment_dir}/cosa_output.log"
          touch "${complete_path}"
        }

        for STEP_COUNT in "${STEPS[@]}"; do
          for SEED in "${SEEDS[@]}"; do
            run_experiment "${STEP_COUNT}" "${DEFAULT_CONTEXT_SIZE}" "${DEFAULT_BATCH_SIZE}" "${SEED}"
          done
        done

        for CONTEXT_SIZE in "${CONTEXT_SIZES[@]}"; do
          for SEED in "${SEEDS[@]}"; do
            run_experiment "${DEFAULT_STEP_COUNT}" "${CONTEXT_SIZE}" "${DEFAULT_BATCH_SIZE}" "${SEED}"
          done
        done

        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
          for SEED in "${SEEDS[@]}"; do
            run_experiment "${DEFAULT_STEP_COUNT}" "${DEFAULT_CONTEXT_SIZE}" "${BATCH_SIZE}" "${SEED}"
          done
        done
      done
    done
  done
done
