#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
if [[ -n "${INPUT_LEN:-}" ]]; then
  [[ "${INPUT_LEN}" =~ ^[1-9][0-9]*$ ]] || { printf 'INPUT_LEN must be a positive integer: %s\n' "${INPUT_LEN}" >&2; exit 1; }
  INPUT_LENS=("${INPUT_LEN}")
else
  INPUT_LENS=(96 192 336 720)
fi
PRED_LENS=(96 192 336 720)
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

COMPLETED=0
SKIPPED=0
FAILED=0
CURRENT_ACTIVE=0
STATUS_FILE=""
LOG_FILE=""
MODEL=""
DATASET=""
SEQ_LEN=""
PRED_LEN=""
EXPECTED_CONFIG_FINGERPRINT=""

write_status() {
  local status="$1"
  local exit_code="${2:-}"

  python - "${STATUS_FILE}" "${status}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${SEED}" "${LOG_FILE}" "${exit_code}" "${EXPECTED_CONFIG_FINGERPRINT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, status, model, dataset, seq_len, pred_len, seed, log_path, exit_code, config_fingerprint = sys.argv[1:]
now = datetime.now(timezone.utc).isoformat()

existing = {}
if os.path.isfile(path):
    try:
        with open(path) as f:
            existing = json.load(f)
    except (OSError, ValueError):
        pass

payload = {
    "status": status,
    "model": model,
    "dataset": dataset,
    "seq_len": int(seq_len),
    "pred_len": int(pred_len),
    "seed": int(seed),
    "log": log_path,
    "config_fingerprint": config_fingerprint,
}
if status == "running":
    payload["started_at"] = now
    payload["pid"] = os.getppid()
else:
    payload["started_at"] = existing.get("started_at", now)
    timestamp_key = "completed_at" if status == "complete" else f"{status}_at"
    payload[timestamp_key] = now
if exit_code:
    payload["exit_code"] = int(exit_code)

tmp_path = f"{path}.tmp"
with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, path)
PY
}

status_is_complete() {
  python - "${STATUS_FILE}" "${EXPECTED_CONFIG_FINGERPRINT}" "${SEQ_LEN}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        status = json.load(f)
except (OSError, ValueError):
    raise SystemExit(1)

is_complete = status.get("status") == "complete"
fingerprint_matches = status.get("config_fingerprint") == sys.argv[2]
seq_len_matches = status.get("seq_len") == int(sys.argv[3])
raise SystemExit(0 if is_complete and fingerprint_matches and seq_len_matches else 1)
PY
}

resolve_config_fingerprint() {
  python main.py --print-config-fingerprint \
    SEED "${SEED}" \
    DATA.NAME "${DATASET}" \
    DATA.SEQ_LEN "${SEQ_LEN}" \
    DATA.PRED_LEN "${PRED_LEN}" \
    MODEL.NAME "${MODEL}" \
    MODEL.seq_len "${SEQ_LEN}" \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE True \
    TEST.ENABLE True \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}"
}

read_result_summary() {
  local result_dir="$1"
  python - "${result_dir}" <<'PY'
import json
import os
import sys

with open(os.path.join(sys.argv[1], "metrics.json")) as f:
    metrics = json.load(f)

print(f"test_mse={metrics['test_mse']:.8f} test_mae={metrics['test_mae']:.8f}")
PY
}

finalize_metrics() {
  local result_dir="$1"
  python - "${result_dir}" "${STATUS_FILE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

result_dir, status_path = sys.argv[1:]
metrics_path = os.path.join(result_dir, "metrics.json")

with open(metrics_path) as f:
    metrics = json.load(f)
with open(status_path) as f:
    run_status = json.load(f)

started_at = run_status.get("started_at")
if run_status.get("status") != "running" or not started_at:
    raise RuntimeError("run status does not contain a valid start time")

started = datetime.fromisoformat(started_at)
completed = datetime.now(timezone.utc)
metrics.update({
    "status": "complete",
    "started_at": started_at,
    "completed_at": completed.isoformat(),
    "duration_seconds": (completed - started).total_seconds(),
})

tmp_path = f"{metrics_path}.tmp"
with open(tmp_path, "w") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, metrics_path)
PY
}

validate_run() {
  local checkpoint_path="$1"
  local result_dir="$2"

  python - "${checkpoint_path}" "${result_dir}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${SEED}" "${EXPECTED_CONFIG_FINGERPRINT}" <<'PY'
import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml

checkpoint_path, result_dir, model, dataset, seq_len, pred_len, seed, config_fingerprint = sys.argv[1:]
if not config_fingerprint:
    raise RuntimeError("expected configuration fingerprint is empty")

required_files = (
    checkpoint_path,
    os.path.join(result_dir, "config.yaml"),
    os.path.join(result_dir, "test.txt"),
    os.path.join(result_dir, "metrics.json"),
    os.path.join(result_dir, "test_mse.npy"),
    os.path.join(result_dir, "test_mae.npy"),
    os.path.join(result_dir, "train_mse.npy"),
    os.path.join(result_dir, "train_mae.npy"),
    os.path.join(result_dir, "test_mse_all.npy"),
    os.path.join(result_dir, "test_mae_all.npy"),
)
missing = [path for path in required_files if not os.path.isfile(path) or os.path.getsize(path) == 0]
if missing:
    raise RuntimeError(f"missing or empty artifacts: {missing}")

with open(os.path.join(result_dir, "metrics.json")) as f:
    metrics = json.load(f)

config_path = os.path.join(result_dir, "config.yaml")
with open(config_path, "rb") as f:
    saved_config_fingerprint = hashlib.sha256(f.read()).hexdigest()
if saved_config_fingerprint != config_fingerprint:
    raise RuntimeError("config.yaml fingerprint does not match the requested configuration")

expected = {
    "model": model,
    "dataset": dataset,
    "seq_len": int(seq_len),
    "pred_len": int(pred_len),
    "seed": int(seed),
    "config_fingerprint": config_fingerprint,
}
for key, value in expected.items():
    if metrics.get(key) != value:
        raise RuntimeError(f"metrics identity mismatch for {key}: {metrics.get(key)!r} != {value!r}")

if metrics.get("schema_version") != 1:
    raise RuntimeError(f"unsupported metrics schema: {metrics.get('schema_version')!r}")
if metrics.get("status") != "complete":
    raise RuntimeError(f"metrics are not finalized: {metrics.get('status')!r}")
if not metrics.get("started_at") or not metrics.get("completed_at"):
    raise RuntimeError("metrics do not contain run timestamps")
if not np.isfinite(metrics.get("duration_seconds", np.nan)) or metrics["duration_seconds"] < 0:
    raise RuntimeError(f"invalid run duration: {metrics.get('duration_seconds')!r}")

for key in ("test_mse", "test_mae", "train_mse", "train_mae"):
    if key not in metrics or not np.isfinite(metrics[key]):
        raise RuntimeError(f"invalid metric {key}: {metrics.get(key)!r}")

arrays = {}
for name in ("test_mse", "test_mae", "train_mse", "train_mae", "test_mse_all", "test_mae_all"):
    arrays[name] = np.load(os.path.join(result_dir, f"{name}.npy"), allow_pickle=False)
    if arrays[name].size == 0 or not np.isfinite(arrays[name]).all():
        raise RuntimeError(f"invalid array {name}.npy")

for array_name, metric_name in (
    ("test_mse", "test_mse"),
    ("test_mae", "test_mae"),
    ("train_mse", "train_mse"),
    ("train_mae", "train_mae"),
):
    if not np.isclose(
        float(arrays[array_name].mean()), metrics[metric_name], rtol=1e-6, atol=1e-8
    ):
        raise RuntimeError(f"{array_name}.npy does not match metrics.json")
if not np.isclose(float(arrays["test_mse_all"].mean()), metrics["test_mse"], rtol=1e-6, atol=1e-8):
    raise RuntimeError("test_mse_all.npy does not match metrics.json")
if not np.isclose(float(arrays["test_mae_all"].mean()), metrics["test_mae"], rtol=1e-6, atol=1e-8):
    raise RuntimeError("test_mae_all.npy does not match metrics.json")
if metrics.get("test_samples") != int(arrays["test_mse_all"].size):
    raise RuntimeError("test sample count does not match test_mse_all.npy")
if metrics["test_samples"] != int(arrays["test_mae_all"].size):
    raise RuntimeError("test sample count does not match test_mae_all.npy")
if metrics.get("train_samples") != int(arrays["train_mse"].size):
    raise RuntimeError("train sample count does not match train_mse.npy")

test_mse_all = arrays["test_mse_all"]
for key, actual in {
    "test_mse_std": np.std(test_mse_all),
    "test_mse_min": np.min(test_mse_all),
    "test_mse_max": np.max(test_mse_all),
}.items():
    if not np.isclose(float(actual), metrics.get(key, np.nan), rtol=1e-6, atol=1e-8):
        raise RuntimeError(f"{key} does not match test_mse_all.npy")

test_mae_all = arrays["test_mae_all"]
for key, actual in {
    "test_mae_std": np.std(test_mae_all),
    "test_mae_min": np.min(test_mae_all),
    "test_mae_max": np.max(test_mae_all),
}.items():
    if not np.isclose(float(actual), metrics.get(key, np.nan), rtol=1e-6, atol=1e-8):
        raise RuntimeError(f"{key} does not match test_mae_all.npy")

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
if not isinstance(checkpoint, dict) or not checkpoint.get("model_state"):
    raise RuntimeError("checkpoint does not contain a model_state")
checkpoint_cfg = checkpoint.get("cfg")
if not isinstance(checkpoint_cfg, str):
    raise RuntimeError("checkpoint does not contain its resolved configuration")
if hashlib.sha256(checkpoint_cfg.encode("utf-8")).hexdigest() != config_fingerprint:
    raise RuntimeError("checkpoint configuration does not match the requested configuration")
try:
    parsed_checkpoint_cfg = yaml.safe_load(checkpoint_cfg)
    checkpoint_data_seq_len = parsed_checkpoint_cfg["DATA"]["SEQ_LEN"]
    checkpoint_model_seq_len = parsed_checkpoint_cfg["MODEL"]["seq_len"]
except (TypeError, KeyError, yaml.YAMLError) as error:
    raise RuntimeError("checkpoint configuration does not contain valid sequence lengths") from error
expected_seq_len = int(seq_len)
if checkpoint_data_seq_len != expected_seq_len:
    raise RuntimeError(
        f"checkpoint DATA.SEQ_LEN mismatch: {checkpoint_data_seq_len!r} != {expected_seq_len!r}"
    )
if checkpoint_model_seq_len != expected_seq_len:
    raise RuntimeError(
        f"checkpoint MODEL.seq_len mismatch: {checkpoint_model_seq_len!r} != {expected_seq_len!r}"
    )

checkpoint_selection = checkpoint.get("selection_metric")
if not isinstance(checkpoint_selection, str):
    raise RuntimeError("checkpoint does not identify its selection metric")
expected_selection = checkpoint_selection
if expected_selection != "closed_form_fit":
    expected_selection = f"val_{expected_selection.lower()}"
if metrics.get("selection_metric") != expected_selection:
    raise RuntimeError("selection metric does not match the checkpoint")
if metrics.get("best_epoch") != checkpoint.get("epoch"):
    raise RuntimeError("best epoch does not match the checkpoint")
if os.path.normpath(metrics.get("checkpoint", "")) != os.path.normpath(checkpoint_path):
    raise RuntimeError("checkpoint path in metrics does not match the validated checkpoint")

checkpoint_best_metric = checkpoint.get("best_metric")
metrics_best_metric = metrics.get("best_val_metric")
if checkpoint_best_metric is None and metrics_best_metric is not None:
    raise RuntimeError("best validation metric does not match the checkpoint")
if checkpoint_best_metric is not None and (
    metrics_best_metric is None or not np.isclose(metrics_best_metric, checkpoint_best_metric)
):
    raise RuntimeError("best validation metric does not match the checkpoint")

validation_metrics = checkpoint.get("validation_metrics", {})
for key, checkpoint_key in (("best_val_mse", "MSE"), ("best_val_mae", "MAE")):
    expected_value = validation_metrics.get(checkpoint_key)
    actual_value = metrics.get(key)
    if expected_value is None and actual_value is None:
        continue
    if expected_value is None or actual_value is None or not np.isclose(actual_value, expected_value):
        raise RuntimeError(f"{key} does not match the checkpoint")
PY
}

handle_interrupt() {
  local exit_code="$1"
  if ((CURRENT_ACTIVE)); then
    write_status "interrupted" "${exit_code}" || true
  fi
  exit "${exit_code}"
}

trap 'handle_interrupt 130' INT
trap 'handle_interrupt 143' TERM

TOTAL=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#INPUT_LENS[@]} * ${#PRED_LENS[@]}))
RUN_NUMBER=0

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SEQ_LEN in "${INPUT_LENS[@]}"; do
      for PRED_LEN in "${PRED_LENS[@]}"; do
      RUN_NUMBER=$((RUN_NUMBER + 1))
      RUN_ID="${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${RUN_ID}/seed_${SEED}/"
      CHECKPOINT_PATH="${CHECKPOINT_DIR}checkpoint_best.pth"
      RESULT_DIR="./results/${MODEL}/${RUN_ID}/seed_${SEED}/"
      LOG_DIR="./logs/train/${MODEL}"
      LOG_FILE="${LOG_DIR}/${RUN_ID}_seed_${SEED}.log"
      STATUS_FILE="${RESULT_DIR}run_status.json"

      mkdir -p "${CHECKPOINT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

      if ! EXPECTED_CONFIG_FINGERPRINT="$(resolve_config_fingerprint 2>>"${LOG_FILE}")"; then
        CURRENT_ACTIVE=1
        write_status "failed" "1"
        CURRENT_ACTIVE=0
        FAILED=$((FAILED + 1))
        printf '[%d/%d] FAIL model=%s dataset=%s seq_len=%s pred_len=%s (could not resolve configuration; log=%s)\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${LOG_FILE}"
        continue
      fi

      if [[ "${FORCE}" != "1" ]] && status_is_complete && validate_run "${CHECKPOINT_PATH}" "${RESULT_DIR}"; then
        result_summary="$(read_result_summary "${RESULT_DIR}")"
        printf '[%d/%d] SKIP model=%s dataset=%s seq_len=%s pred_len=%s %s (validated complete)\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${result_summary}"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      printf '\n[%d/%d] START model=%s dataset=%s seq_len=%s pred_len=%s seed=%s time=%s\n' \
        "${RUN_NUMBER}" "${TOTAL}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${SEED}" "$(date -Is)" | tee -a "${LOG_FILE}"

      CURRENT_ACTIVE=1
      write_status "running"

      if python main.py \
        SEED "${SEED}" \
        DATA.NAME "${DATASET}" \
        DATA.SEQ_LEN "${SEQ_LEN}" \
        DATA.PRED_LEN "${PRED_LEN}" \
        MODEL.NAME "${MODEL}" \
        MODEL.seq_len "${SEQ_LEN}" \
        MODEL.pred_len "${PRED_LEN}" \
        TRAIN.ENABLE True \
        TEST.ENABLE True \
        TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" 2>&1 | tee -a "${LOG_FILE}"; then
        if finalize_metrics "${RESULT_DIR}" 2>&1 | tee -a "${LOG_FILE}" && \
          validate_run "${CHECKPOINT_PATH}" "${RESULT_DIR}" 2>&1 | tee -a "${LOG_FILE}"; then
          write_status "complete"
          COMPLETED=$((COMPLETED + 1))
          result_summary="$(read_result_summary "${RESULT_DIR}")"
          printf '[%d/%d] PASS model=%s dataset=%s seq_len=%s pred_len=%s %s\n' \
            "${RUN_NUMBER}" "${TOTAL}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${result_summary}" | tee -a "${LOG_FILE}"
        else
          write_status "failed" "1"
          FAILED=$((FAILED + 1))
          printf '[%d/%d] FAIL model=%s dataset=%s seq_len=%s pred_len=%s (artifact validation failed; log=%s)\n' \
            "${RUN_NUMBER}" "${TOTAL}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${LOG_FILE}"
        fi
      else
        exit_code="$?"
        write_status "failed" "${exit_code}"
        FAILED=$((FAILED + 1))
        printf '[%d/%d] FAIL model=%s dataset=%s seq_len=%s pred_len=%s exit_code=%s log=%s\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${exit_code}" "${LOG_FILE}"
      fi

      CURRENT_ACTIVE=0
      done
    done
  done
done

printf '\nTraining summary: completed=%d skipped=%d failed=%d total=%d\n' \
  "${COMPLETED}" "${SKIPPED}" "${FAILED}" "${TOTAL}"

if ((FAILED > 0)); then
  exit 1
fi
