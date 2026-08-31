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

METHOD="${TTA_METHOD:?TTA_METHOD must be TAFAS or PETSA}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
PAAS="${PAAS:-True}"
PERIOD_N="${PERIOD_N:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS="${STEPS:-1}"
ADJUST_PRED="${ADJUST_PRED:-True}"
GATING_INIT="${GATING_INIT:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
GCM_VAR_WISE="${GCM_VAR_WISE:-True}"
MODULE_NAMES_TO_ADAPT="${MODULE_NAMES_TO_ADAPT:-cali}"
BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
RANK="${RANK:-16}"
LOSS_ALPHA="${LOSS_ALPHA:-0.1}"

COMPLETED=0
SKIPPED=0
FAILED=0
CURRENT_ACTIVE=0
MODEL=""
DATASET=""
SEQ_LEN=""
PRED_LEN=""
CHECKPOINT_DIR=""
CHECKPOINT_PATH=""
RESULT_DIR=""
METRICS_PATH=""
STATUS_FILE=""
LOG_FILE=""
SUMMARY_FILE=""
RAW_OUTPUT=""
EXPECTED_CONFIG_FINGERPRINT=""
EXPECTED_CHECKPOINT_SHA256=""
RUN_OPTS=()

validate_settings() {
  python - "${METHOD}" "${SEED}" "${FORCE}" "${PERIOD_N}" "${BATCH_SIZE}" "${STEPS}" \
    "${PAAS}" "${ADJUST_PRED}" "${GCM_VAR_WISE}" "${GATING_INIT}" \
    "${HIDDEN_DIM}" "${BASE_LR}" "${WEIGHT_DECAY}" "${MODULE_NAMES_TO_ADAPT}" \
    "${RANK}" "${LOSS_ALPHA}" <<'PY'
import math
import sys

(
    method, seed, force, period_n, batch_size, steps, paas, adjust_pred,
    gcm_var_wise, gating_init, hidden_dim, base_lr, weight_decay,
    module_names, rank, loss_alpha,
) = sys.argv[1:]

def integer(name, value, minimum):
    try:
        parsed = int(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer: {value!r}") from error
    if parsed < minimum:
        raise SystemExit(f"{name} must be >= {minimum}: {parsed}")

def number(name, value, minimum=0.0):
    try:
        parsed = float(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be numeric: {value!r}") from error
    if not math.isfinite(parsed) or parsed < minimum:
        raise SystemExit(f"{name} must be finite and >= {minimum}: {value!r}")

integer("SEED", seed, 0)
integer("PERIOD_N", period_n, 1)
integer("BATCH_SIZE", batch_size, 1)
integer("STEPS", steps, 1)
integer("HIDDEN_DIM", hidden_dim, 1)
if method not in {"TAFAS", "PETSA"}:
    raise SystemExit("TTA_METHOD must be TAFAS or PETSA")
if force not in {"0", "1"}:
    raise SystemExit("FORCE must be 0 or 1")
for name, value in {
    "PAAS": paas,
    "ADJUST_PRED": adjust_pred,
    "GCM_VAR_WISE": gcm_var_wise,
}.items():
    if value not in {"True", "False"}:
        raise SystemExit(f"{name} must be True or False")
number("GATING_INIT", gating_init)
number("BASE_LR", base_lr)
number("WEIGHT_DECAY", weight_decay)
if not module_names.strip():
    raise SystemExit("MODULE_NAMES_TO_ADAPT must not be empty")
if method == "PETSA":
    integer("RANK", rank, 1)
    number("LOSS_ALPHA", loss_alpha)
PY
}

write_status() {
  local status="$1"
  local exit_code="${2:-}"

  python - "${STATUS_FILE}" "${status}" "${METHOD}" "${MODEL}" "${DATASET}" \
    "${SEQ_LEN}" "${PRED_LEN}" "${SEED}" "${LOG_FILE}" "${exit_code}" \
    "${EXPECTED_CONFIG_FINGERPRINT}" "${CHECKPOINT_PATH}" \
    "${EXPECTED_CHECKPOINT_SHA256}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    path, status, method, model, dataset, seq_len, pred_len, seed, log_path,
    exit_code, config_fingerprint, checkpoint_path, checkpoint_sha256,
) = sys.argv[1:]
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
    "method": method,
    "model": model,
    "dataset": dataset,
    "seq_len": int(seq_len),
    "pred_len": int(pred_len),
    "seed": int(seed),
    "log": log_path,
    "config_fingerprint": config_fingerprint,
    "checkpoint": os.path.normpath(checkpoint_path),
    "checkpoint_sha256": checkpoint_sha256,
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
  python - "${STATUS_FILE}" "${EXPECTED_CONFIG_FINGERPRINT}" \
    "${EXPECTED_CHECKPOINT_SHA256}" "${METHOD}" "${SEQ_LEN}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        status = json.load(f)
except (OSError, ValueError):
    raise SystemExit(1)

valid = (
    status.get("status") == "complete"
    and status.get("config_fingerprint") == sys.argv[2]
    and status.get("checkpoint_sha256") == sys.argv[3]
    and status.get("method") == sys.argv[4]
    and status.get("seq_len") == int(sys.argv[5])
)
raise SystemExit(0 if valid else 1)
PY
}

resolve_config_fingerprint() {
  python main.py --print-config-fingerprint "${RUN_OPTS[@]}"
}

resolve_training_fingerprint() {
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

validate_checkpoint() {
  local training_fingerprint="$1"
  python - "${CHECKPOINT_PATH}" "${training_fingerprint}" "${MODEL}" "${DATASET}" \
    "${SEQ_LEN}" "${PRED_LEN}" "${SEED}" <<'PY'
import hashlib
import os
import sys

import torch
import yaml

path, expected_fingerprint, model, dataset, seq_len, pred_len, seed = sys.argv[1:]
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    raise RuntimeError(f"missing or empty checkpoint: {path}")

try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
if not isinstance(checkpoint, dict) or not checkpoint.get("model_state"):
    raise RuntimeError("checkpoint does not contain a model_state")
checkpoint_cfg = checkpoint.get("cfg")
if not isinstance(checkpoint_cfg, str):
    raise RuntimeError("checkpoint does not contain its resolved configuration")
if hashlib.sha256(checkpoint_cfg.encode("utf-8")).hexdigest() != expected_fingerprint:
    raise RuntimeError("checkpoint configuration does not match the requested training configuration")

cfg = yaml.safe_load(checkpoint_cfg)
expected = {
    "MODEL.NAME": model,
    "DATA.NAME": dataset,
    "DATA.SEQ_LEN": int(seq_len),
    "DATA.PRED_LEN": int(pred_len),
    "MODEL.seq_len": int(seq_len),
    "SEED": int(seed),
}
actual = {
    "MODEL.NAME": cfg["MODEL"]["NAME"],
    "DATA.NAME": cfg["DATA"]["NAME"],
    "DATA.SEQ_LEN": cfg["DATA"]["SEQ_LEN"],
    "DATA.PRED_LEN": cfg["DATA"]["PRED_LEN"],
    "MODEL.seq_len": cfg["MODEL"]["seq_len"],
    "SEED": cfg["SEED"],
}
for key, value in expected.items():
    if actual[key] != value:
        raise RuntimeError(f"checkpoint identity mismatch for {key}: {actual[key]!r} != {value!r}")

digest = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

finalize_metrics() {
  python - "${RAW_OUTPUT}" "${METRICS_PATH}" "${STATUS_FILE}" "${METHOD}" \
    "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${SEED}" \
    "${EXPECTED_CONFIG_FINGERPRINT}" "${CHECKPOINT_PATH}" \
    "${EXPECTED_CHECKPOINT_SHA256}" <<'PY'
import json
import math
import os
import sys
from datetime import datetime, timezone

(
    output_path, metrics_path, status_path, method, model, dataset,
    seq_len, pred_len, seed, config_fingerprint, checkpoint_path, checkpoint_sha256,
) = sys.argv[1:]

with open(output_path) as f:
    output = f.read()
decoder = json.JSONDecoder()
adapter_result = None
for index, character in enumerate(output):
    if character != "{":
        continue
    try:
        value, _ = decoder.raw_decode(output[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and value.get("model") == method and "final_results" in value:
        adapter_result = value
if adapter_result is None:
    raise RuntimeError(f"{method} output did not contain its final JSON result")

results = adapter_result.get("final_results", {})
mse = results.get("test_mse")
adaptations = results.get("adaptation_count")
params = adapter_result.get("parameters", {}).get("total_params")
if type(mse) not in (int, float) or not math.isfinite(mse) or mse < 0:
    raise RuntimeError(f"invalid test MSE: {mse!r}")
if type(adaptations) is not int or adaptations < 0:
    raise RuntimeError(f"invalid adaptation count: {adaptations!r}")
if type(params) is not int or params < 0:
    raise RuntimeError(f"invalid trainable parameter count: {params!r}")

with open(status_path) as f:
    status = json.load(f)
started_at = status.get("started_at")
if status.get("status") != "running" or not started_at:
    raise RuntimeError("run status does not contain a valid start time")
started = datetime.fromisoformat(started_at)
completed = datetime.now(timezone.utc)

adapter_result.update({
    "schema_version": 1,
    "status": "complete",
    "method": method,
    "model": model,
    "dataset": dataset,
    "seq_len": int(seq_len),
    "pred_len": int(pred_len),
    "seed": int(seed),
    "config_fingerprint": config_fingerprint,
    "checkpoint": os.path.normpath(checkpoint_path),
    "checkpoint_sha256": checkpoint_sha256,
    "started_at": started_at,
    "completed_at": completed.isoformat(),
    "duration_seconds": (completed - started).total_seconds(),
})

tmp_path = f"{metrics_path}.tmp"
with open(tmp_path, "w") as f:
    json.dump(adapter_result, f, indent=2, sort_keys=True)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, metrics_path)
PY
}

validate_run() {
  python - "${METRICS_PATH}" "${RESULT_DIR}/config.yaml" \
    "${EXPECTED_CONFIG_FINGERPRINT}" "${CHECKPOINT_PATH}" \
    "${EXPECTED_CHECKPOINT_SHA256}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" \
    "${SEED}" "${METHOD}" <<'PY'
import hashlib
import json
import math
import os
import sys

(
    metrics_path, config_path, config_fingerprint, checkpoint_path,
    checkpoint_sha256, model, dataset, seq_len, pred_len, seed, method,
) = sys.argv[1:]
for path in (metrics_path, config_path, checkpoint_path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"missing or empty artifact: {path}")
with open(config_path, "rb") as f:
    if hashlib.sha256(f.read()).hexdigest() != config_fingerprint:
        raise RuntimeError("config.yaml fingerprint does not match the requested configuration")
with open(metrics_path) as f:
    metrics = json.load(f)

expected = {
    "schema_version": 1,
    "status": "complete",
    "method": method,
    "model": model,
    "dataset": dataset,
    "seq_len": int(seq_len),
    "pred_len": int(pred_len),
    "seed": int(seed),
    "config_fingerprint": config_fingerprint,
    "checkpoint": os.path.normpath(checkpoint_path),
    "checkpoint_sha256": checkpoint_sha256,
}
for key, value in expected.items():
    if metrics.get(key) != value:
        raise RuntimeError(f"metrics identity mismatch for {key}: {metrics.get(key)!r} != {value!r}")
if not metrics.get("started_at") or not metrics.get("completed_at"):
    raise RuntimeError("metrics do not contain run timestamps")
duration = metrics.get("duration_seconds")
if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
    raise RuntimeError(f"invalid duration: {duration!r}")
results = metrics.get("final_results", {})
for key in ("test_mse",):
    value = results.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid metric {key}: {value!r}")
adaptations = results.get("adaptation_count")
if type(adaptations) is not int or adaptations < 0:
    raise RuntimeError(f"invalid adaptation count: {adaptations!r}")
params = metrics.get("parameters", {}).get("total_params")
if type(params) is not int or params < 0:
    raise RuntimeError(f"invalid trainable parameter count: {params!r}")
overall = metrics.get("time_statistics", {}).get("overall_stats", {})
for key in ("total_time_seconds", "throughput_samples_per_sec"):
    value = overall.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid timing metric {key}: {value!r}")
PY
}

write_summary() {
  python - "${METRICS_PATH}" "${SUMMARY_FILE}" <<'PY'
import json
import os
import sys

with open(sys.argv[1]) as f:
    metrics = json.load(f)
results = metrics["final_results"]
summary = (
    f"test_mse={results['test_mse']:.8f} "
    f"adaptations={results['adaptation_count']} "
    f"duration_seconds={metrics['duration_seconds']:.3f}\n"
)
tmp_path = f"{sys.argv[2]}.tmp"
with open(tmp_path, "w") as f:
    f.write(summary)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, sys.argv[2])
print(summary, end="")
PY
}

handle_interrupt() {
  local exit_code="$1"
  if ((CURRENT_ACTIVE)); then
    write_status "interrupted" "${exit_code}" || true
    CURRENT_ACTIVE=0
  fi
  exit "${exit_code}"
}

handle_exit() {
  local exit_code="$?"
  if ((exit_code != 0 && CURRENT_ACTIVE)); then
    write_status "failed" "${exit_code}" || true
  fi
}

trap 'handle_interrupt 130' INT
trap 'handle_interrupt 143' TERM
trap handle_exit EXIT

validate_settings
python -c 'import torch, yacs' >/dev/null

TOTAL=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#INPUT_LENS[@]} * ${#PRED_LENS[@]}))
RUN_NUMBER=0

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SEQ_LEN in "${INPUT_LENS[@]}"; do
      for PRED_LEN in "${PRED_LENS[@]}"; do
      RUN_NUMBER=$((RUN_NUMBER + 1))
      RUN_KEY="${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${RUN_KEY}/seed_${SEED}/"
      CHECKPOINT_PATH="${CHECKPOINT_DIR}checkpoint_best.pth"
      RESULT_ROOT="./results/${METHOD}/"
      RESULT_DIR="./results/${METHOD}/${MODEL}/${RUN_KEY}/seed_${SEED}"
      METRICS_PATH="${RESULT_DIR}/metrics.json"
      STATUS_FILE="${RESULT_DIR}/run_status.json"
      LOG_DIR="./logs/${METHOD,,}/${MODEL}"
      LOG_FILE="${LOG_DIR}/${RUN_KEY}_seed_${SEED}.log"
      SUMMARY_DIR="./results/summary/${METHOD}/${MODEL}/${RUN_KEY}"
      SUMMARY_FILE="${SUMMARY_DIR}/seed_${SEED}.txt"
      RAW_OUTPUT="${RESULT_DIR}/adapter_output.tmp"

      mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}"

      RUN_OPTS=(
        SEED "${SEED}"
        DATA.NAME "${DATASET}"
        DATA.SEQ_LEN "${SEQ_LEN}"
        DATA.PRED_LEN "${PRED_LEN}"
        MODEL.NAME "${MODEL}"
        MODEL.seq_len "${SEQ_LEN}"
        MODEL.pred_len "${PRED_LEN}"
        TRAIN.ENABLE False
        TEST.ENABLE False
        TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}"
        TTA.ENABLE True
        TTA.MODULE_NAMES_TO_ADAPT "${MODULE_NAMES_TO_ADAPT}"
        TTA.SOLVER.BASE_LR "${BASE_LR}"
        TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}"
      )

      if [[ "${METHOD}" == "TAFAS" ]]; then
        RUN_OPTS+=(
          TTA.TAFAS.PAAS "${PAAS}"
          TTA.TAFAS.PERIOD_N "${PERIOD_N}"
          TTA.TAFAS.BATCH_SIZE "${BATCH_SIZE}"
          TTA.TAFAS.STEPS "${STEPS}"
          TTA.TAFAS.ADJUST_PRED "${ADJUST_PRED}"
          TTA.TAFAS.CALI_MODULE True
          TTA.TAFAS.GATING_INIT "${GATING_INIT}"
          TTA.TAFAS.HIDDEN_DIM "${HIDDEN_DIM}"
          TTA.TAFAS.GCM_VAR_WISE "${GCM_VAR_WISE}"
        )
      else
        RUN_OPTS+=(
          TTA.PETSA.PAAS "${PAAS}"
          TTA.PETSA.PERIOD_N "${PERIOD_N}"
          TTA.PETSA.BATCH_SIZE "${BATCH_SIZE}"
          TTA.PETSA.STEPS "${STEPS}"
          TTA.PETSA.ADJUST_PRED "${ADJUST_PRED}"
          TTA.PETSA.CALI_MODULE True
          TTA.PETSA.GATING_INIT "${GATING_INIT}"
          TTA.PETSA.HIDDEN_DIM "${HIDDEN_DIM}"
          TTA.PETSA.GCM_VAR_WISE "${GCM_VAR_WISE}"
          TTA.PETSA.RANK "${RANK}"
          TTA.PETSA.LOSS_ALPHA "${LOSS_ALPHA}"
        )
      fi
      RUN_OPTS+=(RESULT_DIR "${RESULT_ROOT}")

      EXPECTED_CONFIG_FINGERPRINT=""
      EXPECTED_CHECKPOINT_SHA256=""
      if ! EXPECTED_CONFIG_FINGERPRINT="$(resolve_config_fingerprint 2>>"${LOG_FILE}")"; then
        CURRENT_ACTIVE=1
        write_status "failed" "1"
        CURRENT_ACTIVE=0
        FAILED=$((FAILED + 1))
        printf '[%d/%d] FAIL method=%s model=%s dataset=%s seq_len=%s pred_len=%s (configuration; log=%s)\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${LOG_FILE}"
        continue
      fi
      if ! TRAINING_CONFIG_FINGERPRINT="$(resolve_training_fingerprint 2>>"${LOG_FILE}")"; then
        CURRENT_ACTIVE=1
        write_status "failed" "1"
        CURRENT_ACTIVE=0
        FAILED=$((FAILED + 1))
        printf '[%d/%d] FAIL method=%s model=%s dataset=%s seq_len=%s pred_len=%s (training configuration; log=%s)\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${LOG_FILE}"
        continue
      fi
      if ! EXPECTED_CHECKPOINT_SHA256="$(validate_checkpoint "${TRAINING_CONFIG_FINGERPRINT}" 2>>"${LOG_FILE}")"; then
        CURRENT_ACTIVE=1
        write_status "failed" "1"
        CURRENT_ACTIVE=0
        FAILED=$((FAILED + 1))
        printf '[%d/%d] FAIL method=%s model=%s dataset=%s seq_len=%s pred_len=%s (checkpoint; log=%s)\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${LOG_FILE}"
        continue
      fi

      if [[ "${FORCE}" != "1" ]] && status_is_complete && validate_run; then
        result_summary="$(write_summary 2>>"${LOG_FILE}")"
        printf '[%d/%d] SKIP method=%s model=%s dataset=%s seq_len=%s pred_len=%s %s (validated complete)\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" "${result_summary}"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      printf '\n[%d/%d] START method=%s model=%s dataset=%s seq_len=%s pred_len=%s seed=%s time=%s\n' \
        "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" \
        "${SEED}" "$(date -Is)" | tee -a "${LOG_FILE}"

      CURRENT_ACTIVE=1
      write_status "running"
      rm -f "${METRICS_PATH}" "${METRICS_PATH}.tmp" "${SUMMARY_FILE}" \
        "${SUMMARY_FILE}.tmp" "${RAW_OUTPUT}"

      if python main.py "${RUN_OPTS[@]}" 2>&1 | tee -a "${LOG_FILE}" | tee "${RAW_OUTPUT}"; then
        if finalize_metrics 2>&1 | tee -a "${LOG_FILE}" && \
          validate_run 2>&1 | tee -a "${LOG_FILE}" && \
          result_summary="$(write_summary 2>>"${LOG_FILE}")"; then
          write_status "complete"
          rm -f "${RAW_OUTPUT}"
          COMPLETED=$((COMPLETED + 1))
          printf '[%d/%d] PASS method=%s model=%s dataset=%s seq_len=%s pred_len=%s %s\n' \
            "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" \
            "${result_summary}" | tee -a "${LOG_FILE}"
        else
          write_status "failed" "1"
          FAILED=$((FAILED + 1))
          printf '[%d/%d] FAIL method=%s model=%s dataset=%s seq_len=%s pred_len=%s (artifact validation; log=%s)\n' \
            "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" \
            "${LOG_FILE}"
        fi
      else
        exit_code="$?"
        write_status "failed" "${exit_code}"
        FAILED=$((FAILED + 1))
        printf '[%d/%d] FAIL method=%s model=%s dataset=%s seq_len=%s pred_len=%s exit_code=%s log=%s\n' \
          "${RUN_NUMBER}" "${TOTAL}" "${METHOD}" "${MODEL}" "${DATASET}" "${SEQ_LEN}" "${PRED_LEN}" \
          "${exit_code}" "${LOG_FILE}"
      fi
      CURRENT_ACTIVE=0
      done
    done
  done
done

printf '\n%s summary: completed=%d skipped=%d failed=%d total=%d\n' \
  "${METHOD}" "${COMPLETED}" "${SKIPPED}" "${FAILED}" "${TOTAL}"

if ((FAILED > 0)); then
  exit 1
fi
