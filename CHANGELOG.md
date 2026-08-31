# Change Log

## 2026-08-31 - COSA Sensitivity Sweeps

### Sensitivity launcher

- Added the independent `scripts/sensitivity.sh` launcher for one-factor-at-a-time COSA-F TTA experiments.
- Added default sweeps for adaptation steps (`1`, `2`, `3`, `4`), buffer context size (`0`, `10`, `20`), fixed batch size (`24`, `48`, `96`), and adaptive learning rate (`False`, `True`).
- Kept the remaining COSA settings fixed to the `scripts/cosa.sh` defaults for each sensitivity dimension.
- Default to input length 96 across all configured models, datasets, and prediction lengths, with environment filters for experiment dimensions, models, datasets, input lengths, prediction lengths, and sweep values.
- Run COSA-F with `PAAS=False` so the configured batch-size sweep is applied rather than replaced by PAAS batch-size selection.

### Checkpoints and artifacts

- Reuse pretrained base models with training and testing disabled during sensitivity runs.
- Resolve both current input-length-aware checkpoint paths and legacy `<dataset>_<pred_len>` paths for input length 96, while requiring exact embedded training-configuration and experiment-identity matches.
- Store results under `results/SIMPLE/sensitivity/COSA-F/<dimension>/<value>/`, with matching sensitivity log and summary trees.
- Record the sensitivity dimension and value in status and canonical metrics artifacts, and retain configuration fingerprint, checkpoint hash, restart-safe skip, interruption handling, and artifact validation checks.

### Verification

- Bash syntax validation passed for `scripts/sensitivity.sh`.
- Completed and validated an OLS/ETTh1 test run for the `steps=1` condition using a legacy pretrained checkpoint.
- Verified that rerunning the completed condition is detected and skipped safely.
- Completed and validated the zero-context (`context=0`) condition.
- `shellcheck` verification was not run because `shellcheck` is not installed in the environment.

## 2026-08-31 - Input-Length Experiment Support

### Experiment matrix and configuration

- Added input-length sweeps for `96`, `192`, `336`, and `720` to baseline training, COSA, TAFAS, and PETSA launchers.
- Expanded each method's experiment matrix to 576 model, dataset, input-length, and prediction-length combinations per seed.
- Explicitly pass both `DATA.SEQ_LEN` and `MODEL.seq_len` so data windows and model dimensions remain synchronized.
- Added a validated `INPUT_LEN` environment override for running a single input length while preserving the full sweep by default.
- Updated `scripts/exp1.sh` to run training, COSA-F, COSA-P, TAFAS, and PETSA sequentially with input length 96.

### Artifact identity and validation

- Added input length to checkpoint, result, log, and summary identities using `<dataset>_sl<seq_len>_pl<pred_len>`.
- Added `seq_len` to `run_status.json` and method metrics, including COSA's canonical `metrics.json`.
- Extended completion checks and artifact validation to reject results whose input length does not match the requested experiment.
- Extended checkpoint validation to verify both `DATA.SEQ_LEN` and `MODEL.seq_len` in the embedded training configuration.
- Updated COSA prediction and PAAS CSV filenames to include input and prediction lengths.
- Existing checkpoints under the old `<dataset>_<pred_len>` layout are intentionally not reused because model parameters can depend on input length.

### Shared TTA orchestration

- Added `scripts/tta.sh` as the shared robust runner used by `scripts/tafas.sh` and `scripts/petsa.sh`.
- Added strict error handling, repository-root resolution, configuration validation, checkpoint provenance checks, atomic status and metrics artifacts, restart-safe skipping, per-run logs, and aggregate run reporting for TAFAS and PETSA.
- Explicitly disable the separate baseline predictor during TAFAS and PETSA adaptation with `TEST.ENABLE False`.

### Verification

- Bash syntax validation passed for `scripts/train.sh`, `scripts/cosa.sh`, `scripts/tta.sh`, `scripts/tafas.sh`, and `scripts/petsa.sh`.
- Python compilation passed for `tta/cosa.py`.
- Configuration fingerprint resolution passed for training and TTA with a non-default input length of 192.
- Verified that all 576 generated run keys are unique.
- Verified nonempty train, validation, and test splits for all configured datasets at input length 720 and prediction length 720.

## 2026-08-31 - COSA Execution Robustness

### COSA orchestration

- Rebuilt `scripts/cosa.sh` with strict Bash error handling, repository-root resolution, and configurable CUDA device selection.
- Added environment overrides for `SEED`, `FORCE`, batch size, adaptation steps, context size, PAAS, and learning-rate settings.
- Corrected COSA optimizer overrides to use `TTA.SOLVER.BASE_LR` and `TTA.SOLVER.WEIGHT_DECAY`.
- Explicitly disabled the baseline predictor with `TEST.ENABLE False` during COSA adaptation.
- Added per-experiment logs under `logs/cosa/<model>/` and atomic summaries under `results/summary/SIMPLE/`.
- Added separate COSA-P (`PAAS=True`) and COSA-F (`PAAS=False`) identities and artifact trees.
- Stored COSA-P results under `results/SIMPLE/COSA-P/` and COSA-F results under `results/SIMPLE/COSA-F/`, with matching summary and log separation.
- Added completed, skipped, and failed counters with a nonzero final exit status when any experiment fails.
- Added `SIGINT`, `SIGTERM`, and abnormal-exit handling so active runs are not left silently incomplete.

### Restart and artifact safety

- Added atomic `run_status.json` records for running, complete, failed, and interrupted runs.
- Added resolved COSA configuration fingerprints and source-training configuration validation.
- Added checkpoint preflight validation for checkpoint existence, model state, embedded configuration, experiment identity, and SHA-256 content hash.
- Changed skip behavior to require matching status, configuration fingerprint, checkpoint hash, and validated COSA artifacts.
- Added fail-fast validation for integer, boolean, and learning-rate settings.
- Removed stale metrics and summary files before rerunning an experiment.
- Added atomic finalization with start time, completion time, and duration metadata.

### Structured COSA metrics

- Updated `tta/cosa.py` to write an atomic `metrics.json` only after adaptation completes.
- Added schema version, method, model, dataset, prediction length, seed, test sample count, configuration fingerprint, checkpoint path, and checkpoint hash metadata.
- Recorded `COSA-P` or `COSA-F` as the metrics method according to the resolved PAAS setting.
- Added full-precision test MSE and MAE with standard deviation, minimum, and maximum values.
- Added adaptation count, trainable parameter count, and detailed timing statistics.
- Preserved the human-readable JSON output on stdout while making `metrics.json` the canonical completion artifact.

### COSA variants

- Defined COSA-P as COSA with PAAS enabled and COSA-F as COSA with PAAS disabled.
- Kept variant execution separate so each invocation runs only the requested PAAS mode:

  ```bash
  PAAS=True mamba run -n cosa bash scripts/cosa.sh   # COSA-P
  PAAS=False mamba run -n cosa bash scripts/cosa.sh  # COSA-F
  ```

- Isolated variant artifacts under the following roots:

  ```text
  results/SIMPLE/COSA-P/
  results/SIMPLE/COSA-F/
  results/summary/SIMPLE/COSA-P/
  results/summary/SIMPLE/COSA-F/
  logs/cosa/COSA-P/
  logs/cosa/COSA-F/
  ```
- Updated status and metrics validation to reject artifacts whose COSA-P/COSA-F identity does not match the selected PAAS mode.
- Updated optional CSV export directories to use the matching COSA-P or COSA-F identity.

### Verification

- Bash syntax validation passed for `scripts/cosa.sh`.
- Python compilation passed for `tta/cosa.py`.
- `git diff --check` passed.
- The `cosa` Mamba environment successfully imported all required packages with PyTorch 2.4.0 and detected one CUDA GPU.
- All six configured dataset CSV files were found and loaded successfully.
- COSA configuration fingerprint resolution succeeded for the first experiment configuration.
- COSA-P and COSA-F resolved to distinct configuration fingerprints.
- Full COSA execution was not launched because trained baseline checkpoints are not currently present.
- ShellCheck was not available in the Mamba environment.

## 2026-08-31 - Baseline Training Robustness

### Training orchestration

- Updated `scripts/train.sh` to use strict Bash error handling with `set -Eeuo pipefail`.
- Made the script resolve and enter the repository root before using relative paths.
- Added environment overrides for `CUDA_VISIBLE_DEVICES`, `SEED`, and `FORCE`.
- Added per-experiment logs under `logs/train/<model>/`.
- Added `run_status.json` records for `running`, `complete`, `failed`, and `interrupted` states.
- Added handling for `SIGINT` and `SIGTERM` so interrupted runs are recorded.
- Allowed failed experiments to be recorded while remaining combinations continue.
- Added a final completed, skipped, and failed run summary.
- Explicitly enabled baseline testing with `TEST.ENABLE True`.

### Safe skipping and result validation

- Added a SHA-256 fingerprint of the fully resolved configuration.
- Added `--print-config-fingerprint` to `main.py` for no-training fingerprint resolution.
- Stored the configuration fingerprint in `metrics.json` and `run_status.json`.
- Changed skip behavior to require a completed status and successful artifact validation.
- Added validation for model, dataset, prediction length, seed, configuration, metrics, NumPy arrays, and checkpoint contents.
- Added consistency checks between aggregate metrics and per-window arrays.
- Added consistency checks between `metrics.json`, `config.yaml`, and checkpoint metadata.
- Added seed-specific checkpoint and result directories:

  ```text
  checkpoints/<model>/<dataset>_<pred_len>/seed_<seed>/
  results/<model>/<dataset>_<pred_len>/seed_<seed>/
  ```

- Updated `scripts/cosa.sh`, `scripts/tafas.sh`, `scripts/petsa.sh`, and `scripts/dynatta.sh` to load seed-specific checkpoints and use seed-specific summary filenames.
- Added test MSE and MAE to pass and skip messages.

### Atomic artifact writes

- Made `config.yaml` writes atomic in `main.py`.
- Made model and normalization checkpoint writes atomic in `trainer.py`.
- Made `best_result.txt` writes atomic in `trainer.py`.
- Made text, JSON, and NumPy result writes atomic in `predictor.py`.
- Atomic writes use a temporary file, flush and `fsync`, followed by `os.replace`.

### Structured metrics

- Added versioned, full-precision `metrics.json` output.
- Added model, dataset, sequence length, prediction length, seed, scaling, and normalization metadata.
- Added checkpoint path and configuration fingerprint metadata.
- Added checkpoint selection metric, best epoch, selected validation metric, validation MSE, and validation MAE.
- Added full-precision train and test MSE and MAE.
- Added train and test sample counts.
- Added test MSE and MAE standard deviation, minimum, and maximum.
- Added run start time, completion time, and duration when executed through `scripts/train.sh`.
- Added `schema_version: 1` for future result-format compatibility.
- Added `test_mae_all.npy` alongside `test_mse_all.npy`.
- Increased readable precision in `test.txt` from four to eight decimal places.
- Added final baseline metric output to the terminal and per-run log.

### Checkpoint metadata

- Kept the existing validation-MAE checkpoint selection behavior unchanged.
- Added both validation MSE and validation MAE from the selected epoch to checkpoints.
- Added the selected metric and selected value to checkpoints.
- Exposed loaded checkpoint metadata to `Predictor` without reloading the checkpoint.
- Labeled OLS checkpoints as `closed_form_fit` with no validation-selected epoch.

### Configuration integrity

- Prevented `Predictor` from mutating the shared experiment configuration when disabling train-loader shuffling and dropped batches for evaluation.
- Added a reusable `get_config_fingerprint` function in `config.py`.

### Usage

Run all incomplete or invalid baseline experiments:

```bash
bash scripts/train.sh
```

Force all experiments to rerun:

```bash
FORCE=1 bash scripts/train.sh
```

Run and consume a different seed:

```bash
SEED=1 bash scripts/train.sh
SEED=1 bash scripts/cosa.sh
```

### Migration notes

- Checkpoints in the old non-seeded path are no longer selected automatically.
- Existing results without `schema_version: 1`, matching fingerprints, complete timing metadata, or `test_mae_all.npy` will fail validation and be rerun.
- New seed-zero checkpoints use paths such as `checkpoints/DLinear/ETTh1_96/seed_0/checkpoint_best.pth`.

### Verification

- Bash syntax checks passed for all modified launch scripts.
- Python compilation checks passed for all modified Python files.
- ShellCheck passed for `scripts/train.sh` when available.
- `git diff --check` passed.
- Full model training was not launched as part of these changes.
