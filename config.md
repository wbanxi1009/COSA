# Configuration Reference

This document describes every option declared in `config.py`, how `main.py`
loads it, and whether the current implementation actually uses it.

## Running With Configuration

The program has two configuration sources:

```bash
# Load defaults, then values from a YAML file.
python main.py --cfg experiment.yaml

# Load defaults, then override individual values on the command line.
python main.py DATA.NAME ETTh1 DATA.PRED_LEN 96 MODEL.pred_len 96

# YAML values are loaded first; command-line pairs override them.
python main.py --cfg experiment.yaml TTA.COSA.STEPS 3
```

The precedence is:

```text
config.py defaults < YAML file passed with --cfg < command-line key/value pairs
```

Use one key followed by one value. Booleans must be written as `True` or
`False`. Strings normally do not need quotation marks. Tuples in a YAML file
can use YAML list syntax, for example `METRIC_NAMES: [MAE]`.

`main.py` runs these stages in order:

1. Train if `TRAIN.ENABLE` is `True`.
2. Load the best checkpoint if `TTA.ENABLE` or `TEST.ENABLE` is `True`.
3. Run test-time adaptation (TTA) if `TTA.ENABLE` is `True`.
4. Run regular test prediction if `TEST.ENABLE` is `True`.

The saved checkpoint path is always:

```text
TRAIN.CHECKPOINT_DIR/checkpoint_best.pth
```

## Minimal Commands

Train a base model:

```bash
python main.py \
  DATA.NAME ETTh1 \
  DATA.PRED_LEN 96 \
  MODEL.NAME iTransformer \
  MODEL.pred_len 96 \
  TRAIN.ENABLE True \
  TEST.ENABLE False \
  TRAIN.CHECKPOINT_DIR ./checkpoints/iTransformer/ETTh1_96/
```

Run COSA on an existing checkpoint:

```bash
python main.py \
  DATA.NAME ETTh1 \
  DATA.PRED_LEN 96 \
  MODEL.NAME iTransformer \
  MODEL.pred_len 96 \
  TRAIN.ENABLE False \
  TEST.ENABLE False \
  TRAIN.CHECKPOINT_DIR ./checkpoints/iTransformer/ETTh1_96/ \
  RESULT_DIR ./results/SIMPLE/ \
  TTA.ENABLE True \
  TTA.SOLVER.BASE_LR 0.001 \
  TTA.COSA.BATCH_SIZE 48 \
  TTA.COSA.STEPS 3 \
  TTA.COSA.BUFFER_CONTEXT_SIZE 10 \
  TTA.COSA.PAAS True
```

Keep `DATA.PRED_LEN` and `MODEL.pred_len` equal. Also keep
`DATA.SEQ_LEN` and `MODEL.seq_len` equal if either is changed. The code does
not synchronize these pairs automatically.

## Important Current-Code Behavior

- The supported model names are `iTransformer`, `PatchTST`, `DLinear`,
  `OLS`, `FreTS`, and `MICN`.
- Supported dataset names are `weather`, `illness`, `electricity`, `traffic`,
  `exchange_rate`, `ETTh1`, `ETTh2`, `ETTm1`, and `ETTm2`.
- After all YAML and command-line options are loaded,
  `update_cfg_from_dataset()` overwrites `DATA.N_VAR`, `DATA.FEATURES`,
  `DATA.TARGET_START_IDX`, `DATA.PERIOD_LEN`, `DATA.TRAIN_RATIO`,
  `DATA.TEST_RATIO`, `MODEL.enc_in`, `MODEL.dec_in`, and `MODEL.c_out` for
  every built-in dataset.
- The TTA implementation is selected from text in `RESULT_DIR`, not from a
  method setting: `TAFAS` selects TAFAS, `PETSA` selects PETSA, `SIMPLE`
  selects COSA, and `DYNATTA` selects DynaTTA. No matching text means no TTA
  adapter is run.
- `TEST.ENABLE` defaults to `True`. Set it to `False` in a training-only or
  TTA-only run.
- The repository README and `scripts/cosa.sh` refer to `TTA.SIMPLE.*`.
  That key does not exist in `config.py`; use `TTA.COSA.*`.

## Root Options

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `SEED` | `0` | Active | Random seeds for Python, NumPy, and PyTorch. Use the same value to make runs more reproducible. |
| `NUM_GPUS` | `8` | Unused | Declared but not read by the current code. |
| `VISIBLE_DEVICES` | `0` | Active | GPU device selection passed to the CUDA setup. Usually use `0` for the first visible GPU. |
| `RESULT_DIR` | `results/` | Active | Result directory, saved `config.yaml` location, COSA CSV parent directory, and TTA-method selector. |
| `NORMALIZE` | `NST` | Unused | Declared but not read; normalization is selected by `NORM_MODULE.*`. |

## Data Loader Options

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `DATA_LOADER.NUM_WORKERS` | `2` | Active | Number of background data-loading worker processes. More may speed loading but consumes CPU and memory. |
| `DATA_LOADER.PIN_MEMORY` | `True` | Active | Uses pinned CPU memory for faster CPU-to-GPU transfer. Keep enabled for CUDA runs. |
| `DATA_LOADER.DROP_LAST` | `True` | Unused | Declared but loader construction uses `TRAIN.DROP_LAST`, `VAL.DROP_LAST`, and `TEST.DROP_LAST` instead. |

## Dataset Options

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `DATA.BASE_DIR` | `data/` | Active | Parent directory of dataset folders. A weather run loads `data/weather/weather.csv`. |
| `DATA.NAME` | `weather` | Active | Dataset name. Must be one of the supported names listed above. |
| `DATA.N_VAR` | `21` | Overwritten | Number of input variables/channels. It is derived from the selected built-in dataset. |
| `DATA.SEQ_LEN` | `96` | Active | Historical input-window length given to the forecasting model. |
| `DATA.LABEL_LEN` | `48` | Active/conditional | Decoder context length. Used to construct windows; its importance depends on the model. |
| `DATA.PRED_LEN` | `96` | Active | Number of future steps to forecast. Set `MODEL.pred_len` to the same value. |
| `DATA.FEATURES` | `M` | Overwritten/stored | Overwritten for built-in datasets and stored in the dataset object, but it does not change the current loading logic. |
| `DATA.TIMEENC` | `0` | Active | Time-feature encoding: `0` creates month/day/weekday/hour fields; `1` uses continuous time features. |
| `DATA.FREQ` | `h` | Active | Sampling frequency passed to time-feature generation, such as `h` for hourly. |
| `DATA.SCALE` | `standard` | Active | Data scaling: `standard`, `min-max`, or `min-max_fixed` (skip scaling). |
| `DATA.FRACTION` | `1.0` | Active | Fraction of the earliest chronological rows to retain before train/validation/test splitting. Use `0.5` for a half-size fast experiment. Must be in `(0, 1]`. |
| `DATA.TRAIN_RATIO` | `0.7` | Overwritten | Training split fraction for built-in datasets. Validation receives the remainder after train and test. |
| `DATA.TEST_RATIO` | `0.2` | Overwritten | Test split fraction for built-in datasets. |
| `DATA.DATE_IDX` | `0` | Active | Index of the `date` column. The loader asserts that this column is named `date`. |
| `DATA.TARGET_START_IDX` | `0` | Overwritten/stored | Overwritten for built-in datasets and stored, but not otherwise used by the current dataset implementation. |
| `DATA.PERIOD_LEN` | `24` | Overwritten/conditional | Seasonal period used only when SAN normalization is enabled. |
| `DATA.STATION_TYPE` | `adaptive` | Conditional | SAN station-selection behavior. |

## Training, Validation, and Test Options

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `TRAIN.ENABLE` | `False` | Active | Runs base-model training. |
| `TRAIN.SPLIT` | `train` | Unused | Declared but training data is built with the hard-coded train split. |
| `TRAIN.BATCH_SIZE` | `128` | Active | Number of windows per base-model training update. |
| `TRAIN.SHUFFLE` | `True` | Active | Randomizes training-window order. Normally keep `True`. |
| `TRAIN.DROP_LAST` | `True` | Active | Discards a final incomplete training batch. |
| `TRAIN.CHECKPOINT_DIR` | `results/` | Active | Checkpoint save/load directory. |
| `TRAIN.RESUME` | empty string | Unused | Declared but automatic resume from this path is not implemented. |
| `TRAIN.EVAL_PERIOD` | `5` | Active | Validate every N training epochs. |
| `TRAIN.PRINT_FREQ` | `100` | Active | Print training progress every N iterations. |
| `TRAIN.BEST_METRIC_INITIAL` | `inf` | Active | Initial best validation metric; infinity lets the first finite lower loss win. |
| `TRAIN.BEST_LOWER` | `True` | Active | Whether lower metric values indicate improvement. Correct for MAE and MSE. |
| `VAL.SPLIT` | `val` | Unused | Declared but validation data uses the hard-coded validation split. |
| `VAL.BATCH_SIZE` | `256` | Active | Validation batch size. |
| `VAL.SHUFFLE` | `False` | Active | Validation ordering. Keep `False`. |
| `VAL.DROP_LAST` | `False` | Active | Keep the final incomplete validation batch when `False`. |
| `VAL.VIS` | `False` | Unused | Declared but not consumed. |
| `TEST.ENABLE` | `True` | Active | Runs regular evaluation with `Predictor` after earlier stages. |
| `TEST.SPLIT` | `test` | Unused | Declared but test data uses the hard-coded test split. |
| `TEST.BATCH_SIZE` | `256` | Active/modified by COSA | Standard test batch size. COSA replaces it with the full test dataset, then creates its own sequential batches. |
| `TEST.SHUFFLE` | `False` | Active | Test ordering. Keep `False`; time-series TTA requires chronological order. |
| `TEST.DROP_LAST` | `False` | Active | Retains the final incomplete test batch. |

## Common TTA Options

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `TTA.ENABLE` | `False` | Active | Enables test-time adaptation. |
| `TTA.MODULE_NAMES_TO_ADAPT` | `cali` | Conditional | Module selection used by adaptation implementations that support it. |
| `TTA.LOG` | `False` | Unused | Declared but not consumed. |
| `TTA.SOLVER.OPTIMIZING_METHOD` | `adam` | Active | Optimizer for TTA adapter parameters. |
| `TTA.SOLVER.BASE_LR` | `0.005` | Active | Initial TTA adapter learning rate. COSA resets to this value per batch when configured to do so. |
| `TTA.SOLVER.WEIGHT_DECAY` | `0.0001` | Active | TTA optimizer L2 weight decay. |
| `TTA.SOLVER.MOMENTUM` | `0.9` | Conditional | Momentum for supported TTA optimizers, mainly SGD. |
| `TTA.SOLVER.NESTEROV` | `True` | Conditional | Nesterov momentum for supported optimizers. |
| `TTA.SOLVER.DAMPENING` | `0.0` | Conditional | Momentum dampening for supported optimizers. |

## COSA Options

These are used only when `TTA.ENABLE True` and `RESULT_DIR` contains `SIMPLE`.

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `TTA.COSA.BATCH_SIZE` | `25` | Active | Number of consecutive test windows per COSA adaptation batch when PAAS is off. Larger batches adapt less often. |
| `TTA.COSA.STEPS` | `20` | Active | Gradient steps for the output adapter per batch. More steps cost more time and may overfit. |
| `TTA.COSA.BUFFER_SIZE` | `10` | Unused | Declared but COSA uses a fixed `sample_history` maximum of 200 instead. |
| `TTA.COSA.BUFFER_CONTEXT_SIZE` | `5` | Active | Number of recent observed target-mean values concatenated to each base prediction. |
| `TTA.COSA.ADAPT_FREQUENCY` | `50` | Unused | Declared but COSA adapts every processed batch. |
| `TTA.COSA.FAST_ADAPTATION` | `True` | Active | Limits each batch to at most five adaptation steps and enables fast-mode gradient clipping/early stop. |
| `TTA.COSA.ADAPTIVE_LR` | `True` | Active/conditional | Adapts learning rate from loss behavior. It is used only when `FAST_ADAPTATION` is also `True`. |
| `TTA.COSA.MAX_LR` | `0.005` | Active/conditional | Upper bound for COSA's adaptive learning rate. |
| `TTA.COSA.MIN_LR` | `0.0001` | Active/conditional | Lower bound for COSA's adaptive learning rate. |
| `TTA.COSA.MOMENTUM_FACTOR` | `0.9` | Unused | Declared but not read by COSA. |
| `TTA.COSA.CONVERGENCE_THRESHOLD` | `0.0001` | Active/conditional | Stops fast adaptation when consecutive tracked losses differ by less than this value. |
| `TTA.COSA.VAR_WISE_GATING` | `True` | Active | Creates one correction network and one gate per time-series variable. `False` shares one network across variables. |
| `TTA.COSA.ADAPTER_LAYERS` | `1` | Active | Adapter architecture: `1` is a single linear layer; values other than `1` create a two-linear-layer network with Tanh and Dropout. |
| `TTA.COSA.HIDDEN_DIM` | `64` | Active/conditional | Hidden width when `ADAPTER_LAYERS` is not `1`. |
| `TTA.COSA.PER_BATCH_LR_RESET` | `True` | Active/conditional | Resets COSA adaptive LR to `TTA.SOLVER.BASE_LR` at each new batch. |
| `TTA.COSA.SAVE_CSV` | `False` | Active | Exports per-prediction values/errors to `RESULT_DIR/csv_predictions/COSA/`. Can create very large files. |
| `TTA.COSA.SAVE_PAAS_CSV` | `False` | Active | Exports PAAS batch-size and period details to a COSA CSV file. |
| `TTA.COSA.PAAS` | `False` | Active | Enables Periodicity-Aware Adaptive Scheduling: an FFT estimates a period and chooses batch size dynamically. |
| `TTA.COSA.PERIOD_N` | `1` | Active/conditional | Multiplies the detected PAAS period before deriving the batch size. |

## TAFAS Options

These are used only when `TTA.ENABLE True` and `RESULT_DIR` contains `TAFAS`.

| Option | Default | What it controls |
|---|---:|---|
| `TTA.TAFAS.PAAS` | `True` | Enables periodicity-aware adaptive scheduling. |
| `TTA.TAFAS.PERIOD_N` | `1` | Multiplies the detected period used by PAAS. |
| `TTA.TAFAS.BATCH_SIZE` | `64` | TAFAS batch size when PAAS is off. |
| `TTA.TAFAS.STEPS` | `1` | TAFAS adaptation gradient steps per batch. |
| `TTA.TAFAS.ADJUST_PRED` | `True` | Enables prediction adjustment. |
| `TTA.TAFAS.CALI_MODULE` | `True` | Enables the calibration module. |
| `TTA.TAFAS.GATING_INIT` | `0.01` | Initial value/scale for the TAFAS gating mechanism. |
| `TTA.TAFAS.HIDDEN_DIM` | `128` | Hidden dimension of TAFAS internal networks. |
| `TTA.TAFAS.GCM_VAR_WISE` | `True` | Uses variable-wise gating/calibration behavior. |

## PETSA Options

These are used only when `TTA.ENABLE True` and `RESULT_DIR` contains `PETSA`.

| Option | Default | What it controls |
|---|---:|---|
| `TTA.PETSA.PAAS` | `True` | Enables periodicity-aware adaptive scheduling. |
| `TTA.PETSA.PERIOD_N` | `1` | Multiplies the detected PAAS period. |
| `TTA.PETSA.BATCH_SIZE` | `64` | PETSA batch size when PAAS is off. |
| `TTA.PETSA.STEPS` | `1` | PETSA adaptation steps per batch. |
| `TTA.PETSA.ADJUST_PRED` | `True` | Enables PETSA prediction adjustment. |
| `TTA.PETSA.CALI_MODULE` | `True` | Enables PETSA calibration module. |
| `TTA.PETSA.GATING_INIT` | `0.01` | Initial gate value/scale. |
| `TTA.PETSA.HIDDEN_DIM` | `128` | Hidden dimension of PETSA internal networks. |
| `TTA.PETSA.GCM_VAR_WISE` | `True` | Variable-wise gating/calibration. This option is declared twice in `config.py`; both declarations set the same value. |
| `TTA.PETSA.RANK` | `16` | Rank of PETSA's low-rank component. Larger ranks add capacity and parameters. |
| `TTA.PETSA.LOSS_ALPHA` | `0.1` | Weight for PETSA's additional loss term. |

## DynaTTA Options

These are used only when `TTA.ENABLE True` and `RESULT_DIR` contains
`DYNATTA`.

| Option | Default | What it controls |
|---|---:|---|
| `TTA.DYNATTA.MSE_BUFFER_SIZE` | `256` | Number of recent MSE values used for z-score calculations. |
| `TTA.DYNATTA.METRIC_HISTORY_SIZE` | `256` | Number of recent metrics retained for normalization. |
| `TTA.DYNATTA.ALPHA_MIN` | `0.0001` | Minimum dynamic adaptation rate. |
| `TTA.DYNATTA.ALPHA_MAX` | `0.001` | Maximum dynamic adaptation rate. |
| `TTA.DYNATTA.KAPPA` | `1.0` | Sensitivity of adaptation rate to change/difficulty signals. |
| `TTA.DYNATTA.ETA` | `0.1` | Smoothing factor for the dynamic adaptation rate. |
| `TTA.DYNATTA.EPS` | `0.000001` | Small numerical-stability constant. |
| `TTA.DYNATTA.WARMUP_FACTOR` | `1` | Warmup duration multiplier; used with prediction length. |
| `TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL` | `1` | Number of steps between buffer updates. |
| `TTA.DYNATTA.UPDATE_METRICS_INTERVAL` | `1` | Number of steps between metric updates. |
| `TTA.DYNATTA.RTAB_SIZE` | `360` | Recent Time-series Adaptation Buffer size. |
| `TTA.DYNATTA.RDB_SIZE` | `100` | Representative Database size. |

## Base Forecasting Model Options

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `MODEL.NAME` | `iTransformer` | Active | Forecasting architecture. See the supported names above. |
| `MODEL.task_name` | `long_term_forecast` | Active | Task passed into model implementations. Leave as long-term forecasting for this project. |
| `MODEL.seq_len` | `DATA.SEQ_LEN` (`96`) | Active | Model input length. Keep equal to `DATA.SEQ_LEN`. |
| `MODEL.label_len` | `DATA.LABEL_LEN` (`48`) | Mostly unused | Declared in model config; model-specific implementations may ignore it. |
| `MODEL.pred_len` | `DATA.PRED_LEN` (`96`) | Active | Model forecast horizon. Keep equal to `DATA.PRED_LEN`. |
| `MODEL.e_layers` | `4` | Conditional | Encoder-layer count for attention-based models such as iTransformer/PatchTST. |
| `MODEL.d_layers` | `1` | Conditional | Decoder-layer count for model implementations that use a decoder. |
| `MODEL.factor` | `3` | Conditional | ProbSparse-attention factor in relevant source architectures; not used by iTransformer full attention. |
| `MODEL.enc_in` | `DATA.N_VAR` | Overwritten/conditional | Input channel count. Derived from built-in dataset selection. |
| `MODEL.dec_in` | `DATA.N_VAR` | Overwritten/conditional | Decoder input-channel count for models that use it. |
| `MODEL.c_out` | `DATA.N_VAR` | Overwritten/conditional | Output channel count for models that use it. |
| `MODEL.d_model` | `512` | Conditional | Embedding/hidden width. Larger values increase capacity, memory use, and compute. |
| `MODEL.d_ff` | `512` | Conditional | Feed-forward hidden width in transformer-style models. |
| `MODEL.moving_avg` | `25` | Conditional | Moving-average window for DLinear decomposition. |
| `MODEL.output_attention` | `False` | Conditional | Returns attention weights in models that support it. Usually leave `False`. |
| `MODEL.dropout` | `0.1` | Conditional | Dropout probability in models that contain dropout. |
| `MODEL.n_heads` | `8` | Conditional | Attention-head count. It should divide `MODEL.d_model`. |
| `MODEL.activation` | `gelu` | Conditional | Activation used by supported neural architectures. |
| `MODEL.channel_independence` | `True` | Conditional | Channel-independence behavior used by FreTS. |
| `MODEL.METRIC_NAMES` | `('MAE',)` | Active | Metrics selected for base-model training/evaluation reporting. |
| `MODEL.LOSS_NAMES` | `('MSE',)` | Active | Loss functions selected for base-model training. |
| `MODEL.embed` | `timeF` | Conditional | Time-feature embedding approach for models that consume time stamps. |
| `MODEL.freq` | `h` | Conditional | Model-side time frequency. Normally match `DATA.FREQ`. |
| `MODEL.ignore_stamp` | `False` | Conditional | Whether a supporting model should ignore timestamp features. |
| `MODEL.instance_norm` | `True` | Conditional | Instance normalization option used by OLS. |
| `MODEL.individual` | `False` | Conditional | Individual-channel handling option used by OLS. |
| `MODEL.alpha` | `0.000001` | Conditional | OLS regularization/coefficient setting. |

## Normalization Module Options

`NORM_MODULE.ENABLE` controls whether a separate normalizer is built. Select
one implementation with `NORM_MODULE.NAME`.

| Option | Default | Status | What it controls |
|---|---:|---|---|
| `NORM_MODULE.ENABLE` | `False` | Active | Enables a normalization module before/after forecasting. |
| `NORM_MODULE.NAME` | `SAN` | Active | Normalizer name: `SAN`, `RevIN`, or `DishTS`. |
| `SAN.RESULT_DIR` | `results/station/` | Conditional | SAN result directory. |
| `SAN.TRAIN.CHECKPOINT_DIR` | `results/station/` | Conditional | SAN checkpoint directory. |
| `SAN.SOLVER.OPTIMIZING_METHOD` | `adam` | Conditional | SAN optimizer. |
| `SAN.SOLVER.START_EPOCH` | `0` | Conditional | SAN initial epoch. |
| `SAN.SOLVER.MAX_EPOCH` | `10` | Conditional | SAN training epochs. |
| `SAN.SOLVER.BASE_LR` | `0.001` | Conditional | SAN initial learning rate. |
| `SAN.SOLVER.WEIGHT_DECAY` | `0.0001` | Conditional | SAN L2 weight decay. |
| `SAN.SOLVER.MOMENTUM` | `0.9` | Conditional | SAN momentum for applicable optimizers. |
| `SAN.SOLVER.NESTEROV` | `True` | Conditional | SAN Nesterov momentum for applicable optimizers. |
| `SAN.SOLVER.DAMPENING` | `0.0` | Conditional | SAN momentum dampening. |
| `SAN.SOLVER.LR_POLICY` | `cosine` | Conditional | SAN learning-rate policy. |
| `SAN.SOLVER.COSINE_END_LR` | `0.0` | Conditional | SAN final cosine learning rate. |
| `SAN.SOLVER.COSINE_AFTER_WARMUP` | `False` | Conditional | Enables cosine decay after SAN warmup. |
| `SAN.SOLVER.WARMUP_EPOCHS` | `0` | Conditional | SAN warmup epoch count. |
| `SAN.SOLVER.WARMUP_START_LR` | `0.001` | Conditional | SAN learning rate at warmup start. |
| `REVIN.EPS` | `0.00001` | Conditional | RevIN numerical-stability constant. |
| `REVIN.AFFINE` | `True` | Conditional | Lets RevIN learn affine scale and bias. |
| `REVIN.RESULT_DIR` | `results/revin/` | Conditional | RevIN result directory. |
| `REVIN.TRAIN.CHECKPOINT_DIR` | `results/revin/` | Conditional | RevIN checkpoint directory. |
| `DISHTS.INIT` | `standard` | Unused | Declared initialization option; current code does not consume it. |
| `DISHTS.RESULT_DIR` | `results/dishts/` | Conditional | DishTS result directory. |
| `DISHTS.TRAIN.CHECKPOINT_DIR` | `results/dishts/` | Conditional | DishTS checkpoint directory. |

`DishTS` currently imports `models.DishTS`, which is not present in this
repository. Do not enable it until that module is provided.

## Base-Model Optimizer Options

These options train the original forecasting model. They are separate from
`TTA.SOLVER.*`, which trains an adaptation module.

| Option | Default | What it controls |
|---|---:|---|
| `SOLVER.START_EPOCH` | `0` | First base-model training epoch. |
| `SOLVER.MAX_EPOCH` | `30` | Number of base-model training epochs. |
| `SOLVER.OPTIMIZING_METHOD` | `adam` | Base-model optimizer. |
| `SOLVER.BASE_LR` | `0.0001` | Base-model initial learning rate. |
| `SOLVER.WEIGHT_DECAY` | `0.0001` | Base-model L2 weight decay. |
| `SOLVER.MOMENTUM` | `0.9` | Momentum for applicable base-model optimizers. |
| `SOLVER.NESTEROV` | `True` | Nesterov momentum for applicable base-model optimizers. |
| `SOLVER.DAMPENING` | `0.0` | Momentum dampening for applicable base-model optimizers. |
| `SOLVER.LR_POLICY` | `cosine` | Base-model learning-rate schedule. |
| `SOLVER.COSINE_END_LR` | `0.0` | Final base-model cosine learning rate. |
| `SOLVER.COSINE_AFTER_WARMUP` | `False` | Enables cosine decay after warmup. |
| `SOLVER.WARMUP_EPOCHS` | `0` | Base-model warmup epoch count. |
| `SOLVER.WARMUP_START_LR` | `0.001` | Base-model learning rate at warmup start. |
