#!/usr/bin/env bash

# Run the resumable input/prediction-length COSA-P sweep. All arguments and
# LABEL_LEN handling are provided by train_eval_cosa_lengths.sh.

set -euo pipefail

exec env PAAS=True PERIOD_N="${PERIOD_N:-1}" \
  bash "$(dirname "$0")/train_eval_cosa_lengths.sh" "$@"
