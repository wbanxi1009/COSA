#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if (( $# == 0 )); then
    exec python datasets/analyze_datasets.py
fi

files=()
for dataset in "$@"; do
    file="data/${dataset}/${dataset}.csv"
    if [[ ! -f "${file}" ]]; then
        printf 'Unknown dataset: %s\n' "${dataset}" >&2
        printf '%s\n' 'Use a dataset directory name, for example: weather, ETTh1, or exchange_rate.' >&2
        exit 2
    fi
    files+=("${file}")
done

exec python datasets/analyze_datasets.py "${files[@]}"
