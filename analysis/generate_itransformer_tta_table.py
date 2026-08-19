"""Generate the iTransformer COSA-F, PETSA, and TAFAS MSE LaTeX table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT_PATH = ROOT / "figure" / "itransformer_tta_mse_table.tex"
INPUT_LENGTHS = [96, 192, 336, 720]
PREDICTION_LENGTHS = [96, 192, 336, 720]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"]
METHODS = ["COSA-F", "PETSA", "TAFAS"]
METHOD_FILES = {
    "COSA-F": "cosa_results.json",
    "PETSA": "petsa_results.json",
    "TAFAS": "tafas_results.json",
}


def load_results() -> pd.DataFrame:
    frames = []
    for method, filename in METHOD_FILES.items():
        with (REPORT_DIR / filename).open() as file:
            frame = pd.DataFrame(json.load(file))
        frames.append(frame[(frame["model"] == "iTransformer") & (frame["method"] == method)])
    return pd.concat(frames, ignore_index=True)


def format_mse(value: float, best: bool) -> str:
    formatted = f"{value:.3f}"
    return f"\\textbf{{{formatted}}}" if best else formatted


def table_for_input(results: pd.DataFrame, input_length: int) -> list[str]:
    linebreak = r"\\"
    lines = [
        f"\\subcaption{{Input length $L_{{in}}={input_length}$}}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Method & $L_{out}=96$ & $L_{out}=192$ & $L_{out}=336$ & $L_{out}=720$ " + linebreak,
        "\\midrule",
    ]
    subset = results[results["input_len"] == input_length]
    for dataset_index, dataset in enumerate(DATASETS):
        dataset_rows = subset[subset["dataset"] == dataset]
        for method_index, method in enumerate(METHODS):
            method_rows = dataset_rows[dataset_rows["method"] == method].set_index("pred_len")["test_mse"]
            cells = []
            for prediction_length in PREDICTION_LENGTHS:
                value = method_rows[prediction_length]
                best = value == dataset_rows[dataset_rows["pred_len"] == prediction_length]["test_mse"].min()
                cells.append(format_mse(value, best))
            dataset_cell = dataset.replace("_", "\\_") if method_index == 0 else ""
            lines.append(f"{dataset_cell} & {method} & " + " & ".join(cells) + " " + linebreak)
        if dataset_index != len(DATASETS) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def main() -> None:
    results = load_results()
    expected = len(METHODS) * len(DATASETS) * len(INPUT_LENGTHS) * len(PREDICTION_LENGTHS)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} matched results, found {len(results)}.")

    lines = [
        "% Requires: \\usepackage{booktabs} and \\usepackage{subcaption}",
        "% Boldface marks the lowest MSE for each dataset and prediction length within a fixed input length.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test MSE comparison of test-time adaptation methods with the iTransformer backbone. Each subtable fixes the input length; lower is better.}",
        "\\label{tab:itransformer-tta-mse}",
    ]
    for index, input_length in enumerate(INPUT_LENGTHS):
        lines.append("\\begin{subtable}[t]{0.49\\textwidth}")
        lines.extend(table_for_input(results, input_length))
        lines.append("\\end{subtable}")
        if index % 2 == 0:
            lines.append("\\hfill")
        elif index != len(INPUT_LENGTHS) - 1:
            lines.append("\\par\\vspace{0.75em}")
    lines.append("\\end{table*}")
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
