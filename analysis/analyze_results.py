#!/usr/bin/env python3
"""Collect experiment results into tables and summarize matched comparisons.

Examples:
    python analysis/analyze_results.py
    python analysis/analyze_results.py --results-root /path/to/results
    python analysis/analyze_results.py --results-root results --output-dir report/results

The collector supports baseline ``test.txt`` files and TTA ``*_output.log``
files whose final complete JSON object contains the test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


METRIC_PATTERN = re.compile(
    r"(?P<name>test_mse|test_mae|train_mse|train_mae):\s*"
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
RUN_ID_PATTERN = re.compile(r"input_(?P<input_len>\d+)_pred_(?P<pred_len>\d+)")
CONFIG_PATTERN = re.compile(r"^\s*(?P<key>NAME|SEQ_LEN|PRED_LEN):\s*(?P<value>.+?)\s*$")
TTA_METHODS = {
    "SimpleAdapter": "COSA-F",
    "PETSA": "PETSA",
    "TAFAS": "TAFAS",
    "DynATTA": "DynATTA",
}


def last_json_object(path: Path) -> dict[str, Any]:
    """Return the final complete result object printed in a mixed console log."""
    content = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    for position in reversed([index for index, character in enumerate(content) if character == "{"]):
        try:
            value, _ = decoder.raw_decode(content[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("final_results"), dict):
            return value
    raise ValueError("no complete result JSON object found")


def config_metadata(directory: Path, results_root: Path) -> dict[str, Any]:
    """Read the minimal metadata needed from the closest run config.yaml."""
    for parent in (directory, *directory.parents):
        if parent == results_root.parent:
            break
        config_path = parent / "config.yaml"
        if not config_path.is_file():
            continue
        values: dict[str, list[str]] = {"NAME": [], "SEQ_LEN": [], "PRED_LEN": []}
        for line in config_path.read_text(errors="replace").splitlines():
            match = CONFIG_PATTERN.match(line)
            if match:
                values[match.group("key")].append(match.group("value").strip(" '\""))
        # The first NAME is DATA.NAME and the second is MODEL.NAME in this project's configs.
        return {
            "dataset": values["NAME"][0] if values["NAME"] else None,
            "model": values["NAME"][1] if len(values["NAME"]) > 1 else None,
            "input_len": int(values["SEQ_LEN"][0]) if values["SEQ_LEN"] else None,
            "pred_len": int(values["PRED_LEN"][0]) if values["PRED_LEN"] else None,
        }
    return {"dataset": None, "model": None, "input_len": None, "pred_len": None}


def path_metadata(path: Path, results_root: Path) -> dict[str, Any]:
    metadata = config_metadata(path.parent, results_root)
    for parent in (path.parent, *path.parents):
        match = RUN_ID_PATTERN.fullmatch(parent.name)
        if match:
            metadata["input_len"] = int(match.group("input_len"))
            metadata["pred_len"] = int(match.group("pred_len"))
            parts = parent.relative_to(results_root).parts
            if len(parts) >= 4:
                metadata["dataset"] = metadata["dataset"] or parts[-3]
                metadata["model"] = metadata["model"] or parts[-4]
            break
    return metadata


def parse_test_file(path: Path, results_root: Path) -> dict[str, Any]:
    metrics = {match.group("name"): float(match.group("value")) for match in METRIC_PATTERN.finditer(path.read_text())}
    missing = {"test_mse", "test_mae", "train_mse", "train_mae"} - metrics.keys()
    if missing:
        raise ValueError(f"missing metrics: {', '.join(sorted(missing))}")
    return {
        "method": "baseline",
        **path_metadata(path, results_root),
        **metrics,
        "adaptation_count": None,
        "total_time_seconds": None,
        "throughput_samples_per_sec": None,
        "total_params": None,
        "source_path": str(path.relative_to(results_root)),
        "source_type": "test.txt",
    }


def parse_tta_log(path: Path, results_root: Path) -> dict[str, Any]:
    result = last_json_object(path)
    final = result.get("final_results")
    if not isinstance(final, dict) or "test_mse" not in final:
        raise ValueError("JSON does not contain final_results.test_mse")
    timing = result.get("time_statistics", {})
    overall = timing.get("overall_stats", {}) if isinstance(timing, dict) else {}
    parameters = result.get("parameters", {})
    model_name = result.get("model")
    return {
        "method": TTA_METHODS.get(str(model_name), str(model_name) if model_name else "unknown_tta"),
        **path_metadata(path, results_root),
        "test_mse": float(final["test_mse"]),
        "test_mae": float(final["test_mae"]) if "test_mae" in final else None,
        "train_mse": None,
        "train_mae": None,
        "adaptation_count": int(final["adaptation_count"]) if "adaptation_count" in final else None,
        "total_time_seconds": float(overall["total_time_seconds"]) if "total_time_seconds" in overall else None,
        "throughput_samples_per_sec": float(overall["throughput_samples_per_sec"]) if "throughput_samples_per_sec" in overall else None,
        "total_params": int(parameters["total_params"]) if isinstance(parameters, dict) and "total_params" in parameters else None,
        "source_path": str(path.relative_to(results_root)),
        "source_type": "tta_log",
    }


def collect(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    paths = sorted(results_root.rglob("test.txt")) + sorted(results_root.rglob("*_output.log"))
    for path in paths:
        try:
            row = parse_test_file(path, results_root) if path.name == "test.txt" else parse_tta_log(path, results_root)
            missing_metadata = [key for key in ("model", "dataset", "input_len", "pred_len") if row[key] is None]
            if missing_metadata:
                raise ValueError(f"missing metadata: {', '.join(missing_metadata)}")
            rows.append(row)
        except (OSError, TypeError, ValueError) as error:
            errors.append({"source_path": str(path.relative_to(results_root)), "error": str(error)})
    return sorted(rows, key=lambda row: (str(row["method"]), str(row["model"]), str(row["dataset"]), int(row["input_len"]), int(row["pred_len"]))), errors


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    keys = ("model", "dataset", "input_len", "pred_len")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    best_rows = []
    comparisons = []
    for key, group in sorted(grouped.items()):
        best = min(group, key=lambda row: float(row["test_mse"]))
        best_rows.append({**dict(zip(keys, key)), "best_method": best["method"], "best_test_mse": best["test_mse"], "run_count": len(group)})
        baselines = [row for row in group if row["method"] == "baseline"]
        if not baselines:
            continue
        baseline_mse = sum(float(row["test_mse"]) for row in baselines) / len(baselines)
        for row in group:
            if row["method"] == "baseline":
                continue
            comparisons.append({
                **dict(zip(keys, key)),
                "method": row["method"],
                "baseline_mse": baseline_mse,
                "baseline_run_count": len(baselines),
                "test_mse": row["test_mse"],
                "improvement_pct": 100 * (baseline_mse - row["test_mse"]) / baseline_mse,
                "source_path": row["source_path"],
            })
    summary = {
        "total_runs": len(rows),
        "methods": dict(sorted(Counter(str(row["method"]) for row in rows).items())),
        "models": dict(sorted(Counter(str(row["model"]) for row in rows).items())),
        "datasets": dict(sorted(Counter(str(row["dataset"]) for row in rows).items())),
        "configuration_count": len(grouped),
        "matched_baseline_comparisons": len(comparisons),
    }
    return best_rows, comparisons, summary


def write_markdown(path: Path, summary: dict[str, Any], errors: list[dict[str, str]], comparisons: list[dict[str, Any]]) -> None:
    lines = ["# Results Analysis", "", f"- Collected runs: {summary['total_runs']}", f"- Unique configurations: {summary['configuration_count']}", f"- Matched baseline comparisons: {summary['matched_baseline_comparisons']}", f"- Parse failures: {len(errors)}", "", "## Runs By Method", "", "| Method | Runs |", "|---|---:|"]
    lines.extend(f"| {method} | {count} |" for method, count in summary["methods"].items())
    if comparisons:
        lines.extend(["", "## Baseline Comparisons", "", "| Model | Dataset | Input | Prediction | Method | MSE Improvement |", "|---|---|---:|---:|---|---:|"])
        lines.extend(f"| {row['model']} | {row['dataset']} | {row['input_len']} | {row['pred_len']} | {row['method']} | {row['improvement_pct']:.2f}% |" for row in comparisons)
    if errors:
        lines.extend(["", "## Parse Failures", "", "| Source | Error |", "|---|---|"])
        lines.extend(f"| `{row['source_path']}` | {row['error']} |" for row in errors)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and analyze experiment results.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Directory containing experiment results.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated files (default: <results-root>/analysis).")
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        parser.error(f"results root does not exist: {results_root}")
    output_dir = (args.output_dir or results_root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, errors = collect(results_root)
    best_rows, comparisons, summary = analyze(rows)
    fields = ["method", "model", "dataset", "input_len", "pred_len", "test_mse", "test_mae", "train_mse", "train_mae", "adaptation_count", "total_time_seconds", "throughput_samples_per_sec", "total_params", "source_type", "source_path"]
    write_csv(output_dir / "collected_results.csv", rows, fields)
    (output_dir / "collected_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(output_dir / "best_results.csv", best_rows, ["model", "dataset", "input_len", "pred_len", "best_method", "best_test_mse", "run_count"])
    write_csv(output_dir / "baseline_comparisons.csv", comparisons, ["model", "dataset", "input_len", "pred_len", "method", "baseline_mse", "baseline_run_count", "test_mse", "improvement_pct", "source_path"])
    (output_dir / "analysis_summary.json").write_text(json.dumps({**summary, "parse_failures": errors}, indent=2) + "\n")
    write_markdown(output_dir / "analysis_summary.md", summary, errors, comparisons)
    print(f"Collected {len(rows)} run(s); skipped {len(errors)} invalid file(s).")
    print(f"Wrote analysis to {output_dir}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
