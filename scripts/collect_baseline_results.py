#!/usr/bin/env python3
"""Collect baseline metrics produced by the length-sweep scripts.

Expected layout:
  results/baseline/<model>/<dataset>/input_<input_len>_pred_<pred_len>/test.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path


RUN_ID_PATTERN = re.compile(r"input_(?P<input_len>\d+)_pred_(?P<pred_len>\d+)")
METRIC_PATTERN = re.compile(
    r"(?P<name>test_mse|test_mae|train_mse|train_mae):\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_metrics(path: Path) -> dict[str, float]:
    metrics = {
        match.group("name"): float(match.group("value"))
        for match in METRIC_PATTERN.finditer(path.read_text())
    }
    required = {"test_mse", "test_mae", "train_mse", "train_mae"}
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"missing metrics: {', '.join(sorted(missing))}")
    return metrics


def collect(results_root: Path) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows = []
    errors = []
    for test_path in results_root.glob("*/*/input_*_pred_*/test.txt"):
        relative = test_path.relative_to(results_root)
        model, dataset, run_id, _ = relative.parts
        run_id_match = RUN_ID_PATTERN.fullmatch(run_id)
        if not run_id_match:
            errors.append({"path": str(test_path), "error": "invalid run directory name"})
            continue

        try:
            metrics = parse_metrics(test_path)
        except (OSError, ValueError) as error:
            errors.append({"path": str(test_path), "error": str(error)})
            continue

        rows.append(
            {
                "method": "baseline",
                "model": model,
                "dataset": dataset,
                "input_len": int(run_id_match.group("input_len")),
                "pred_len": int(run_id_match.group("pred_len")),
                **metrics,
                "result_path": str(test_path),
            }
        )
    return sorted(rows, key=lambda row: (row["model"], row["dataset"], row["input_len"], row["pred_len"])), errors


def format_markdown(rows: list[dict[str, object]]) -> str:
    columns = ("model", "dataset", "input_len", "pred_len", "test_mse", "test_mae", "train_mse", "train_mae")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect baseline length-sweep metrics from test.txt files.")
    parser.add_argument("--results-root", type=Path, default=Path("results/baseline"))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write the collected table or JSON to this file.")
    args = parser.parse_args()

    if not args.results_root.is_dir():
        print(f"Baseline results directory does not exist: {args.results_root}", file=sys.stderr)
        return 1

    rows, errors = collect(args.results_root)
    if args.format == "json":
        output = json.dumps(rows, indent=2) + "\n"
    else:
        output = format_markdown(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Wrote results to {args.output}", file=sys.stderr)
    else:
        print(output, end="")

    print(f"Collected {len(rows)} baseline result(s).", file=sys.stderr)
    for error in errors:
        print(f"Skipped {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
