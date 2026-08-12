#!/usr/bin/env python3
"""Collect TAFAS metrics produced by the length-sweep script.

Expected layout:
  results/TAFAS/<model>/<dataset>/input_<input_len>_pred_<pred_len>/
    tafas_output.log
    tafas_complete
"""

import argparse
import json
import re
import sys
from pathlib import Path


RUN_ID_PATTERN = re.compile(r"input_(?P<input_len>\d+)_pred_(?P<pred_len>\d+)")


def parse_final_result(path: Path) -> dict[str, object]:
    content = path.read_text()
    decoder = json.JSONDecoder()

    for position in reversed([index for index, char in enumerate(content) if char == "{"]):
        try:
            result, _ = decoder.raw_decode(content[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("model") == "TAFAS":
            return result
    raise ValueError("no complete TAFAS JSON result found")


def collect(results_root: Path) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows = []
    errors = []
    for log_path in results_root.glob("*/*/input_*_pred_*/tafas_output.log"):
        relative = log_path.relative_to(results_root)
        model, dataset, run_id, _ = relative.parts
        run_id_match = RUN_ID_PATTERN.fullmatch(run_id)
        if not run_id_match:
            errors.append({"path": str(log_path), "error": "invalid run directory name"})
            continue

        try:
            result = parse_final_result(log_path)
            final_results = result["final_results"]
            overall_stats = result["time_statistics"]["overall_stats"]
            parameters = result["parameters"]
            if not all(isinstance(value, dict) for value in (final_results, overall_stats, parameters)):
                raise ValueError("result has an invalid summary structure")
            row = {
                "method": "TAFAS",
                "model": model,
                "dataset": dataset,
                "input_len": int(run_id_match.group("input_len")),
                "pred_len": int(run_id_match.group("pred_len")),
                "test_mse": float(final_results["test_mse"]),
                "adaptation_count": int(final_results["adaptation_count"]),
                "total_time_seconds": float(overall_stats["total_time_seconds"]),
                "throughput_samples_per_sec": float(overall_stats["throughput_samples_per_sec"]),
                "total_params": int(parameters["total_params"]),
                "status": "complete" if (log_path.parent / "tafas_complete").is_file() else "missing_completion_marker",
                "log_path": str(log_path),
            }
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append({"path": str(log_path), "error": str(error)})
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: (row["model"], row["dataset"], row["input_len"], row["pred_len"])), errors


def format_markdown(rows: list[dict[str, object]]) -> str:
    columns = (
        "model", "dataset", "input_len", "pred_len", "test_mse", "adaptation_count",
        "total_time_seconds", "throughput_samples_per_sec", "total_params", "status",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect TAFAS length-sweep metrics from output logs.")
    parser.add_argument("--results-root", type=Path, default=Path("results/TAFAS"))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write the collected table or JSON to this file.")
    args = parser.parse_args()

    if not args.results_root.is_dir():
        print(f"TAFAS results directory does not exist: {args.results_root}", file=sys.stderr)
        return 1

    rows, errors = collect(args.results_root)
    output = json.dumps(rows, indent=2) + "\n" if args.format == "json" else format_markdown(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Wrote results to {args.output}", file=sys.stderr)
    else:
        print(output, end="")

    print(f"Collected {len(rows)} TAFAS result(s).", file=sys.stderr)
    for error in errors:
        print(f"Skipped {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
