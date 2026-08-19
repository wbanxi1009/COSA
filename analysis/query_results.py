#!/usr/bin/env python3
"""Filter collected experiment results and aggregate metrics by method.

Examples:
    python analysis/query_results.py --model DLinear --dataset ETTh1 --input-len 96 --pred-len 96
    python analysis/query_results.py --model DLinear --group-by dataset method
    python analysis/query_results.py --dataset ETTh1 --input-len 96 --group-by pred_len method --output etth1.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


FILTER_FIELDS = ("method", "model", "dataset", "input_len", "pred_len")


def parse_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = set(FILTER_FIELDS) | {"test_mse", "test_mae"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def matches(row: dict[str, str], args: argparse.Namespace) -> bool:
    return (
        (args.method is None or row["method"] == args.method)
        and (args.model is None or row["model"] == args.model)
        and (args.dataset is None or row["dataset"] == args.dataset)
        and (args.input_len is None or row["input_len"] == str(args.input_len))
        and (args.pred_len is None or row["pred_len"] == str(args.pred_len))
    )


def aggregate(rows: list[dict[str, str]], group_by: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_by)].append(row)

    summaries = []
    for key, group in sorted(groups.items()):
        mse_values = [float(row["test_mse"]) for row in group if row["test_mse"]]
        mae_values = [float(row["test_mae"]) for row in group if row["test_mae"]]
        summaries.append({
            **dict(zip(group_by, key)),
            "run_count": len(group),
            "mean_test_mse": statistics.mean(mse_values),
            "std_test_mse": statistics.stdev(mse_values) if len(mse_values) > 1 else 0.0,
            "min_test_mse": min(mse_values),
            "max_test_mse": max(mse_values),
            "mean_test_mae": statistics.mean(mae_values) if mae_values else None,
        })
    return summaries


def format_value(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def print_table(rows: list[dict[str, Any]], fields: list[str]) -> None:
    widths = {
        field: max(len(field), *(len(format_value(row[field])) for row in rows))
        for field in fields
    }
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(format_value(row[field]).ljust(widths[field]) for field in fields))


def write_output(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter collected results and compute metric summaries.")
    parser.add_argument("--csv", type=Path, default=Path("results/analysis/collected_results.csv"), help="Input file from analyze_results.py.")
    parser.add_argument("--method", help="Keep one method, for example baseline or COSA-F.")
    parser.add_argument("--model", help="Keep one model backbone.")
    parser.add_argument("--dataset", help="Keep one dataset.")
    parser.add_argument("--input-len", type=int, help="Keep one input length.")
    parser.add_argument("--pred-len", type=int, help="Keep one prediction length.")
    parser.add_argument("--group-by", nargs="+", choices=FILTER_FIELDS, default=["method"], help="Columns defining each aggregate row (default: method).")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the aggregate table.")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV does not exist: {args.csv}")

    try:
        selected = [row for row in parse_rows(args.csv) if matches(row, args)]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not selected:
        print("No runs matched the supplied filters.", file=sys.stderr)
        return 1

    summaries = aggregate(selected, args.group_by)
    metric_fields = ["run_count", "mean_test_mse", "std_test_mse", "min_test_mse", "max_test_mse", "mean_test_mae"]
    fields = args.group_by + metric_fields
    print(f"Matched {len(selected)} run(s); reporting {len(summaries)} group(s).")
    print_table(summaries, fields)
    if args.output:
        write_output(args.output, summaries, fields)
        print(f"Wrote aggregate table to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
