"""Generate architecture-agnostic COSA-F comparisons at input length 96.

Run from the repository root:
    python analysis/plot_architecture_agnostic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT_DIR = ROOT / "figure"
BACKBONES = ["DLinear", "FreTS", "PatchTST", "iTransformer"]
HORIZONS = [96, 192, 336, 720]
SHARED_DATASETS = ["ETTh1", "exchange_rate"]
COLORS = {"ETTh1": "#4477AA", "ETTh2": "#EE6677", "ETTm1": "#228833", "ETTm2": "#CCBB44", "exchange_rate": "#AA3377", "weather": "#66CCEE"}


def load_results(path: Path) -> pd.DataFrame:
    with path.open() as file:
        return pd.DataFrame(json.load(file))


def matched_improvements() -> pd.DataFrame:
    baseline = load_results(REPORT_DIR / "baseline_results.json")
    cosa = load_results(REPORT_DIR / "cosa_results.json")
    keys = ["model", "dataset", "input_len", "pred_len"]

    baseline = baseline[keys + ["test_mse"]].rename(columns={"test_mse": "baseline_mse"})
    cosa = cosa[cosa["method"] == "COSA-F"]
    cosa = cosa[keys + ["test_mse"]].rename(columns={"test_mse": "cosa_mse"})
    results = baseline.merge(cosa, on=keys, how="inner", validate="one_to_one")
    results = results[
        (results["input_len"] == 96)
        & results["model"].isin(BACKBONES)
        & results["dataset"].isin(SHARED_DATASETS)
        & results["pred_len"].isin(HORIZONS)
    ].copy()
    results["improvement_pct"] = 100 * (results["baseline_mse"] - results["cosa_mse"]) / results["baseline_mse"]
    return results.sort_values(["model", "dataset", "pred_len"])


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_backbone_panels(results: pd.DataFrame) -> None:
    datasets = SHARED_DATASETS
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    offsets = np.linspace(-0.3, 0.3, len(datasets))
    width = 0.7 / len(datasets)

    for axis, backbone in zip(axes.flat, BACKBONES):
        subset = results[results["model"] == backbone]
        for dataset, offset in zip(datasets, offsets):
            values = subset[subset["dataset"] == dataset].set_index("pred_len")["improvement_pct"].reindex(HORIZONS)
            axis.bar(np.arange(len(HORIZONS)) + offset, values, width=width, label=dataset, color=COLORS.get(dataset))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(backbone, fontweight="bold")
        axis.set_xticks(range(len(HORIZONS)), HORIZONS)
        axis.set_xlabel("Prediction length")
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].set_ylabel("Relative MSE reduction (%)")
    axes[1, 0].set_ylabel("Relative MSE reduction (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(datasets), frameon=False)
    figure.suptitle("COSA-F improvement over each backbone baseline (input length = 96)", y=0.995, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.83))
    save_figure(figure, "architecture_agnostic_improvement")


def plot_heatmap(results: pd.DataFrame) -> None:
    columns = [(dataset, horizon) for dataset in SHARED_DATASETS for horizon in HORIZONS]
    matrix = np.array([
        [
            results.loc[
                (results["model"] == backbone)
                & (results["dataset"] == dataset)
                & (results["pred_len"] == horizon),
                "improvement_pct",
            ].iloc[0]
            if not results.loc[
                (results["model"] == backbone)
                & (results["dataset"] == dataset)
                & (results["pred_len"] == horizon),
                "improvement_pct",
            ].empty
            else np.nan
            for dataset, horizon in columns
        ]
        for backbone in BACKBONES
    ])
    maximum = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    figure, axis = plt.subplots(figsize=(14, 3.8))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-maximum, vmax=maximum, aspect="auto")
    axis.set_yticks(range(len(BACKBONES)), BACKBONES)
    axis.set_xticks(range(len(columns)), [str(horizon) for _, horizon in columns])
    axis.set_xlabel("Prediction length within dataset groups")
    axis.set_ylabel("Backbone")
    figure.suptitle("COSA-F relative MSE reduction over matched baseline (input length = 96)", y=0.98, fontweight="bold")

    for index, (dataset, _) in enumerate(columns):
        if index == 0 or dataset != columns[index - 1][0]:
            axis.axvline(index - 0.5, color="white", linewidth=2)
            axis.text(index + 1.5, 1.05, dataset, transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontweight="bold")
    axis.axvline(len(columns) - 0.5, color="white", linewidth=2)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.1f}" if not np.isnan(value) else "-"
            axis.text(column, row, label, ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Relative MSE reduction (%)")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, "architecture_agnostic_heatmap")


def write_summary(results: pd.DataFrame) -> None:
    summary = results.groupby("model", sort=False)["improvement_pct"].agg(
        settings="count", mean_improvement_pct="mean", median_improvement_pct="median", worst_setting_pct="min"
    )
    summary["win_rate_pct"] = 100 * results.groupby("model", sort=False)["improvement_pct"].apply(lambda values: (values > 0).mean())
    summary = summary.reindex(BACKBONES).reset_index().rename(columns={"model": "backbone"})
    summary.to_csv(OUTPUT_DIR / "architecture_agnostic_summary.csv", index=False, float_format="%.2f")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = matched_improvements()
    coverage = results.groupby(["model", "dataset"])["pred_len"].agg(list)
    incomplete = coverage[coverage.apply(lambda horizons: sorted(horizons) != HORIZONS)]
    if not incomplete.empty:
        raise ValueError(f"Incomplete horizon coverage: {incomplete.to_dict()}")
    results.to_csv(OUTPUT_DIR / "architecture_agnostic_matched_results.csv", index=False, float_format="%.6f")
    plot_backbone_panels(results)
    plot_heatmap(results)
    write_summary(results)


if __name__ == "__main__":
    main()
