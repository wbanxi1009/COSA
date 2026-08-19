"""Plot iTransformer input-output comparisons for COSA-F, PETSA, and TAFAS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT_DIR = ROOT / "figure"
METHODS = ["COSA-F", "PETSA", "TAFAS"]
INPUT_LENGTHS = [96, 192, 336, 720]
PREDICTION_LENGTHS = [96, 192, 336, 720]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"]
METHOD_COLORS = {"COSA-F": "#4477AA", "PETSA": "#EE7733", "TAFAS": "#228833"}
METHOD_FILES = {"COSA-F": "cosa_results.json", "PETSA": "petsa_results.json", "TAFAS": "tafas_results.json"}


def load_json(filename: str) -> pd.DataFrame:
    with (REPORT_DIR / filename).open() as file:
        return pd.DataFrame(json.load(file))


def load_matched_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for method, filename in METHOD_FILES.items():
        frame = load_json(filename)
        frames.append(frame[(frame["model"] == "iTransformer") & (frame["method"] == method)])
    methods = pd.concat(frames, ignore_index=True)
    baseline = load_json("baseline_results.json")
    baseline = baseline[baseline["model"] == "iTransformer"]
    keys = ["dataset", "input_len", "pred_len"]
    baseline = baseline[keys + ["test_mse"]].rename(columns={"test_mse": "baseline_mse"})
    methods = methods.merge(baseline, on=keys, validate="many_to_one")
    methods["improvement_pct"] = 100 * (methods["baseline_mse"] - methods["test_mse"]) / methods["baseline_mse"]
    expected = len(METHODS) * len(DATASETS) * len(INPUT_LENGTHS) * len(PREDICTION_LENGTHS)
    if len(methods) != expected:
        raise ValueError(f"Expected {expected} matched results, found {len(methods)}.")
    return methods, baseline


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_winner_map(results: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    cmap = ListedColormap([METHOD_COLORS[method] for method in METHODS])
    for axis, input_length in zip(axes.flat, INPUT_LENGTHS):
        subset = results[results["input_len"] == input_length]
        winners = np.empty((len(DATASETS), len(PREDICTION_LENGTHS)), dtype=int)
        margins = np.empty_like(winners, dtype=float)
        for row, dataset in enumerate(DATASETS):
            for column, prediction_length in enumerate(PREDICTION_LENGTHS):
                values = subset[(subset["dataset"] == dataset) & (subset["pred_len"] == prediction_length)].set_index("method")["test_mse"]
                ordered = values.sort_values()
                winners[row, column] = METHODS.index(ordered.index[0])
                margins[row, column] = 100 * (ordered.iloc[1] - ordered.iloc[0]) / ordered.iloc[1]
        axis.imshow(winners, cmap=cmap, vmin=-0.5, vmax=len(METHODS) - 0.5, aspect="auto")
        for row in range(len(DATASETS)):
            for column in range(len(PREDICTION_LENGTHS)):
                axis.text(column, row, f"+{margins[row, column]:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        axis.set_title(f"$L_{{in}}={input_length}$", fontweight="bold")
        axis.set_xticks(range(len(PREDICTION_LENGTHS)), PREDICTION_LENGTHS)
        axis.set_yticks(range(len(DATASETS)), DATASETS)
        axis.set_xlabel("Prediction length")
    axes[0, 0].set_ylabel("Dataset")
    axes[1, 0].set_ylabel("Dataset")
    figure.legend([Patch(color=METHOD_COLORS[method]) for method in METHODS], METHODS, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    figure.suptitle("Best iTransformer adaptation method by input-output configuration", y=0.995, fontweight="bold")
    figure.text(0.5, 0.01, "Cell annotation: relative MSE advantage over the second-best method", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.84))
    save_figure(figure, "itransformer_winner_map")


def plot_horizon_sensitivity(results: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for axis, input_length in zip(axes.flat, INPUT_LENGTHS):
        subset = results[results["input_len"] == input_length]
        for method in METHODS:
            values = subset[subset["method"] == method].groupby("pred_len")["improvement_pct"].mean().reindex(PREDICTION_LENGTHS)
            axis.plot(PREDICTION_LENGTHS, values, marker="o", linewidth=2.2, label=method, color=METHOD_COLORS[method])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"$L_{{in}}={input_length}$", fontweight="bold")
        axis.set_xticks(PREDICTION_LENGTHS)
        axis.set_xlabel("Prediction length")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Mean relative MSE reduction (%)")
    axes[1, 0].set_ylabel("Mean relative MSE reduction (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    figure.suptitle("Horizon sensitivity: mean improvement over iTransformer baseline", y=0.80, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(figure, "itransformer_horizon_sensitivity")


def plot_input_sensitivity(results: pd.DataFrame) -> None:
    figure, axes = plt.subplots(len(METHODS), len(DATASETS), figsize=(16, 8), sharex=True)
    horizon_colors = {96: "#4477AA", 192: "#EE7733", 336: "#228833", 720: "#AA3377"}
    for row, method in enumerate(METHODS):
        for column, dataset in enumerate(DATASETS):
            axis = axes[row, column]
            subset = results[(results["method"] == method) & (results["dataset"] == dataset)]
            for prediction_length in PREDICTION_LENGTHS:
                values = subset[subset["pred_len"] == prediction_length].set_index("input_len")["test_mse"].reindex(INPUT_LENGTHS)
                axis.plot(INPUT_LENGTHS, values, marker="o", linewidth=1.8, markersize=3.5, color=horizon_colors[prediction_length], label=str(prediction_length))
            if row == 0:
                axis.set_title(dataset, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{method}\nMSE")
            if row == len(METHODS) - 1:
                axis.set_xlabel("Input length")
            axis.set_xticks(INPUT_LENGTHS)
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, [f"$L_{{out}}={label}$" for label in labels], loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False)
    figure.suptitle("Input-length sensitivity across prediction horizons", y=1, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(figure, "itransformer_input_sensitivity")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    results, _ = load_matched_results()
    plot_winner_map(results)
    plot_horizon_sensitivity(results)
    plot_input_sensitivity(results)


if __name__ == "__main__":
    main()
