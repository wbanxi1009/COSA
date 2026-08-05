"""Summarize quality, distribution, dependence, and shift in forecasting CSVs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew


def numeric_summary(values, sample_interval_minutes):
    """Return per-channel statistics computed on the unscaled observations."""
    result = pd.DataFrame(index=values.columns)
    result["missing_fraction"] = values.isna().mean()
    result["mean"] = values.mean()
    result["std"] = values.std()
    result["min"] = values.min()
    result["q01"] = values.quantile(0.01)
    result["median"] = values.median()
    result["q99"] = values.quantile(0.99)
    result["max"] = values.max()
    result["skew"] = values.apply(lambda x: skew(x.dropna()))
    result["excess_kurtosis"] = values.apply(lambda x: kurtosis(x.dropna()))
    result["lag_1_acf"] = values.apply(lambda x: x.autocorr(lag=1))

    seasonal_lag = round(24 * 60 / sample_interval_minutes)
    if seasonal_lag >= 2 and seasonal_lag < len(values):
        result["daily_lag_acf"] = values.apply(lambda x: x.autocorr(lag=seasonal_lag))
    else:
        result["daily_lag_acf"] = np.nan

    # Compare the beginning and end using the full-series scale, avoiding units.
    chunk = max(1, len(values) // 5)
    start_mean = values.iloc[:chunk].mean()
    end_mean = values.iloc[-chunk:].mean()
    result["early_late_mean_shift_std"] = (end_mean - start_mean) / result["std"].replace(0, np.nan)
    result["dominant_period_steps"] = values.apply(dominant_period)
    return result


def dominant_period(series):
    series = series.interpolate(limit_direction="both").to_numpy()
    if len(series) < 4 or np.std(series) == 0:
        return np.nan
    frequencies, power = periodogram(series - series.mean())
    valid = frequencies > 0
    return round(1 / frequencies[valid][np.argmax(power[valid])], 2)


def median_or_nan(series):
    series = series.dropna()
    return series.median() if not series.empty else np.nan


def analyze(path, train_ratio, test_ratio):
    frame = pd.read_csv(path)
    timestamp = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    intervals = timestamp.diff().dt.total_seconds().div(60).dropna()
    interval_minutes = intervals.median()
    if pd.isna(interval_minutes) or interval_minutes <= 0:
        raise ValueError(f"{path}: first column must be a regular timestamp")

    channels = numeric_summary(values, interval_minutes)
    train_end = int(len(values) * train_ratio)
    test_start = len(values) - int(len(values) * test_ratio)
    train_mean, train_std = values.iloc[:train_end].mean(), values.iloc[:train_end].std()
    channels["test_train_mean_shift_std"] = (
        values.iloc[test_start:].mean() - train_mean
    ) / train_std.replace(0, np.nan)

    corr = values.corr().abs().mask(np.eye(values.shape[1], dtype=bool))
    summary = {
        "dataset": path.stem,
        "rows": len(frame),
        "channels": values.shape[1],
        "start": timestamp.min(),
        "end": timestamp.max(),
        "median_interval_minutes": interval_minutes,
        "irregular_interval_fraction": (intervals != interval_minutes).mean(),
        "duplicate_timestamps": timestamp.duplicated().sum(),
        "invalid_timestamps": timestamp.isna().sum(),
        "missing_values": int(values.isna().sum().sum()),
        "channels_with_missing": int((values.isna().any()).sum()),
        "median_abs_skew": median_or_nan(channels["skew"].abs()),
        "median_excess_kurtosis": median_or_nan(channels["excess_kurtosis"]),
        "median_lag_1_acf": median_or_nan(channels["lag_1_acf"]),
        "median_daily_lag_acf": median_or_nan(channels["daily_lag_acf"]),
        "median_early_late_shift_std": median_or_nan(channels["early_late_mean_shift_std"]),
        "median_test_train_shift_std": median_or_nan(channels["test_train_mean_shift_std"]),
        "median_max_abs_correlation": median_or_nan(corr.max()),
    }
    return summary, channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, default=list(Path("data").glob("*/*.csv")))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()
    if not args.files:
        parser.error("no CSV files supplied or found under data/")
    if not 0 < args.train_ratio < 1 or not 0 < args.test_ratio < 1:
        parser.error("split ratios must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in args.files:
        summary, channels = analyze(path, args.train_ratio, args.test_ratio)
        summaries.append(summary)
        channels.to_csv(args.output_dir / f"{path.stem}_channels.csv", index_label="channel")
    pd.DataFrame(summaries).to_csv(args.output_dir / "dataset_summary.csv", index=False)
    print(f"Wrote dataset and channel summaries to {args.output_dir}/")


if __name__ == "__main__":
    main()
