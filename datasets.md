# Datasets

## Analysis

Analyze raw, unnormalized series before choosing a model or reporting robustness
claims. Activate the project environment, then run:

```bash
bash scripts/data_analyzer.sh
```

Select one or more datasets by directory name instead of analyzing every local
CSV:

```bash
bash scripts/data_analyzer.sh weather ETTh1 exchange_rate
```

The analyzer checks timestamp quality and missingness, distribution tails, serial
and daily dependence, cross-channel correlation, dominant spectral periods, and
standardized temporal shifts. It writes a dataset-level comparison to
`analysis/dataset_summary.csv` and one per-channel table per dataset at
`analysis/<dataset>_channels.csv`.

For each channel, the report includes missingness, mean, standard deviation,
quantiles, skewness, excess kurtosis, lag-1 and daily autocorrelation, an
early-to-late mean shift, train-to-test mean shift, and dominant spectral
period. The dataset table adds timestamp validity, duplicate and irregular
interval counts, and median cross-channel correlation.

Large early-to-late or test-to-train mean shifts indicate nonstationarity. High
autocorrelation or a stable dominant period indicates exploitable temporal
structure. High skewness or kurtosis means scale-sensitive metrics and
normalization need particular care. The train/test shift uses the loader's 70/10/20
chronological split convention.
