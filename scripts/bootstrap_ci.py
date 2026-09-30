"""Compute paired bootstrap confidence intervals for per-image metric deltas.

A model comparison is considered statistically robust at the selected confidence level when the interval for the mean paired delta excludes zero. Resampling is performed over image IDs with replacement, preserving the paired structure of the comparison.
"""
import argparse
import csv
import json
import os

import numpy as np


def load_deltas(csv_path: str, metric_columns: list[str]) -> dict:
    """Read the CSV from compare_evaluations.py and return a dict
    {metric_column: np.ndarray of values}, one per image."""
    values = {col: [] for col in metric_columns}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=_sniff_delimiter(f))

        missing = [c for c in metric_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Columns not found in CSV: {missing}. "
                f"Available columns: {reader.fieldnames}"
            )

        n_rows = 0
        for row in reader:
            for col in metric_columns:
                values[col].append(float(row[col]))
            n_rows += 1

    if n_rows == 0:
        raise ValueError(f"Il CSV {csv_path} contains no rows.")

    return {col: np.array(v, dtype=np.float64) for col, v in values.items()}, n_rows


def _sniff_delimiter(f) -> str:
    """Detect whether the comparison CSV uses commas or tabs, then rewind the file."""
    pos = f.tell()
    first_line = f.readline()
    f.seek(pos)
    if "\t" in first_line and "," not in first_line:
        return "\t"
    return ","


def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Non-parametric bootstrap of the mean per-image delta."""
    rng = np.random.default_rng(seed)
    n = len(deltas)

    resampled_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = deltas[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(resampled_means, alpha)
    upper = np.quantile(resampled_means, 1.0 - alpha)

    observed_mean = deltas.mean()
    # If the CI includes zero, the improvement is not distinguishable
    # from sampling noise at this confidence level.
    excludes_zero = (lower > 0.0) or (upper < 0.0)

    return {
        "n_samples": n,
        "n_resamples": n_resamples,
        "observed_mean": float(observed_mean),
        "ci_level": ci,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "significant_at_ci_level": bool(excludes_zero),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_csv", type=str,
                         help="Output from scripts/compare_evaluations.py")
    parser.add_argument("--metrics", type=str, nargs="+",
                         default=["delta_cc", "delta_sim", "delta_kld"],
                         help="Names of delta columns to analyze")
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--output", type=str, default=None,
                         help="If provided, save results as JSON")
    args = parser.parse_args()

    deltas_by_metric, n_rows = load_deltas(args.comparison_csv, args.metrics)
    print(f"Rows read from {args.comparison_csv}: {n_rows}")
    print(f"Resamples: {args.n_resamples} | Seed: {args.seed} | CI: {args.ci:.0%}\n")

    results = {}
    for metric in args.metrics:
        result = paired_bootstrap_ci(
            deltas_by_metric[metric],
            n_resamples=args.n_resamples,
            seed=args.seed,
            ci=args.ci,
        )
        results[metric] = result

        direction = "migliora" if result["observed_mean"] > 0 else "peggiora"
        # Lower KLD is better; report this without flipping
        # the sign because the comparison file defines the sign convention.
        sig = "SI" if result["significant_at_ci_level"] else "NO (CI include lo zero)"

        print(f"{metric}: mean={result['observed_mean']:+.6f}  "
              f"CI[{args.ci:.0%}]=[{result['ci_lower']:+.6f}, {result['ci_upper']:+.6f}]  "
              f"significativo={sig}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()