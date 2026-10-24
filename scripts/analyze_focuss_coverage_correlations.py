#!/usr/bin/env python3
"""Test associations between FOCUSS losses and evaluation coverage-rate gain."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import wandb
from scipy.stats import pearsonr, rankdata, spearmanr


HISTORY_PREFIX = "info/learner/default_policy/learner_stats/"
COVERAGE_CANDIDATES = (
    "custom_metrics/coverage_rate_mean",
    "eval/mean_coverage_rate",
    "evaluation/custom_metrics/coverage_rate_mean",
)
RUN_PAIRS = {
    # Longest non-empty seed-0 history for each variant in this W&B project.
    "facmac_focuss": ("f04b7_00000", "f052d_00000"),
    "resq_focuss": ("cdbcf_00000", "resq_focuss-1494505-8463bc34"),
    "riskq_focuss": ("riskq-1494503-09472b1c", "riskq_focuss-1630178-550d9a76"),
    "qfree_focuss": ("qfree-1630177-fd3edb95", "qfree_focuss-1732233-bd21616e"),
}
METRICS = {
    "focuss_credit_loss": f"{HISTORY_PREFIX}focuss_credit_loss",
    "focuss_action_loss": f"{HISTORY_PREFIX}focuss_action_loss",
    "credit_alignment_kl": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="mate-value-factorization-2m")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/focuss_statistics")
    )
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20_261_024)
    return parser.parse_args()


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def scan_run(run: Any) -> pd.DataFrame:
    records = []
    for row in run.scan_history(min_step=0):
        record = {
            "timesteps_total": finite(row.get("timesteps_total")),
            **{key: finite(row.get(key)) for key in COVERAGE_CANDIDATES},
            "focuss_credit_loss": finite(
                row.get(f"{HISTORY_PREFIX}focuss_credit_loss")
            ),
            "focuss_action_loss": finite(
                row.get(f"{HISTORY_PREFIX}focuss_action_loss")
            ),
            "focus_rho_entropy": finite(
                row.get(f"{HISTORY_PREFIX}focus_rho_entropy")
            ),
        }
        if math.isfinite(record["timesteps_total"]):
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame.sort_values("timesteps_total", inplace=True)
    frame.drop_duplicates("timesteps_total", keep="last", inplace=True)
    return frame


def choose_shared_coverage_key(
    backbone: pd.DataFrame, focuss: pd.DataFrame
) -> str:
    for key in COVERAGE_CANDIDATES:
        if backbone[key].notna().any() and focuss[key].notna().any():
            return key
    raise RuntimeError("No shared coverage-rate metric exists for the run pair")


def locf_with_endpoint_clamp(
    source_steps: np.ndarray, source_values: np.ndarray, target_steps: np.ndarray
) -> np.ndarray:
    """LOCF alignment; use first/last observation outside the observed range."""
    valid = np.isfinite(source_steps) & np.isfinite(source_values)
    steps = source_steps[valid]
    values = source_values[valid]
    if len(steps) == 0:
        return np.full(len(target_steps), np.nan)
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]
    indices = np.searchsorted(steps, target_steps, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def align_pair(
    algorithm: str,
    backbone_id: str,
    focuss_id: str,
    backbone: pd.DataFrame,
    focuss: pd.DataFrame,
) -> pd.DataFrame:
    coverage_key = choose_shared_coverage_key(backbone, focuss)
    metric_rows = focuss.dropna(
        subset=["timesteps_total", "focuss_credit_loss", "focuss_action_loss", "focus_rho_entropy"]
    ).copy()
    target_steps = metric_rows["timesteps_total"].to_numpy(dtype=float)
    focus_coverage = locf_with_endpoint_clamp(
        focuss["timesteps_total"].to_numpy(dtype=float),
        focuss[coverage_key].to_numpy(dtype=float),
        target_steps,
    )
    backbone_coverage = locf_with_endpoint_clamp(
        backbone["timesteps_total"].to_numpy(dtype=float),
        backbone[coverage_key].to_numpy(dtype=float),
        target_steps,
    )
    aligned = pd.DataFrame(
        {
            "algorithm": algorithm,
            "backbone_run_id": backbone_id,
            "focuss_run_id": focuss_id,
            "coverage_metric": coverage_key,
            "timesteps_total": target_steps,
            "focuss_coverage_rate": focus_coverage,
            "backbone_coverage_rate": backbone_coverage,
            "coverage_rate_gain": focus_coverage - backbone_coverage,
            "focuss_credit_loss": metric_rows["focuss_credit_loss"].to_numpy(),
            "focuss_action_loss": metric_rows["focuss_action_loss"].to_numpy(),
            "focus_rho_entropy": metric_rows["focus_rho_entropy"].to_numpy(),
        }
    )
    aligned["credit_alignment_kl"] = (
        aligned["focuss_credit_loss"] - aligned["focus_rho_entropy"]
    )
    return aligned.dropna().reset_index(drop=True)


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def circular_shift_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """Two-sided circular-shift test preserving within-series autocorrelation."""
    x_work = transform(x) if transform is not None else x
    y_work = transform(y) if transform is not None else y
    observed = corrcoef(x_work, y_work)
    if not math.isfinite(observed):
        return math.nan
    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    exceedances = 0
    for _ in range(permutations):
        shifted = x_work.copy()
        for indices in group_indices:
            if len(indices) > 1:
                offset = int(rng.integers(1, len(indices)))
                shifted[indices] = np.roll(x_work[indices], offset)
        statistic = corrcoef(shifted, y_work)
        if math.isfinite(statistic) and abs(statistic) >= abs(observed):
            exceedances += 1
    return (exceedances + 1.0) / (permutations + 1.0)


def residualize_time(values: np.ndarray, steps: np.ndarray, groups: np.ndarray) -> np.ndarray:
    residuals = np.empty_like(values, dtype=float)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        x = steps[indices]
        y = values[indices]
        if len(indices) < 2 or np.std(x) == 0:
            residuals[indices] = y - np.mean(y)
        else:
            slope, intercept = np.polyfit(x, y, 1)
            residuals[indices] = y - (slope * x + intercept)
    return residuals


def test_scope(
    frame: pd.DataFrame,
    scope: str,
    metric: str,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    subset = frame[["algorithm", "timesteps_total", "coverage_rate_gain", metric]].dropna()
    x = subset[metric].to_numpy(dtype=float)
    y = subset["coverage_rate_gain"].to_numpy(dtype=float)
    steps = subset["timesteps_total"].to_numpy(dtype=float)
    groups = subset["algorithm"].to_numpy()

    if scope == "pooled_within_algorithm":
        for group in np.unique(groups):
            indices = groups == group
            x[indices] -= np.mean(x[indices])
            y[indices] -= np.mean(y[indices])

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    detrended_x = residualize_time(x, steps, groups)
    detrended_y = residualize_time(y, steps, groups)
    detrended = pearsonr(detrended_x, detrended_y)

    return {
        "scope": scope,
        "metric": metric,
        "n": len(subset),
        "pearson_r": pearson.statistic,
        "pearson_p_iid": pearson.pvalue,
        "pearson_p_circular_shift": circular_shift_pvalue(
            x, y, groups, permutations, rng
        ),
        "spearman_rho": spearman.statistic,
        "spearman_p_iid": spearman.pvalue,
        "spearman_p_circular_shift": circular_shift_pvalue(
            x, y, groups, permutations, rng, transform=rankdata
        ),
        "detrended_pearson_r": detrended.statistic,
        "detrended_pearson_p_iid": detrended.pvalue,
        "detrended_pearson_p_circular_shift": circular_shift_pvalue(
            detrended_x, detrended_y, groups, permutations, rng
        ),
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    m = len(valid)
    adjusted = valid.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "scope",
        "metric",
        "n",
        "pearson_r",
        "pearson_p_circular_shift",
        "pearson_q_bh",
        "spearman_rho",
        "spearman_p_circular_shift",
        "detrended_pearson_r",
        "detrended_pearson_p_circular_shift",
        "detrended_pearson_q_bh",
    ]
    labels = [
        "Scope",
        "Metric",
        "n",
        "Pearson r",
        "Circular p",
        "BH q",
        "Spearman rho",
        "Spearman circular p",
        "Detrended r",
        "Detrended circular p",
        "Detrended BH q",
    ]
    lines = ["| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column == "n":
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pairs_markdown(frame: pd.DataFrame) -> str:
    columns = [
        "algorithm",
        "backbone_run_id",
        "focuss_run_id",
        "seed",
        "coverage_metric",
        "aligned_points",
        "min_timestep",
        "max_timestep",
    ]
    labels = [
        "Algorithm",
        "Backbone run",
        "FOCUSS run",
        "Seed",
        "Coverage metric",
        "n",
        "Min timestep",
        "Max timestep",
    ]
    lines = ["| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in {"seed", "aligned_points", "min_timestep", "max_timestep"}:
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    aligned_frames = []
    pair_rows = []
    for algorithm, (backbone_id, focuss_id) in RUN_PAIRS.items():
        backbone_run = api.run(f"{project_path}/{backbone_id}")
        focuss_run = api.run(f"{project_path}/{focuss_id}")
        backbone = scan_run(backbone_run)
        focuss = scan_run(focuss_run)
        aligned = align_pair(
            algorithm, backbone_id, focuss_id, backbone, focuss
        )
        aligned_frames.append(aligned)
        pair_rows.append(
            {
                "algorithm": algorithm,
                "backbone_run_id": backbone_id,
                "focuss_run_id": focuss_id,
                "seed": focuss_run.config.get("seed"),
                "coverage_metric": aligned["coverage_metric"].iloc[0],
                "aligned_points": len(aligned),
                "min_timestep": aligned["timesteps_total"].min(),
                "max_timestep": aligned["timesteps_total"].max(),
            }
        )
        print(f"{algorithm}: {backbone_id} vs {focuss_id} -> {len(aligned)} points")

    data = pd.concat(aligned_frames, ignore_index=True)
    pairs = pd.DataFrame(pair_rows)
    data.to_csv(args.output_dir / "coverage_gain_aligned.csv", index=False)
    pairs.to_csv(args.output_dir / "run_pairs.csv", index=False)

    rng = np.random.default_rng(args.seed)
    results = []
    for algorithm, subset in data.groupby("algorithm", sort=True):
        for metric in METRICS:
            results.append(
                test_scope(subset, algorithm, metric, args.permutations, rng)
            )
    for metric in METRICS:
        results.append(
            test_scope(
                data,
                "pooled_within_algorithm",
                metric,
                args.permutations,
                rng,
            )
        )
    result_frame = pd.DataFrame(results)
    result_frame["pearson_q_bh"] = benjamini_hochberg(
        result_frame["pearson_p_circular_shift"]
    )
    result_frame["spearman_q_bh"] = benjamini_hochberg(
        result_frame["spearman_p_circular_shift"]
    )
    result_frame["detrended_pearson_q_bh"] = benjamini_hochberg(
        result_frame["detrended_pearson_p_circular_shift"]
    )
    result_frame.to_csv(args.output_dir / "coverage_gain_correlations.csv", index=False)

    coverage_metrics = ", ".join(f"`{key}`" for key in sorted(pairs["coverage_metric"].unique()))
    report = "\n".join(
        [
            "# Coverage-rate gain correlation tests",
            "",
            f"Coverage uses {coverage_metrics}. At each FOCUSS metric timestep, "
            "both FOCUSS and backbone coverage are aligned by last observation carried "
            "forward; values outside a run's observed range are clamped to its first/last "
            "observation. Gain is FOCUSS minus backbone.",
            "",
            f"Circular-shift tests use {args.permutations:,} permutations and preserve "
            "within-run temporal autocorrelation. BH q-values correct the Pearson circular "
            "tests across all reported algorithm/metric combinations. Detrended r removes "
            "a separate linear timestep trend within each algorithm.",
            "",
            "## Run pairs",
            "",
            pairs_markdown(pairs),
            "",
            "## Results",
            "",
            markdown_table(result_frame),
            "",
            "IID p-values are retained in the CSV but should not be the primary inference "
            "because adjacent timesteps are autocorrelated.",
        ]
    )
    (args.output_dir / "coverage_gain_correlation_report.md").write_text(
        report, encoding="utf-8"
    )
    print(result_frame.to_string(index=False))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
