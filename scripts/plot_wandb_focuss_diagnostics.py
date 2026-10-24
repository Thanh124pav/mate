#!/usr/bin/env python3
"""Summarize and plot FOCUSS diagnostics against environment timesteps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "facmac_focuss": "#0072B2",
    "resq_focuss": "#D55E00",
    "riskq_focuss": "#009E73",
    "qfree_focuss": "#CC79A7",
}
BIN_WIDTH = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/wandb_mate_value_factorization_2m_focuss_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/focuss_plots"))
    return parser.parse_args()


def prepare_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["algorithm"] != "concaveq_focuss"].copy()
    frame = frame.dropna(subset=["timesteps_total"])
    frame.sort_values(["algorithm", "run_id", "timesteps_total"], inplace=True)
    return frame


def binned_stats(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute point-weighted mean and SD in fixed environment-timestep bins."""
    work = frame[["algorithm", "timesteps_total", metric]].dropna().copy()
    work["timestep_bin"] = (
        np.floor(work["timesteps_total"] / BIN_WIDTH) * BIN_WIDTH
    ).astype(int)
    summary = (
        work.groupby(["algorithm", "timestep_bin"], as_index=False)
        .agg(
            timesteps_total=("timesteps_total", "mean"),
            mean=(metric, "mean"),
            std=(metric, "std"),
            count=(metric, "size"),
        )
        .sort_values(["algorithm", "timesteps_total"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def plot_mean_std(ax: plt.Axes, summary: pd.DataFrame, algorithm: str) -> None:
    part = summary.loc[summary["algorithm"] == algorithm]
    x = part["timesteps_total"].to_numpy()
    mean = part["mean"].to_numpy()
    std = part["std"].to_numpy()
    ax.plot(x, mean, color=COLORS[algorithm], linewidth=2.25, label=algorithm)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=COLORS[algorithm],
        alpha=0.16,
        linewidth=0,
    )


def style_timestep_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)
    ax.set_xlabel("Environment timesteps")


def write_ragc_table(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write point-weighted RAGC averages over every available logging row."""
    table = (
        frame.groupby("algorithm", as_index=False)
        .agg(
            run_count=("run_id", "nunique"),
            logging_points=("relative_action_gradient_contribution_percent", "size"),
            mean_ragc_percent=("relative_action_gradient_contribution_percent", "mean"),
        )
        .sort_values("algorithm")
    )
    table.to_csv(path, index=False)
    return table


def plot_loss_panels(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    """Use independent panels/scales so the plot emphasizes each trajectory."""
    summary = binned_stats(frame, metric)
    algorithms = [name for name in COLORS if name in set(frame["algorithm"])]
    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(16.5, 5.2),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    for ax, algorithm in zip(axes, algorithms):
        plot_mean_std(ax, summary, algorithm)
        style_timestep_axis(ax)
        ax.set_title(algorithm)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = prepare_frame(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    per_summary = binned_stats(frame, "prediction_error_reduction_percent")
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    for algorithm in COLORS:
        if algorithm in set(frame["algorithm"]):
            plot_mean_std(ax, per_summary, algorithm)
    ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_ylabel("Prediction Error Reduction (%)")
    style_timestep_axis(ax)
    ax.legend(frameon=False, ncol=2)
    per_path = args.output_dir / "per_vs_timesteps.png"
    fig.savefig(per_path, dpi=220)
    plt.close(fig)

    ragc_path = args.output_dir / "ragc_mean_table.csv"
    ragc_table = write_ragc_table(frame, ragc_path)
    credit_path = args.output_dir / "credit_loss_trends_vs_timesteps.png"
    plot_loss_panels(
        frame,
        "focuss_credit_loss",
        "FOCUSS Credit Loss",
        credit_path,
    )
    action_path = args.output_dir / "action_loss_trends_vs_timesteps.png"
    plot_loss_panels(
        frame,
        "focuss_action_loss",
        "FOCUSS Action Loss",
        action_path,
    )

    print(f"Summarized {len(frame)} rows from {frame['run_id'].nunique()} runs")
    print(ragc_table.to_string(index=False))
    print(per_path.resolve())
    print(ragc_path.resolve())
    print(credit_path.resolve())
    print(action_path.resolve())


if __name__ == "__main__":
    main()
