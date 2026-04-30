"""
Figure generation from Monte Carlo results.

Reads ``benchmarks/results/results_raw.csv`` and writes:

    benchmarks/results/figures/bias_by_dgp.png
    benchmarks/results/figures/rmse_comparison.png
    benchmarks/results/figures/coverage_vs_width.png
    benchmarks/results/figures/runtime_comparison.png

Usage:
    python -m benchmarks.plot_results
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Colour the proposed method distinctly everywhere
RX_COLORS = {"RX-Learner (robust)": "#d62728", "RX-Learner (std)": "#ff9896"}
BASELINE_COLOR = "#1f77b4"


def _load() -> pd.DataFrame:
    csv = RESULTS_DIR / "results_raw.csv"
    if not csv.exists():
        sys.exit(f"Missing {csv} — run `python -m benchmarks.run_pipeline_comparison` first.")
    df = pd.read_csv(csv)
    df = df[df["ate"].notna()].copy()          # drop failed rows
    df["ate"] = df["ate"].astype(float)
    df["bias"] = df["ate"] - df["tau_true"].astype(float)
    df["abs_bias"] = df["bias"].abs()
    df["covered"] = df.apply(
        lambda r: (pd.notna(r["ci_lo"]) and pd.notna(r["ci_hi"])
                   and float(r["ci_lo"]) <= r["tau_true"] <= float(r["ci_hi"])),
        axis=1,
    )
    df["ci_width"] = df["ci_hi"].astype(float) - df["ci_lo"].astype(float)
    return df


def _colour(name: str) -> str:
    return RX_COLORS.get(name, BASELINE_COLOR)


# ---------------------------------------------------------------------------
# Figure 1 — Per-seed bias distribution by DGP (box plot)
# ---------------------------------------------------------------------------

def plot_bias_by_dgp(df: pd.DataFrame):
    dgps = list(df["dgp"].unique())
    n = len(dgps)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.5 * n), squeeze=False)

    for i, dgp in enumerate(dgps):
        ax = axes[i, 0]
        sub = df[df["dgp"] == dgp]
        order = (sub.groupby("estimator")["abs_bias"].median()
                 .sort_values().index.tolist())
        data = [sub[sub["estimator"] == e]["bias"].values for e in order]
        colours = [_colour(e) for e in order]

        bp = ax.boxplot(data, patch_artist=True, tick_labels=order, vert=True)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.6, label="True ATE")
        ax.set_title(f"DGP: {dgp}  (true ATE = {sub['tau_true'].iloc[0]})")
        ax.set_ylabel("ATE − τ_true")
        ax.tick_params(axis="x", rotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")

        # Symmetric log scale when bias range exceeds 3 orders of magnitude
        rng = max(abs(np.concatenate(data).min()), abs(np.concatenate(data).max()))
        if rng > 100:
            ax.set_yscale("symlog", linthresh=1.0)

        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / "bias_by_dgp.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 2 — RMSE horizontal bar chart (one panel per DGP)
# ---------------------------------------------------------------------------

def plot_rmse(df: pd.DataFrame):
    dgps = list(df["dgp"].unique())
    n = len(dgps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for i, dgp in enumerate(dgps):
        ax = axes[0, i]
        sub = df[df["dgp"] == dgp]
        rmse = (sub.groupby("estimator")
                .apply(lambda g: float(np.sqrt(np.mean(g["bias"] ** 2))),
                       include_groups=False)
                .sort_values())
        colours = [_colour(e) for e in rmse.index]
        ax.barh(rmse.index, rmse.values, color=colours, alpha=0.75)
        ax.set_title(f"DGP: {dgp}")
        ax.set_xlabel("RMSE")
        if rmse.max() > 10:
            ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("RMSE by estimator (lower is better)", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "rmse_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Coverage vs CI width (calibration scatter)
# ---------------------------------------------------------------------------

def plot_coverage_vs_width(df: pd.DataFrame):
    dgps = list(df["dgp"].unique())
    n = len(dgps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for i, dgp in enumerate(dgps):
        ax = axes[0, i]
        sub = df[df["dgp"] == dgp]
        sub = sub[sub["ci_lo"].notna() & sub["ci_hi"].notna()]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No CI reported", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"DGP: {dgp}")
            continue

        agg = (sub.groupby("estimator")
               .agg(coverage=("covered", "mean"),
                    ci_width=("ci_width", "mean")))

        for est, row in agg.iterrows():
            c = _colour(est)
            ax.scatter(row["ci_width"], row["coverage"],
                       s=180, color=c, alpha=0.8, edgecolors="black")
            ax.annotate(est, (row["ci_width"], row["coverage"]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8)

        ax.axhline(0.95, color="green", linestyle="--", alpha=0.6,
                   label="Nominal 95 %")
        ax.set_xscale("log")
        ax.set_xlabel("Mean CI width  (log)")
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"DGP: {dgp}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Coverage × efficiency  (top-left corner = best)", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "coverage_vs_width.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Runtime bar chart
# ---------------------------------------------------------------------------

def plot_runtime(df: pd.DataFrame):
    dgps = list(df["dgp"].unique())
    n = len(dgps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for i, dgp in enumerate(dgps):
        ax = axes[0, i]
        sub = df[df["dgp"] == dgp]
        rt = sub.groupby("estimator")["runtime"].mean().sort_values()
        colours = [_colour(e) for e in rt.index]
        ax.barh(rt.index, rt.values, color=colours, alpha=0.75)
        ax.set_title(f"DGP: {dgp}")
        ax.set_xlabel("Mean wall-clock seconds per fit")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("Runtime by estimator", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "runtime_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    df = _load()
    plot_bias_by_dgp(df)
    plot_rmse(df)
    plot_coverage_vs_width(df)
    plot_runtime(df)
    print(f"\nAll figures → {FIG_DIR}")


if __name__ == "__main__":
    main()
