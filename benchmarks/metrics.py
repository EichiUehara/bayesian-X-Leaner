"""
Aggregation metrics across Monte Carlo seeds.

Given per-seed results for one (estimator × DGP) cell, compute:
    mean_ate, bias, rmse, coverage, mean_ci_width, mean_runtime, n_success.
"""

from __future__ import annotations
import numpy as np


def aggregate(results: list[dict], tau_true: float) -> dict:
    """
    Parameters
    ----------
    results : list of per-seed result dicts (from ``estimators``)
    tau_true : ground-truth ATE for this DGP

    Returns
    -------
    dict with aggregated metrics
    """
    ok = [r for r in results if r.get("ate") is not None]
    n_total = len(results)
    n_ok = len(ok)
    if n_ok == 0:
        return {
            "n_success": 0, "n_total": n_total,
            "mean_ate": float("nan"), "bias": float("nan"),
            "rmse": float("nan"), "coverage": float("nan"),
            "mean_ci_width": float("nan"), "mean_runtime": float("nan"),
        }

    ates = np.array([r["ate"] for r in ok], dtype=float)
    bias = float(np.mean(ates) - tau_true)
    mean_ate = float(np.mean(ates))
    rmse = float(np.sqrt(np.mean((ates - tau_true) ** 2)))

    # CI-based metrics (only over runs that reported a CI)
    with_ci = [r for r in ok if r.get("ci_lo") is not None and r.get("ci_hi") is not None]
    if with_ci:
        covered = [1.0 if (r["ci_lo"] <= tau_true <= r["ci_hi"]) else 0.0 for r in with_ci]
        widths = [r["ci_hi"] - r["ci_lo"] for r in with_ci]
        coverage = float(np.mean(covered))
        mean_ci_width = float(np.mean(widths))
    else:
        coverage = float("nan")
        mean_ci_width = float("nan")

    mean_rt = float(np.mean([r["runtime"] for r in ok]))

    return {
        "n_success": n_ok, "n_total": n_total,
        "mean_ate": mean_ate, "bias": bias,
        "rmse": rmse, "coverage": coverage,
        "mean_ci_width": mean_ci_width, "mean_runtime": mean_rt,
    }


def to_markdown_row(name: str, m: dict) -> str:
    """Format one aggregated result row as a markdown table row."""
    cov_str = "—" if np.isnan(m["coverage"]) else f"{m['coverage']:.2f}"
    wid_str = "—" if np.isnan(m["mean_ci_width"]) else f"{m['mean_ci_width']:.3f}"
    return (
        f"| {name} | {m['mean_ate']:+.3f} | {m['bias']:+.3f} | "
        f"{m['rmse']:.3f} | {cov_str} | {wid_str} | "
        f"{m['mean_runtime']:.2f} | {m['n_success']}/{m['n_total']} |"
    )


MARKDOWN_HEADER = (
    "| Estimator | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) | Success |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|"
)
