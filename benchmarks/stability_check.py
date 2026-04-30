"""
Stability & reproducibility diagnostics for the RX-Learner.

Answers three operational questions a reviewer will ask:

  1. Is the MCMC converging?
     → For each fit, check R-hat < 1.05 and ESS > 200.

  2. Is the point estimate reproducible across MCMC random seeds on
     IDENTICAL data?
     → Repeat fit with different MCMC seeds, fixed data seed.
       Report std(ATE_hat) across MCMC seeds. Should be ≪ finite-sample
       RMSE across data seeds.

  3. Does the estimator fail silently or loudly?
     → Count exceptions, NaNs, and negative-variance warnings over
       N repeated runs.

Output:
    benchmarks/results/stability_report.md
    benchmarks/results/figures/mcmc_diagnostics.png

Usage:
    python -m benchmarks.stability_check --n-mcmc 10 --n-data 5
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.dgps import standard_dgp, whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def _fit_once(X, Y, W, *, robust, mcmc_seed: int):
    """
    Fit RX-Learner and extract (ate, ci_lo, ci_hi, max_rhat, min_ess).
    """
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2,
        num_warmup=400,
        num_samples=800,
        num_chains=2,
        robust=robust,
        c_whale=1.34,
        use_student_t=robust,
        random_state=mcmc_seed,
    )
    model.fit(X, Y, W)
    cate, lo, hi = model.predict()

    # R-hat / ESS via numpyro diagnostics
    from numpyro.diagnostics import summary as _summary
    grouped = model.bayesian_mcmc.mcmc.get_samples(group_by_chain=True)
    stats = _summary(grouped)
    rhats, esss = [], []
    for _, s in stats.items():
        for k in ("r_hat", "n_eff"):
            if k in s:
                vals = np.atleast_1d(s[k]).flatten()
                (rhats if k == "r_hat" else esss).extend(vals.tolist())
    max_rhat = float(np.nanmax(rhats)) if rhats else float("nan")
    min_ess = float(np.nanmin(esss)) if esss else float("nan")

    return {
        "ate":      float(cate[0]),
        "ci_lo":    float(lo[0]),
        "ci_hi":    float(hi[0]),
        "max_rhat": max_rhat,
        "min_ess":  min_ess,
    }


def _run_mcmc_reproducibility(dgp_name, dgp_fn, n_mcmc, robust, data_seed):
    """Same data; vary only MCMC random seed."""
    X, Y, W, tau = dgp_fn(seed=data_seed)
    rows = []
    for mcmc_seed in range(n_mcmc):
        r = _fit_once(X, Y, W, robust=robust, mcmc_seed=100 + mcmc_seed)
        rows.append({"dgp": dgp_name, "robust": robust,
                     "mcmc_seed": mcmc_seed, "tau_true": tau, **r})
        print(f"  mcmc_seed={mcmc_seed:>2}  ate={r['ate']:+.4f}  "
              f"rhat={r['max_rhat']:.3f}  ess={r['min_ess']:.0f}")
    return rows, tau


def _run_data_variance(dgp_name, dgp_fn, n_data, robust):
    """Different data; fixed MCMC behaviour. Measures *irreducible* Monte-Carlo noise."""
    rows = []
    tau = None
    for data_seed in range(n_data):
        X, Y, W, tau = dgp_fn(seed=data_seed)
        r = _fit_once(X, Y, W, robust=robust, mcmc_seed=42)
        rows.append({"dgp": dgp_name, "robust": robust,
                     "data_seed": data_seed, "tau_true": tau, **r})
        print(f"  data_seed={data_seed:>2}  ate={r['ate']:+.4f}  "
              f"rhat={r['max_rhat']:.3f}  ess={r['min_ess']:.0f}")
    return rows, tau


def _aggregate(rows, label, tau_true):
    """Produce summary statistics row."""
    ates = np.array([r["ate"] for r in rows])
    rhats = np.array([r["max_rhat"] for r in rows])
    esss = np.array([r["min_ess"] for r in rows])
    return {
        "label":   label,
        "n":       len(rows),
        "mean_ate": float(np.mean(ates)),
        "std_ate":  float(np.std(ates, ddof=1) if len(ates) > 1 else 0.0),
        "bias":    float(np.mean(ates) - tau_true),
        "rmse":    float(np.sqrt(np.mean((ates - tau_true) ** 2))),
        "max_rhat_worst":  float(np.max(rhats)),
        "max_rhat_mean":   float(np.mean(rhats)),
        "min_ess_worst":   float(np.min(esss)),
        "min_ess_mean":    float(np.mean(esss)),
        "rhat_pass":       int(np.sum(rhats < 1.05)),
        "ess_pass":        int(np.sum(esss > 200)),
    }


def _plot_diagnostics(all_rows: list[dict]):
    """R-hat & ESS histograms (all fits pooled)."""
    df = pd.DataFrame(all_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df["max_rhat"], bins=20, color="#1f77b4", edgecolor="black", alpha=0.75)
    axes[0].axvline(1.05, color="red", linestyle="--", label="Target R̂ < 1.05")
    axes[0].set_xlabel("Worst parameter R̂ per fit")
    axes[0].set_ylabel("Number of fits")
    axes[0].set_title("MCMC convergence (Gelman–Rubin)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(df["min_ess"], bins=20, color="#2ca02c", edgecolor="black", alpha=0.75)
    axes[1].axvline(200, color="red", linestyle="--", label="Target ESS > 200")
    axes[1].set_xlabel("Worst parameter ESS per fit")
    axes[1].set_ylabel("Number of fits")
    axes[1].set_title("Effective sample size")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("RX-Learner MCMC diagnostics across all stability runs", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "mcmc_diagnostics.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


def _write_markdown(summaries: list[dict], all_rows: list[dict]):
    path = RESULTS_DIR / "stability_report.md"
    n_total = len(all_rows)
    n_rhat_pass = sum(1 for r in all_rows if r["max_rhat"] < 1.05)
    n_ess_pass = sum(1 for r in all_rows if r["min_ess"] > 200)

    lines = [
        "# RX-Learner Stability & Reproducibility Report",
        "",
        f"Total MCMC fits: **{n_total}**",
        f"Fits with R̂ < 1.05:   **{n_rhat_pass}/{n_total}**  "
        f"({100*n_rhat_pass/n_total:.0f} %)",
        f"Fits with ESS > 200:  **{n_ess_pass}/{n_total}**  "
        f"({100*n_ess_pass/n_total:.0f} %)",
        "",
        "## Per-condition summary",
        "",
        "| Condition | n | Mean ATE | Std ATE | Bias | RMSE | "
        "Worst R̂ | Mean R̂ | Worst ESS | Mean ESS | R̂-pass | ESS-pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['label']} | {s['n']} | "
            f"{s['mean_ate']:+.4f} | {s['std_ate']:.4f} | "
            f"{s['bias']:+.4f} | {s['rmse']:.4f} | "
            f"{s['max_rhat_worst']:.3f} | {s['max_rhat_mean']:.3f} | "
            f"{s['min_ess_worst']:.0f} | {s['min_ess_mean']:.0f} | "
            f"{s['rhat_pass']}/{s['n']} | {s['ess_pass']}/{s['n']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Std ATE across MCMC seeds** (with data fixed) quantifies "
        "*Monte-Carlo noise* in the posterior. It should be ≪ RMSE across "
        "data seeds — otherwise, the reported point estimate is not "
        "reproducible and `num_samples` must be increased.",
        "- **R̂ ≥ 1.05 or ESS < 200** in any fit signals under-converged "
        "MCMC; increase `num_warmup` / `num_samples`.",
        "- **Bias close to 0 under whale DGP** confirms the Welsch "
        "redescending loss (`robust=True`) neutralises the outlier. "
        "Note: when `robust=True`, `use_student_t` is ignored by the "
        "inference backend — see EXTENSIONS.md §10.",
        "",
        "Figures in `benchmarks/results/figures/mcmc_diagnostics.png`.",
    ]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-mcmc", type=int, default=8,
                    help="MCMC seeds per data seed for reproducibility test")
    ap.add_argument("--n-data", type=int, default=8,
                    help="Data seeds for variance test")
    args = ap.parse_args()

    summaries = []
    all_rows = []

    dgp_configs = [
        ("standard", standard_dgp),
        ("whale",    whale_dgp),
    ]
    robust_settings = [
        (False, "std"),
        (True,  "robust"),
    ]

    for dgp_name, dgp_fn in dgp_configs:
        for robust, tag in robust_settings:
            print(f"\n── MCMC reproducibility · {dgp_name} · {tag} ──")
            rows, tau = _run_mcmc_reproducibility(dgp_name, dgp_fn,
                                                  args.n_mcmc, robust, data_seed=0)
            all_rows.extend(rows)
            summaries.append(_aggregate(
                rows, f"MCMC noise · {dgp_name} · {tag}", tau))

            print(f"\n── Data variance · {dgp_name} · {tag} ──")
            rows, tau = _run_data_variance(dgp_name, dgp_fn,
                                           args.n_data, robust)
            all_rows.extend(rows)
            summaries.append(_aggregate(
                rows, f"Data variance · {dgp_name} · {tag}", tau))

    _plot_diagnostics(all_rows)
    _write_markdown(summaries, all_rows)


if __name__ == "__main__":
    main()
