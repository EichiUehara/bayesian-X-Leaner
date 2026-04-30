"""
Prior-scale sensitivity in the Welsch high-contamination failure regime.

§11 revealed that at ≥5 % whale density the robust variant diverges with
large negative ATE estimates (e.g. −1534 at 20 %). The Welsch redescending
loss bounds the influence of large residuals, which can produce a bimodal
pseudo-likelihood when a majority of observations are contaminated: one
mode tracks the clean signal (ATE ≈ +2), the other locks onto the whale
cluster. With the default wide beta prior (Normal(0, 10)) NUTS has no
regularization penalty that separates the two modes and can fixate on the
wrong one.

Prediction. Tightening the beta prior (Normal(0, σ) with σ ≪ 10) should
shrink the wrong-mode posterior toward zero and let NUTS converge on the
correct mode. The effect should be monotone in σ and most visible at the
highest density (20 %).

Usage:
    python -m benchmarks.run_prior_scale_sensitivity --seeds 8
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import whale_dgp


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
DENSITIES = [0.05, 0.10, 0.20]
PRIOR_SCALES = [0.5, 1.0, 2.0, 5.0, 10.0]


def fit_rx(X, Y, W, prior_scale):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, use_overlap=False, random_state=42,
        robust=True, use_student_t=False,
        prior_scale=prior_scale,
    )
    model.fit(X, Y, W)
    mean, lo, hi = model.predict()
    return (
        float(np.asarray(mean).mean()),
        float(np.asarray(lo).mean()),
        float(np.asarray(hi).mean()),
    )


def _run(seeds):
    rows = []
    for density in DENSITIES:
        n_whales = max(1, int(round(density * N)))
        for seed in seeds:
            X, Y, W, tau = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
            for prior_scale in PRIOR_SCALES:
                t0 = time.time()
                try:
                    ate, lo, hi = fit_rx(X, Y, W, prior_scale)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "density": density, "n_whales": n_whales, "seed": seed,
                    "prior_scale": prior_scale, "tau_true": tau_true,
                    "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(
                    f"  density={density:<5} seed={seed:<2} "
                    f"prior_scale={prior_scale:<4} ate={ate:+.3f} "
                    f"cov={'Y' if cov else 'N'} ({rt:.1f}s)"
                    + (f"  ERR={err}" if err else "")
                )
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["density", "prior_scale"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "n_whales": int(g["n_whales"].iloc[0]),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "median_ate": g["ate"].median(),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
                  "mode_flip_rate": (g["ate"] < 0).mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "prior_scale_sensitivity.md"
    lines = [
        "# Prior-scale sensitivity — high-contamination mode-flip",
        "",
        f"N fixed at {N}. Whale density swept over "
        f"{[f'{d*100:g}%' for d in DENSITIES]}, beta-prior scale "
        f"`Normal(0, σ)` swept over {PRIOR_SCALES}. Robust (Welsch) "
        f"variant only. Seeds: {list(seeds)}.",
        "",
        "Background: §11 showed robust fits at density ≥ 5 % diverge with "
        "large negative ATE (−1534 at 20 %). Diagnosis: Welsch's "
        "redescending loss creates a bimodal posterior under majority "
        "contamination; the default wide prior `Normal(0, 10)` does not "
        "distinguish the modes. `mode_flip_rate` reports the fraction of "
        "seeds whose posterior mean ATE came out < 0 (true ATE = +2).",
        "",
    ]
    for density in sorted(agg["density"].unique()):
        sub = agg[agg["density"] == density].sort_values("prior_scale")
        n_whales = int(sub["n_whales"].iloc[0])
        lines += [
            f"## density = {density*100:g}% ({n_whales} whales)",
            "",
            "| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width | Mode-flip rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['prior_scale']} | {int(r['n'])} | "
                f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['median_ate']:+.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} | "
                f"{r['mode_flip_rate']:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "prior_scale_sensitivity_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'prior_scale_sensitivity_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, seeds)


if __name__ == "__main__":
    main()
