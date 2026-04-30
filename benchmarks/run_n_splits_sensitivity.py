"""
`n_splits` sensitivity on whale — tests the Sert-framework claim.

The README states RX-Learner "synthesizes the standard X-Learner with the
Sert et al. sample-splitting and bias-correction framework to isolate
high-dimensional nuisance estimation." That framework's load-bearing
mechanism is cross-fitting: μ̂₀(x_i), μ̂₁(x_i), π̂(x_i) are predicted on
observations NOT used to fit those nuisance models.

Prediction. On whale at N=2000:
  - n_splits=2 (documented default): each fold's train contains ~half the
    whales, contaminating μ̂₀ systemically → largest bias.
  - n_splits=5: each fold's train contains ~80 % whales (16 of 20), each
    leaf absorbs fewer whales in absolute terms since leaf_size drops.
    Modest improvement.
  - n_splits=10: even smaller folds, but each with 18 of 20 whales in
    train. Marginal further improvement; variance may increase.
  - Plateau expected between 5 and 10.

Standard DGP should be ~flat across n_splits (no outliers to isolate).

Usage:
    python -m benchmarks.run_n_splits_sensitivity --seeds 8
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import standard_dgp, whale_dgp


RESULTS_DIR = Path(__file__).parent / "results"
N = 2000
SPLITS = [2, 3, 5, 10]
DGPS = {"standard": standard_dgp, "whale": whale_dgp}


def fit_rx(X, Y, W, n_splits):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=n_splits, num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34, use_student_t=False,
        use_overlap=False, random_state=42,
    )
    model.fit(X, Y, W)
    mean, lo, hi = model.predict()
    return float(np.asarray(mean).mean()), float(np.asarray(lo).mean()), float(np.asarray(hi).mean())


def _run(seeds):
    rows = []
    for dgp_name, dgp_fn in DGPS.items():
        for seed in seeds:
            X, Y, W, tau = dgp_fn(N=N, seed=seed)
            tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
            for splits in SPLITS:
                t0 = time.time()
                try:
                    ate, lo, hi = fit_rx(X, Y, W, splits)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "dgp": dgp_name, "n_splits": splits, "seed": seed,
                    "tau_true": tau_true,
                    "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(f"  {dgp_name:<10}  splits={splits:<2}  seed={seed:<2}  "
                      f"ate={ate:+.3f}  cov={'Y' if cov else 'N'}  ({rt:.1f}s)"
                      + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["dgp", "n_splits"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
                  "mean_runtime": g["runtime"].mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "n_splits_sensitivity.md"
    lines = [
        "# `n_splits` sensitivity — tests the Sert-framework claim",
        "",
        f"N fixed at {N}. Cross-fitting folds `n_splits` swept over {SPLITS}. "
        f"All other settings fixed at RX-Learner (robust). Seeds: {list(seeds)}.",
        "",
        "Prediction: on whale, higher `n_splits` reduces bias (smaller "
        "training folds → more whale concentration per fold relative to "
        "leaf capacity → tighter whale isolation). Plateau expected by "
        "`n_splits=5`. Standard DGP should be flat.",
        "",
    ]
    for dgp in agg["dgp"].unique():
        sub = agg[agg["dgp"] == dgp].sort_values("n_splits")
        tau_true = df[df["dgp"] == dgp]["tau_true"].iloc[0]
        lines += [
            f"## DGP: `{dgp}` (true ATE = {tau_true:.2f})",
            "",
            "| n_splits | n | Bias | RMSE | Coverage | CI Width | Runtime (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {int(r['n_splits'])} | {int(r['n'])} | "
                f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} | "
                f"{r['mean_runtime']:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "n_splits_sensitivity_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'n_splits_sensitivity_raw.csv'}")


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
