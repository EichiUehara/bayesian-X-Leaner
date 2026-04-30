"""
Nuisance tree-depth on whale — direct mechanism check for §9.

§9 *hypothesised* that robust RMSE grows with N on whale because each
XGBoost leaf absorbs a growing absolute number of whales, biasing μ̂₀
systemically. At depth-4 trees there are ~16 leaves, so with 20 whales
at N=2000 each leaf averages ~1.25 whales — mean-shifting its output.

Mechanism verification. If the hypothesis is correct, *deeper trees*
(more leaves → fewer whales per leaf + tighter isolation of whale leaves)
should reduce RMSE on whale at N=2000. Shallower trees should worsen it.

Prediction. max_depth ∈ {2, 4, 6, 8, 10} on whale N=2000:
  - depth 2  (4 leaves, ~5 whales each, severe pollution) → worst RMSE
  - depth 4  (documented default) → baseline §9 result (RMSE ~3-4)
  - depth 10 (~1024 leaves, whales in near-singleton leaves) → best RMSE

On standard (clean) this should be flat or slightly worse at higher depth
due to overfitting variance.

Usage:
    python -m benchmarks.run_nuisance_depth --seeds 8
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
DEPTHS = [2, 4, 6, 8, 10]
DGPS = {"standard": standard_dgp, "whale": whale_dgp}


def fit_rx(X, Y, W, max_depth):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    params = {"max_depth": max_depth, "n_estimators": 150, "verbosity": 0}
    model = TargetedBayesianXLearner(
        outcome_model_params=params,
        propensity_model_params=params,
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
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
            for depth in DEPTHS:
                t0 = time.time()
                try:
                    ate, lo, hi = fit_rx(X, Y, W, depth)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "dgp": dgp_name, "depth": depth, "seed": seed,
                    "tau_true": tau_true,
                    "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(f"  {dgp_name:<10}  depth={depth:<2}  seed={seed:<2}  "
                      f"ate={ate:+.3f}  cov={'Y' if cov else 'N'}  ({rt:.1f}s)"
                      + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["dgp", "depth"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "nuisance_depth.md"
    lines = [
        "# Nuisance tree-depth on whale — §9 mechanism check",
        "",
        f"N fixed at {N}. XGBoost `max_depth` swept over {DEPTHS} for both "
        f"outcome and propensity nuisance models. All other settings fixed "
        f"at RX-Learner (robust), `n_splits=2`, `c_whale=1.34`. Seeds: {list(seeds)}.",
        "",
        "Prediction: deeper trees (more leaves → fewer whales per leaf) "
        "should reduce RMSE on whale, directly verifying the §9 mechanism "
        "hypothesis. Standard DGP should be flat-to-worse with depth.",
        "",
    ]
    for dgp in agg["dgp"].unique():
        sub = agg[agg["dgp"] == dgp].sort_values("depth")
        tau_true = df[df["dgp"] == dgp]["tau_true"].iloc[0]
        lines += [
            f"## DGP: `{dgp}` (true ATE = {tau_true:.2f})",
            "",
            "| max_depth | n | Bias | RMSE | Coverage | CI Width |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {int(r['depth'])} | {int(r['n'])} | "
                f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "nuisance_depth_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'nuisance_depth_raw.csv'}")


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
