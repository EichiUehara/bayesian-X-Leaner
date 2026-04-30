"""
Sensitivity of the Welsch tuning constant c_whale.

The Welsch loss ρ(r) = c²(1 − exp(−r²/2c²)) has influence function
ψ(r) = r · exp(−r²/2c²), which:
  - approaches L² (r) for |r| ≪ c,
  - attains max influence at r = c (influence = c / √e),
  - decays to 0 for |r| ≫ c.

Under a Gaussian likelihood with unit variance, MAD-scaling at c = 1.34
gives ~95% asymptotic efficiency — the Huber (1981) default. But the
default's *practical* behaviour at our DGP scale has not been verified.

Prediction: U-shape on whale —
  - c too small (0.5) → Welsch kills informative residuals, bias grows.
  - c at 1.34 or a bit higher (2.0) → sweet spot.
  - c ≫ 1 (5, 20) → approaches L² loss, RMSE explodes like std variant.
Standard DGP should be ~flat (no outliers to redescend on).

Usage:
    python -m benchmarks.run_c_whale_sensitivity --seeds 8
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

DGPS = {"standard": standard_dgp, "whale": whale_dgp}
C_VALUES = [0.5, 1.0, 1.34, 2.0, 5.0, 20.0]


def fit_rx(X, Y, W, c_whale):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=c_whale, use_student_t=False,
        use_overlap=False, random_state=42,
    )
    model.fit(X, Y, W)
    mean, lo, hi = model.predict()
    return float(np.asarray(mean).mean()), float(np.asarray(lo).mean()), float(np.asarray(hi).mean())


def _run(seeds):
    rows = []
    for dgp_name, dgp_fn in DGPS.items():
        for seed in seeds:
            X, Y, W, tau = dgp_fn(seed=seed)
            tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
            for c in C_VALUES:
                t0 = time.time()
                try:
                    ate, lo, hi = fit_rx(X, Y, W, c)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "dgp": dgp_name, "c_whale": c, "seed": seed,
                    "tau_true": tau_true, "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(f"  {dgp_name:<10}  seed={seed:<2}  c={c:<5}  "
                      f"ate={ate:+.3f}  cov={'Y' if cov else 'N'}  ({rt:.1f}s)"
                      + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["dgp", "c_whale"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "c_whale_sensitivity.md"
    lines = [
        "# `c_whale` sensitivity",
        "",
        f"Seeds: {list(seeds)}. Welsch loss tuning constant `c_whale` swept "
        f"over {C_VALUES}. All other settings fixed at RX-Learner (robust).",
        "",
        "Prediction: RMSE is U-shaped on `whale` (c too small ignores good "
        "residuals; c too large ≈ L²) and flat on `standard` (no outliers).",
        "",
    ]
    for dgp in agg["dgp"].unique():
        sub = agg[agg["dgp"] == dgp].sort_values("c_whale")
        tau_true = df[df["dgp"] == dgp]["tau_true"].iloc[0]
        lines += [
            f"## DGP: `{dgp}` (true ATE = {tau_true:.2f})",
            "",
            "| c_whale | n | Bias | RMSE | Coverage | CI Width |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['c_whale']:.2f} | {int(r['n'])} | "
                f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "c_whale_sensitivity_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'c_whale_sensitivity_raw.csv'}")


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
