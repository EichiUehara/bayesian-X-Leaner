"""§15 — Does robust nuisance subsume the downstream Welsch/MAD/prior story?

§14 revealed that the §11 catastrophe traces to MAD rescaling of
`c_whale` amplifying Welsch's constant because pseudo-outcome MAD is
itself contaminated. That pathology only fires when the non-whale
pseudo-outcomes are systemically shifted — i.e. when μ̂₀ is contaminated
by whales. If the nuisance outcome learner is natively robust (Huber
loss), then μ̂₀ is clean, non-whale D₀ values centre near the true CATE,
MAD stays small, dynamic c_whale stays near 1.34, and Welsch does the
job it was designed to do.

This tests whether changing the *upstream* loss (Huber vs MSE) makes the
downstream Welsch/MAD/prior tangle moot.

Configs (outcome nuisance only; propensity stays classifier):
    - xgb_mse:      XGBRegressor, default squared-error                (baseline)
    - xgb_huber:    XGBRegressor, objective='reg:pseudohubererror'     (test)
    - catboost_mse: CatBoostRegressor, default RMSE                    (baseline)
    - catboost_huber: CatBoostRegressor, loss_function='Huber:delta=1' (test)

All four run under the production code path (robust=True, MAD rescale
on, prior_scale=10.0) to see whether robust nuisance alone rescues it.

Usage:
    python -m benchmarks.run_robust_nuisance --seeds 8
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
DENSITIES = [0.01, 0.05, 0.10, 0.20]

CONFIGS = {
    "xgb_mse": dict(
        method="xgboost",
        outcome={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
    ),
    "xgb_huber": dict(
        method="xgboost",
        outcome={"max_depth": 4, "n_estimators": 150, "verbosity": 0,
                  "objective": "reg:pseudohubererror", "huber_slope": 1.0},
    ),
    "catboost_mse": dict(
        method="catboost",
        outcome={"depth": 4, "iterations": 150},
    ),
    "catboost_huber": dict(
        method="catboost",
        outcome={"depth": 4, "iterations": 150,
                  "loss_function": "Huber:delta=1.0"},
    ),
}
PROPENSITY_XGB = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
PROPENSITY_CATBOOST = {"depth": 4, "iterations": 150}


def fit(X, Y, W, cfg):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    prop_params = PROPENSITY_CATBOOST if cfg["method"] == "catboost" else PROPENSITY_XGB
    model = TargetedBayesianXLearner(
        outcome_model_params=cfg["outcome"],
        propensity_model_params=prop_params,
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, use_overlap=False, random_state=42,
        robust=True, use_student_t=False, prior_scale=10.0,
        nuisance_method=cfg["method"],
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
            for name, cfg in CONFIGS.items():
                t0 = time.time()
                try:
                    ate, lo, hi = fit(X, Y, W, cfg)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "density": density, "n_whales": n_whales, "seed": seed,
                    "nuisance": name, "tau_true": tau_true,
                    "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(
                    f"  dens={density:<5} seed={seed:<2} {name:<16} "
                    f"ATE={ate:+.3f} cov={'Y' if cov else 'N'} ({rt:.1f}s)"
                    + (f" ERR={err}" if err else "")
                )
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["density", "nuisance"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "median_ate": g["ate"].median(),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
                  "runtime_s": g["runtime"].mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "robust_nuisance.md"
    lines = [
        "# Robust nuisance subsumption test",
        "",
        f"N fixed at {N}. Whale density ∈ {[f'{d*100:g}%' for d in DENSITIES]}. "
        f"Nuisance outcome learner ∈ {list(CONFIGS)}. Production code path "
        f"elsewhere (robust=True, MAD rescale on, `prior_scale=10.0`, "
        f"`n_splits=2`, depth-4). Seeds: {list(seeds)}.",
        "",
        "Question: if the nuisance outcome learner is natively robust "
        "(Huber loss), does the downstream Welsch/MAD/prior tangle "
        "(§14) become unnecessary?",
        "",
    ]
    for density in sorted(agg["density"].unique()):
        sub = agg[agg["density"] == density].sort_values("nuisance")
        lines += [
            f"## density = {density*100:g}% ({int(sub['n'].iloc[0])} seeds × 4 configs)",
            "",
            "| nuisance | Bias | RMSE | Median ATE | Coverage | CI Width | Runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['nuisance']} | {r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['median_ate']:+.3f} | {r['coverage']:.2f} | "
                f"{r['mean_ci_width']:.3f} | {r['runtime_s']:.1f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "robust_nuisance_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'robust_nuisance_raw.csv'}")


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
