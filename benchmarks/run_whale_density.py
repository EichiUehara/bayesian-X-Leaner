"""
Whale-density breakdown boundary.

§9 (sample-size scaling) showed that at fixed 1 % whale density, RMSE grows
with N because XGBoost leaves absorb a growing absolute number of whales.
This holds N fixed at 1000 and sweeps density, drawing the practical
contamination tolerance of the robust variant.

Prediction. Welsch rejects local residual outliers but cannot undo
nuisance-model contamination. Leaf-mean shift ≈ (whales_per_leaf /
leaf_size) · whale_size. At depth-4 trees (~16 leaves, leaf_size ≈ N/16),
whales_per_leaf = density · leaf_size. Contamination per leaf grows
linearly with density. Robust should hold through density ~2-5% and fail
above (matching when the DR pseudo-outcome bias exceeds the Welsch `c`
threshold systemically).

Usage:
    python -m benchmarks.run_whale_density --seeds 8
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
DENSITIES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
VARIANTS = {
    "RX-Learner (std)":    dict(robust=False, use_student_t=False),
    "RX-Learner (robust)": dict(robust=True,  use_student_t=False),
}


def fit_rx(X, Y, W, variant_params, nuisance="xgb_mse"):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    if nuisance == "catboost_huber":
        kwargs = dict(
            outcome_model_params={"depth": 4, "iterations": 150,
                                   "loss_function": "Huber:delta=0.5"},
            propensity_model_params={"depth": 4, "iterations": 150},
            nuisance_method="catboost",
        )
    else:
        kwargs = dict(
            outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            nuisance_method="xgboost",
        )
    model = TargetedBayesianXLearner(
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, use_overlap=False, random_state=42,
        **kwargs, **variant_params,
    )
    model.fit(X, Y, W)
    mean, lo, hi = model.predict()
    return float(np.asarray(mean).mean()), float(np.asarray(lo).mean()), float(np.asarray(hi).mean())


def _run(seeds, densities, nuisance="xgb_mse"):
    rows = []
    for density in densities:
        n_whales = max(1, int(round(density * N)))
        for seed in seeds:
            X, Y, W, tau = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
            for name, params in VARIANTS.items():
                t0 = time.time()
                try:
                    ate, lo, hi = fit_rx(X, Y, W, params, nuisance=nuisance)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "density": density, "n_whales": n_whales, "seed": seed,
                    "variant": name, "tau_true": tau_true,
                    "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(f"  density={density:<5}  n_whales={n_whales:<3} "
                      f"seed={seed:<2}  {name:<22} ate={ate:+.3f} "
                      f"cov={'Y' if cov else 'N'} ({rt:.1f}s)"
                      + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["density", "variant"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "n_whales": int(g["n_whales"].iloc[0]),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds, densities, nuisance="xgb_mse"):
    suffix = "_catboost_huber" if nuisance == "catboost_huber" else ""
    path = RESULTS_DIR / f"whale_density{suffix}.md"
    lines = [
        "# Whale-density breakdown boundary",
        "",
        f"N fixed at {N}. Whale density swept over {[f'{d*100:g}%' for d in densities]} "
        f"({[max(1, int(round(d*N))) for d in densities]} whales). Seeds: {list(seeds)}.",
        "",
        "Prediction: robust variant holds through density ~2-5 % and fails "
        "above, set by when leaf-level whale concentration exceeds what "
        "XGBoost can isolate in single leaves. `std` variant fails at every "
        "density.",
        "",
    ]
    for variant in agg["variant"].unique():
        sub = agg[agg["variant"] == variant].sort_values("density")
        lines += [
            f"## {variant}",
            "",
            "| density | n_whales | n | Bias | RMSE | Coverage | CI Width |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['density']*100:.1f}% | {int(r['n_whales'])} | {int(r['n'])} | "
                f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    csv_path = RESULTS_DIR / f"whale_density_raw{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--nuisance", choices=["xgb_mse", "catboost_huber"],
                    default="xgb_mse",
                    help="Nuisance learner: xgb_mse (legacy) or catboost_huber (§16 default)")
    ap.add_argument("--densities", type=str, default=None,
                    help="Comma-separated density list, e.g. '0.01,0.1,0.2,0.3,0.4,0.5'")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    densities = DENSITIES if args.densities is None else [float(d) for d in args.densities.split(",")]
    df = _run(seeds, densities, nuisance=args.nuisance)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, seeds, densities, nuisance=args.nuisance)


if __name__ == "__main__":
    main()
