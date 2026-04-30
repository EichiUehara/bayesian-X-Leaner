"""§16 — Can we eliminate residual nuisance contamination under heavy whale load?

§15 showed CatBoost-Huber gets RMSE 0.20 at density 20 %, but coverage
is 0.12 — the bias is real (~0.19), not a CI width artefact. The
diagnostic at /tmp/diagnose_cb_huber.py attributes it to μ̂₀
contamination: at 20 % density, 40 % of the control training set is
whales, and Huber(δ=1.0) only bounds per-point residual influence (L¹
tail) without rejecting the coherent shift from 200 whales.

This sweep tries to push the residual bias down:

  - Loss function ∈ {Huber:0.1, Huber:0.5, Huber:1.0, Quantile:0.5 (MAE/median),
                      Quantile:0.25, Quantile:0.75, RMSE}
  - max_depth ∈ {4, 8}

Hypothesis: Quantile:0.5 (median regression) is fully robust against
sign-symmetric contamination. With enough tree depth to separate whale
and non-whale leaves, the non-whale μ̂₀ should match the clean leaf
median ≈ 0 (true μ₀ is Normal), removing the downstream bias.

Fixed at production defaults everywhere else (N=1000, density=20 %,
robust=True, MAD rescale on, prior_scale=10, n_splits=2, c_whale=1.34).

Usage:
    python -m benchmarks.run_nuisance_loss_sweep --seeds 8
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
DENSITY = 0.20

LOSSES = [
    "Huber:delta=0.1",
    "Huber:delta=0.5",
    "Huber:delta=1.0",
    "Quantile:alpha=0.5",
    "RMSE",
]
DEPTHS = [4, 8]


def fit(X, Y, W, loss_fn, depth):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    outcome_params = {"depth": depth, "iterations": 150,
                       "loss_function": loss_fn}
    propensity_params = {"depth": depth, "iterations": 150}
    model = TargetedBayesianXLearner(
        outcome_model_params=outcome_params,
        propensity_model_params=propensity_params,
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, use_overlap=False, random_state=42,
        robust=True, use_student_t=False, prior_scale=10.0,
        nuisance_method="catboost",
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
    n_whales = int(round(DENSITY * N))
    for seed in seeds:
        X, Y, W, tau = whale_dgp(N=N, n_whales=n_whales, seed=seed)
        tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
        for loss_fn in LOSSES:
            for depth in DEPTHS:
                t0 = time.time()
                try:
                    ate, lo, hi = fit(X, Y, W, loss_fn, depth)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "seed": seed, "loss": loss_fn, "depth": depth,
                    "tau_true": tau_true,
                    "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(
                    f"  seed={seed:<2} loss={loss_fn:<20} depth={depth:<2} "
                    f"ATE={ate:+.3f} cov={'Y' if cov else 'N'} ({rt:.1f}s)"
                    + (f" ERR={err}" if err else "")
                )
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["loss", "depth"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "median_ate": g["ate"].median(),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "nuisance_loss_sweep.md"
    lines = [
        "# Nuisance-loss sweep — eliminating residual contamination at 20 % whale density",
        "",
        f"N={N}, whale density=20 %, nuisance=catboost, robust=True, MAD rescale on, "
        f"`prior_scale=10.0`, `n_splits=2`. Outcome `loss_function` ∈ {LOSSES}, "
        f"`depth` ∈ {DEPTHS}. Seeds: {list(seeds)}.",
        "",
        "Context: §15's CatBoost-Huber(δ=1) config delivered RMSE 0.20 at this "
        "density but coverage 0.12 — a systematic bias of ~0.19 comes from μ̂₀ "
        "contamination the Huber L¹ tail can't fully remove when 40 % of the "
        "control training set is whales. This sweep tests whether tighter-δ "
        "Huber, median regression (Quantile:0.5), or deeper trees cures it.",
        "",
        "| loss | depth | Bias | RMSE | Median ATE | Coverage | CI Width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.sort_values(["loss", "depth"]).iterrows():
        lines.append(
            f"| {r['loss']} | {int(r['depth'])} | "
            f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
            f"{r['median_ate']:+.3f} | {r['coverage']:.2f} | "
            f"{r['mean_ci_width']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "nuisance_loss_sweep_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'nuisance_loss_sweep_raw.csv'}")


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
