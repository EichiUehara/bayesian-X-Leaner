"""Quantile-DML — paradigm-(iii) baseline closer in spirit to OQRF.

Orthogonal/Quantile Random Forest (Athey-Tibshirani-Wager) is not
natively implemented in Python; we approximate the spirit with a
DML-residualised quantile regression on DR pseudo-outcomes.

Pipeline:
  1. Cross-fitted nuisance fits for μ_w, π (HistGradientBoosting).
  2. DR pseudo-outcomes D_i.
  3. Quantile regression of D on the same basis as RX-Learner,
     using HistGradientBoostingRegressor(loss='quantile') for
     conditional quantiles q ∈ {0.5, 0.75, 0.95}.

The point-estimator is benchmarked on the whale DGP at four
contamination densities. We report the q-th conditional quantile of
the DR pseudo-outcomes — a paradigm-(iii) "tail is the target"
estimand — rather than the conditional mean.

Usage: python -u -m benchmarks.run_quantile_dml --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from benchmarks.dgps import whale_dgp


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0
DENSITIES = [0.00, 0.01, 0.05, 0.20]
QUANTILES = [0.5, 0.75, 0.95]


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def _make_qreg(q):
    return HistGradientBoostingRegressor(
        loss='quantile', quantile=q, max_iter=150, max_depth=4, random_state=42)


def quantile_dml(X, Y, W, q):
    """DML-style: cross-fitted DR pseudo-outcomes, then quantile
    regression of D on X for the conditional q-th quantile τ_q(x).
    Reports the marginal mean of τ_q(x) over the population."""
    K = 2
    n = len(X)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), K)

    D = np.full(n, np.nan)
    for k in range(K):
        idx_test = folds[k]
        idx_train = np.concatenate([folds[j] for j in range(K) if j != k])
        Xt = X[idx_train]; Yt = Y[idx_train]; Wt = W[idx_train]
        Xs = X[idx_test]
        mu0 = _make_reg(); mu0.fit(Xt[Wt == 0], Yt[Wt == 0])
        mu1 = _make_reg(); mu1.fit(Xt[Wt == 1], Yt[Wt == 1])
        pi_m = _make_clf(); pi_m.fit(Xt, Wt)
        pi_s = np.clip(pi_m.predict_proba(Xs)[:, 1], 0.05, 0.95)
        mu0_s = mu0.predict(Xs); mu1_s = mu1.predict(Xs)
        Y_s = Y[idx_test]; W_s = W[idx_test]
        D[idx_test] = np.where(
            W_s == 1,
            mu1_s - mu0_s + (Y_s - mu1_s) / pi_s,
            mu1_s - mu0_s - (Y_s - mu0_s) / (1.0 - pi_s),
        )
    qreg = _make_qreg(q)
    qreg.fit(X, D)
    return float(np.mean(qreg.predict(X)))


def _run(seeds):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_whales = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            for q in QUANTILES:
                t0 = time.time()
                try:
                    est = quantile_dml(X, Y, W, q)
                except Exception as e:
                    est = float("nan"); print(f"  ERR {e}")
                rt = time.time() - t0
                rows.append({
                    "seed": seed, "density": density, "n_whales": n_whales,
                    "quantile": q, "estimate": est, "runtime": rt,
                })
                print(f"  seed={seed} p={density:.2f} q={q:.2f} τ̂_q={est:+.3f} "
                      f"({rt:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby(["density", "quantile"]).agg(
        n=("seed", "count"),
        estimate_mean=("estimate", "mean"),
        estimate_std=("estimate", "std"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "quantile_dml.md"
    lines = [
        "# Quantile-DML — OQRF-style paradigm-(iii) baseline",
        "",
        f"DML-residualised quantile regression of DR pseudo-outcomes.",
        f"Whale DGP, N = {N}, 3 seeds.",
        "Estimand: conditional q-th quantile of τ(x), averaged over the population.",
        "",
        "| density | quantile | n | mean estimate | std |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {r['quantile']:.2f} | {int(r['n'])} | "
            f"{r['estimate_mean']:+.3f} | {r['estimate_std']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "quantile_dml_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──"); print(agg.to_string(index=False))
    _write_markdown(df, agg)


if __name__ == "__main__":
    main()
