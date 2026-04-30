"""Quantile DR-learner — paradigm (iii) "tail is the target" alternative.

Group 1 paradigm (iii) of the heavy-tailed taxonomy
(Section~\\ref{sec:related:tails}): change the estimand from the mean
ATE to a quantile of the treatment effect. Quantile-DR is a quantile
regression of the DR pseudo-outcomes on the basis $\\phi(x)$.

We compare three quantile levels {0.5, 0.75, 0.95} against the mean
ATE on the whale DGP. The point of the exercise is to make concrete
what the "shift target to match the tail" paradigm actually delivers
when whales are signal vs noise.

Usage: python -u -m benchmarks.run_quantile_dr --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
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


def _dr_pseudo_outcomes(X, Y, W):
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
    mu0_all = mu0.predict(X); mu1_all = mu1.predict(X)
    return np.where(
        W == 1,
        mu1_all - mu0_all + (Y - mu1_all) / pi,
        mu1_all - mu0_all - (Y - mu0_all) / (1.0 - pi),
    )


def quantile_dr_at(X, Y, W, q):
    D = _dr_pseudo_outcomes(X, Y, W)
    Xones = np.ones((len(X), 1))
    reg = QuantileRegressor(quantile=q, alpha=0.0, solver='highs')
    reg.fit(Xones, D)
    return float(reg.intercept_)


def _run(seeds):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_whales = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            for q in QUANTILES:
                t0 = time.time()
                try:
                    est = quantile_dr_at(X, Y, W, q)
                except Exception as e:
                    est = float("nan"); print(f"  ERR {e}")
                rt = time.time() - t0
                rows.append({
                    "seed": seed, "density": density, "n_whales": n_whales,
                    "quantile": q, "estimate": est, "runtime": rt,
                })
                print(f"  seed={seed} p={density:.2f} q={q:.2f} "
                      f"τ̂_q={est:+.3f} ({rt:.2f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby(["density", "quantile"]).agg(
        n=("seed", "count"),
        estimate_mean=("estimate", "mean"),
        estimate_std=("estimate", "std"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "quantile_dr.md"
    lines = [
        "# Quantile DR-learner — paradigm (iii) `tail is the target' alternative",
        "",
        f"Whale DGP, N = {N}, true mean ATE = {TRUE_ATE}, 3 seeds.",
        "QuantileRegressor on DR pseudo-outcomes (intercept-only basis).",
        "",
        "Estimand: q-th quantile of the treatment-effect distribution.",
        "Note: under whale contamination the q-th quantile of the DR pseudo-outcomes",
        "drifts with density; its meaning is *not* the contaminated mean.",
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
    df.to_csv(RESULTS_DIR / "quantile_dr_raw.csv", index=False)


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
