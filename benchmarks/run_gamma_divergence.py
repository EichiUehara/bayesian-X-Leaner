"""γ-divergence robust X-Learner — direct conceptual cousin of Welsch.

The γ-divergence (Fujisawa-Eguchi 2008) defines a robust loss
ρ_γ(r) = -1/γ · log E[exp(-γ r²/2)] which acts as a smooth weight
on residuals. For the canonical Gaussian model it specialises to a
weight w_γ(r) = exp(-γ r²/2) — formally similar to Welsch's
w_W(r) = exp(-r²/c²) with c² = 2/γ.

This script implements an X-learner whose stage-2 regressions of
imputed pseudo-effects are γ-weighted iteratively-reweighted
least squares (IRLS) with the γ weight, and compares its ATE
estimate against RX-Welsch on the whale DGP.

Usage: python -u -m benchmarks.run_gamma_divergence --seeds 3
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
GAMMA = 0.5  # γ-divergence parameter; γ=0 reduces to L²


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def gamma_irls(X, y, gamma=GAMMA, n_iter=20, tol=1e-6):
    """γ-divergence weighted least-squares for the intercept+linear model."""
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X @ beta
        # γ weight: w_γ(r) = exp(-γ r² / 2)
        # Adapt scale by MAD to make γ comparable across DGPs
        scale = np.median(np.abs(r - np.median(r))) / 0.6745
        scale = max(scale, 1e-6)
        w = np.exp(-gamma * (r / scale) ** 2 / 2)
        # Weighted least squares update
        WX = X * w[:, None]
        beta_new = np.linalg.lstsq(WX, w * y, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol * (np.linalg.norm(beta) + 1e-12):
            beta = beta_new
            break
        beta = beta_new
    return beta


def gamma_x_learner_ate(X, Y, W, seed=0):
    """X-learner with γ-divergence-weighted stage-2 regressions."""
    rng = np.random.default_rng(seed)
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)

    # Imputed treatment effects (X-learner):
    D1 = Y[W == 1] - mu0.predict(X[W == 1])  # treated
    D0 = mu1.predict(X[W == 0]) - Y[W == 0]  # controls

    # Stage 2: regress D on intercept (scalar ATE) using γ-IRLS
    Xones1 = np.ones((D1.shape[0], 1))
    Xones0 = np.ones((D0.shape[0], 1))
    tau1 = gamma_irls(Xones1, D1)[0]
    tau0 = gamma_irls(Xones0, D0)[0]

    # X-learner weighting: e(x)*tau1 + (1-e(x))*tau0
    return float(np.mean(pi * tau1 + (1 - pi) * tau0))


def _run(seeds):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_whales = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            t0 = time.time()
            try:
                ate = gamma_x_learner_ate(X, Y, W, seed=seed)
                err = None
            except Exception as e:
                ate = float("nan"); err = str(e)
            rt = time.time() - t0
            bias = ate - TRUE_ATE if not np.isnan(ate) else float("nan")
            rows.append({
                "seed": seed, "density": density, "n_whales": n_whales,
                "ate_hat": ate, "bias": bias, "runtime": rt, "err": err,
            })
            print(f"  seed={seed} p={density:.2f} γ-X ate={ate:+.3f} "
                  f"bias={bias:+.3f} ({rt:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby("density").agg(
        n=("seed", "count"),
        ate_mean=("ate_hat", "mean"),
        ate_std=("ate_hat", "std"),
        bias=("bias", "mean"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "gamma_divergence.md"
    lines = [
        "# γ-divergence robust X-Learner",
        "",
        f"X-learner with γ={GAMMA} divergence-weighted IRLS on imputed effects.",
        f"Whale DGP, N = {N}, true ATE = {TRUE_ATE}, 3 seeds. No posterior; point estimate.",
        "",
        "| density | n | mean ATE | std | bias |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {int(r['n'])} | "
            f"{r['ate_mean']:+.3f} | {r['ate_std']:.3f} | "
            f"{r['bias']:+.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "gamma_divergence_raw.csv", index=False)


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
