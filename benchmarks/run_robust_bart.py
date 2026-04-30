"""Robust BART variant — contaminated-normal residuals.

Reviewer concern (round 5): heavy-tailed residuals at the BART
likelihood level. Student-t T-BART exists (run_student_t_bart.py);
this script tests a mixture-of-Gaussians (contaminated-normal)
residual likelihood instead, which can model contamination directly:

    y_i ~ (1 - ε) N(μ(x_i), σ_in²) + ε N(μ(x_i), σ_out²)

with ε ∈ [0, 0.3] as a free parameter (Beta prior).

Note: pymc_bart's tree-splitting rule is fixed (Gaussian-MSE-based)
and not exposed for modification. M-estimation-in-splitting (the
strongest sense of "robust BART") would require a custom BART
implementation, which is out of scope.

Usage: python -u -m benchmarks.run_robust_bart --replications 5
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import load_ihdp


RESULTS_DIR = Path(__file__).parent / "results"


def cate_contamnormal_bart(X, Y, W):
    """T-BART with contaminated-normal residual likelihood.

    Model per arm:  y_i = mu(x_i) + eps_i,
                    eps_i ~ (1-eps) N(0, sigma_in²) + eps N(0, sigma_out²)
    """
    import pymc as pm
    import pymc_bart as pmb
    mu_preds = {}
    for w, mask in [(0, W == 0), (1, W == 1)]:
        X_arm = X[mask]; Y_arm = Y[mask]
        with pm.Model():
            X_data = pm.Data("X_data", X_arm)
            mu = pmb.BART("mu", X=X_data, Y=Y_arm, m=200)
            sigma_in = pm.HalfNormal("sigma_in", 1.0)
            sigma_out = pm.HalfNormal("sigma_out", 5.0)
            eps_w = pm.Beta("eps_w", 1.0, 9.0)  # Mean ~10% contamination prior
            log_in = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma_in), Y_arm) + pm.math.log1p(-eps_w)
            log_out = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma_out), Y_arm) + pm.math.log(eps_w)
            log_lik = pm.math.logaddexp(log_in, log_out)
            pm.Potential("contam_ll", log_lik.sum())
            idata = pm.sample(
                draws=500, tune=500, chains=2, cores=1,
                random_seed=42, progressbar=False,
                compute_convergence_checks=False,
            )
            pm.set_data({"X_data": X})
            pp = pm.sample_posterior_predictive(
                idata, var_names=["mu"], progressbar=False, random_seed=42)
        mu_preds[w] = pp.posterior_predictive["mu"].mean(
            dim=["chain", "draw"]).values
    return mu_preds[1] - mu_preds[0]


def _pehe(tau_hat, tau): return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))
def _ate_err(tau_hat, tau): return float(abs(np.mean(tau_hat) - np.mean(tau)))


def _run(reps):
    rows = []
    for rep in reps:
        X, Y, W, tau = load_ihdp(rep)
        print(f"\n── IHDP replication {rep} (N={len(X)}) ──")
        t0 = time.time()
        try:
            tau_hat = cate_contamnormal_bart(X, Y, W)
            pehe = _pehe(tau_hat, tau); ate = _ate_err(tau_hat, tau); err = None
        except Exception as e:
            pehe = ate = float("nan"); err = str(e)
        rt = time.time() - t0
        rows.append({"estimator": "Contam-Normal T-BART", "replication": rep,
                     "pehe": pehe, "ate_err": ate, "runtime": rt, "error": err})
        print(f"  Contam-Normal T-BART √PEHE={pehe:.3f} ε_ATE={ate:.3f} ({rt:.1f}s)"
              + (f" ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replications", type=int, default=5)
    args = ap.parse_args()
    reps = list(range(1, args.replications + 1))
    df = _run(reps)
    print("\n── Summary ──")
    if df["pehe"].notna().any():
        print(f"Mean √PEHE = {df['pehe'].mean():.3f} ± {df['pehe'].std():.3f}")
    df.to_csv(RESULTS_DIR / "robust_bart_raw.csv", index=False)
    md = ["# Robust BART (contaminated-normal residuals) on IHDP", "",
          "Heavy-tailed Bayesian tree backbone with two-component Gaussian mixture",
          "residuals (eps_w ~ Beta(1,9), sigma_out free).",
          "Note: BART splitting rule unchanged; only the leaf-likelihood is robustified.",
          "", "| rep | √PEHE | ε_ATE | runtime |", "|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(f"| {int(r['replication'])} | {r['pehe']:.3f} | "
                  f"{r['ate_err']:.3f} | {r['runtime']:.1f}s |")
    if df["pehe"].notna().any():
        md += ["",
               f"**Mean √PEHE = {df['pehe'].mean():.3f} ± {df['pehe'].std():.3f}**, "
               f"mean ε_ATE = {df['ate_err'].mean():.3f}."]
    (RESULTS_DIR / "robust_bart.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
