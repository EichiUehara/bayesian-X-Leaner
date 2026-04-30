"""Student-t-likelihood BART/BCF — heavy-tailed Bayesian-tree baseline.

Reviewer concern: bounded-influence Welsch vs unbounded heavy-tailed
likelihood within a Bayesian tree backbone. We replicate T-BART
(Hill 2011) but with Student-t residuals (ν learned via prior
Gamma(2, 0.1)) instead of Gaussian. If Student-t-BART matches our
RX-Welsch performance under contamination, the bounded-influence
operator is unnecessary; if it doesn't, we have empirical evidence
the redescender provides distinctive value.

Usage: python -u -m benchmarks.run_student_t_bart --replications 5
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


def cate_studentt_bart(X, Y, W):
    """T-BART with Student-t residuals (ν free)."""
    import pymc as pm
    import pymc_bart as pmb
    mu_preds = {}
    for w, mask in [(0, W == 0), (1, W == 1)]:
        X_arm = X[mask]
        Y_arm = Y[mask]
        with pm.Model():
            X_data = pm.Data("X_data", X_arm)
            mu = pmb.BART("mu", X=X_data, Y=Y_arm, m=200)
            sigma = pm.HalfNormal("sigma", 1.0)
            nu = pm.Gamma("nu", 2.0, 0.1)
            pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=Y_arm)
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


def _pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def _ate_err(tau_hat, tau):
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def _run(reps):
    rows = []
    for rep in reps:
        X, Y, W, tau = load_ihdp(rep)
        print(f"\n── IHDP replication {rep} (N={len(X)}) ──")
        t0 = time.time()
        try:
            tau_hat = cate_studentt_bart(X, Y, W)
            pehe = _pehe(tau_hat, tau); ate_err = _ate_err(tau_hat, tau); err = None
        except Exception as e:
            pehe = ate_err = float("nan"); err = str(e)
        rt = time.time() - t0
        rows.append({
            "estimator": "Student-t T-BART", "replication": rep,
            "pehe": pehe, "ate_err": ate_err, "runtime": rt, "error": err,
        })
        print(f"  Student-t T-BART √PEHE={pehe:.3f} ε_ATE={ate_err:.3f} ({rt:.1f}s)"
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
        print(f"Mean √PEHE = {df['pehe'].mean():.3f} ± {df['pehe'].std():.3f}, "
              f"mean ε_ATE = {df['ate_err'].mean():.3f}")
    df.to_csv(RESULTS_DIR / "student_t_bart_raw.csv", index=False)
    md = ["# Student-t-likelihood T-BART on IHDP",
          "",
          "Heavy-tailed Bayesian tree backbone: T-BART with Student-t residuals",
          "(ν learned via prior Gamma(2, 0.1)). 5 reps, m=200, 1000 total draws.",
          "",
          "| rep | √PEHE | ε_ATE | runtime |",
          "|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(f"| {int(r['replication'])} | {r['pehe']:.3f} | "
                  f"{r['ate_err']:.3f} | {r['runtime']:.1f}s |")
    if df["pehe"].notna().any():
        md.extend(["",
                   f"**Mean √PEHE = {df['pehe'].mean():.3f} ± {df['pehe'].std():.3f}**, "
                   f"mean ε_ATE = {df['ate_err'].mean():.3f}."])
    (RESULTS_DIR / "student_t_bart.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
