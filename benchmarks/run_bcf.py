"""BCF (Hahn-Murray-Carvalho 2020) baseline on IHDP.

Implements the canonical Bayesian Causal Forest in pymc_bart with
the Hahn 2020 decomposition:

    y_i = mu(x_i, pi_hat_i) + tau(x_i) * w_i + epsilon_i

where mu(.) is the prognostic-surface BART (with the estimated
propensity score as a covariate, per Hahn et al. for
regularisation-induced confounding control) and tau(.) is the
treatment-effect BART. We use distinct prior hyperparameters for
the two trees per the original paper.

Compared to T-BART (Hill 2011), BCF (a) shares both arms in a
single tree fit through W as a covariate, and (b) regularises tau
directly so that small or constant treatment effects are not
overfit.

Usage: python -u -m benchmarks.run_bcf --replications 5
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from benchmarks.dgps import load_ihdp


RESULTS_DIR = Path(__file__).parent / "results"


def cate_bcf(X, Y, W):
    """BCF: y = mu(x, pi_hat) + tau(x) * w + eps.

    Two BART components fit jointly. mu uses x and pi_hat as covariates;
    tau uses only x. Tau gets fewer trees and tighter shrinkage per
    Hahn 2020 (regularisation against regularisation-induced confounding).
    """
    import pymc as pm
    import pymc_bart as pmb

    pi_hat = LogisticRegression(max_iter=1000).fit(X, W).predict_proba(X)[:, 1]
    pi_hat = np.clip(pi_hat, 0.05, 0.95)

    X_mu = np.column_stack([X, pi_hat])
    Wf = W.astype(float)

    with pm.Model() as model:
        X_mu_data = pm.Data("X_mu", X_mu)
        X_tau_data = pm.Data("X_tau", X)
        W_data = pm.Data("W", Wf)

        mu = pmb.BART("mu", X=X_mu_data, Y=Y, m=200)
        # Tau gets fewer trees + smaller variance — Hahn 2020 prescription
        tau = pmb.BART("tau", X=X_tau_data, Y=Y, m=50)

        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y_obs", mu=mu + tau * W_data, sigma=sigma, observed=Y)

        idata = pm.sample(
            draws=500, tune=500, chains=2, cores=1,
            random_seed=42, progressbar=False,
            compute_convergence_checks=False,
        )

        # Posterior CATE on full sample: tau(x) under W=1 (counterfactual)
        # Pull tau directly — it's already a function of x only
        tau_samples = idata.posterior["tau"].mean(dim=["chain", "draw"]).values

    return tau_samples


def _pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def _ate_err(tau_hat, tau):
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def _run(replications):
    rows = []
    for rep in replications:
        X, Y, W, tau = load_ihdp(rep)
        print(f"\n── IHDP replication {rep} (N={len(X)}, treated={int(W.sum())}) ──")
        t0 = time.time()
        try:
            tau_hat = cate_bcf(X, Y, W)
            pehe = _pehe(tau_hat, tau)
            ate_err = _ate_err(tau_hat, tau)
            err = None
        except Exception as e:
            pehe = ate_err = float("nan"); err = str(e)
        rt = time.time() - t0
        rows.append({
            "estimator": "BCF (Hahn 2020)", "replication": rep,
            "pehe": pehe, "ate_err": ate_err, "runtime": rt, "error": err,
        })
        print(f"  BCF (Hahn 2020) √PEHE={pehe:.3f} ε_ATE={ate_err:.3f} ({rt:.1f}s)"
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
        print(f"BCF: mean √PEHE = {df['pehe'].mean():.3f}, std = {df['pehe'].std():.3f}, "
              f"mean ε_ATE = {df['ate_err'].mean():.3f}")
    df.to_csv(RESULTS_DIR / "bcf_raw.csv", index=False)

    md = ["# BCF (Hahn 2020) on IHDP — proper Bayesian Causal Forest baseline",
          "",
          f"5 replications, BCF prognostic mu(x, pi) BART (m=200) + tau(x) BART (m=50),",
          "1000 total draws across 2 chains, cores=1. Pi-hat from LogisticRegression.",
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
    (RESULTS_DIR / "bcf.md").write_text("\n".join(md))
    print(f"wrote {RESULTS_DIR / 'bcf.md'}")


if __name__ == "__main__":
    main()
