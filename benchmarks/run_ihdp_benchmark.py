"""
IHDP semi-synthetic benchmark.

Real covariates (Infant Health & Development Program, N=747, 25 features)
with simulated outcomes from Hill (2011). Ground-truth per-unit τ is
known, enabling PEHE evaluation on realistic data.

Metrics:
  - ε_ATE  = |mean(τ̂) − mean(τ)|      (ATE error)
  - √ε_PEHE = √E[(τ̂(x) − τ(x))²]     (heterogeneous-effect error)

Usage:
    python -m benchmarks.run_ihdp_benchmark --replications 10
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from benchmarks.dgps import load_ihdp


RESULTS_DIR = Path(__file__).parent / "results"


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def cate_t_learner(X, Y, W):
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    return mu1.predict(X) - mu0.predict(X)


def cate_s_learner(X, Y, W):
    m = _make_reg()
    m.fit(np.column_stack([X, W.astype(float)]), Y)
    X1 = np.column_stack([X, np.ones(len(X))])
    X0 = np.column_stack([X, np.zeros(len(X))])
    return m.predict(X1) - m.predict(X0)


def cate_x_learner(X, Y, W):
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.01, 0.99)
    D1 = Y[W == 1] - mu0.predict(X[W == 1])
    D0 = mu1.predict(X[W == 0]) - Y[W == 0]
    tau1 = _make_reg(); tau1.fit(X[W == 1], D1)
    tau0 = _make_reg(); tau0.fit(X[W == 0], D0)
    return (1 - pi) * tau1.predict(X) + pi * tau0.predict(X)


def cate_econml_forest(X, Y, W):
    from econml.dml import CausalForestDML
    est = CausalForestDML(
        model_y=_make_reg(), model_t=_make_clf(),
        discrete_treatment=True, cv=2, n_estimators=200, random_state=42,
    )
    est.fit(Y, W, X=X)
    return est.effect(X).flatten()


def cate_rx_learner(X, Y, W, robust=True, use_overlap=False,
                     nuisance="xgboost", huber_delta=0.5):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    if nuisance == "catboost":
        outcome_params = {"depth": 4, "iterations": 150,
                          "loss_function": f"Huber:delta={huber_delta}"}
        prop_params = {"depth": 4, "iterations": 150}
    else:
        outcome_params = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        prop_params = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
    model = TargetedBayesianXLearner(
        outcome_model_params=outcome_params,
        propensity_model_params=prop_params,
        nuisance_method=nuisance,
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=robust, c_whale=1.34,
        use_student_t=robust, use_overlap=use_overlap,
        random_state=42,
    )
    # On IHDP we don't know a priori which covariate drives τ, so use all
    # continuous + intercept. The CATE basis is then linear in all features
    # — a plain Bayesian linear regression for τ(x).
    X_infer = np.column_stack([np.ones(len(X)), X])
    model.fit(X, Y, W, X_infer=X_infer)
    cate, _, _ = model.predict(X_new_infer=X_infer)
    return np.asarray(cate).flatten()


def cate_causal_bart(X, Y, W):
    """T-learner with BART regressors — the canonical Causal BART
    of Hill (2011). Uses pymc_bart for two per-arm BART fits.
    Full config: m=200 trees, 1000 tune + 1000 draws, 2 chains.
    cores=1 avoids the JAX-vs-multiprocessing fork deadlock."""
    import pymc as pm
    import pymc_bart as pmb
    mu_preds = {}
    for w, mask in [(0, W == 0), (1, W == 1)]:
        X_arm = X[mask]
        Y_arm = Y[mask]
        with pm.Model() as model:
            X_data = pm.Data("X_data", X_arm)
            mu = pmb.BART("mu", X=X_data, Y=Y_arm, m=200)
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal("y", mu=mu, sigma=sigma, observed=Y_arm)
            idata = pm.sample(draws=1000, tune=1000, chains=2, cores=1,
                              random_seed=42, progressbar=False,
                              compute_convergence_checks=False)
            pm.set_data({"X_data": X})
            pp = pm.sample_posterior_predictive(
                idata, var_names=["mu"], progressbar=False, random_seed=42)
        mu_preds[w] = pp.posterior_predictive["mu"].mean(
            dim=["chain", "draw"]).values
    return mu_preds[1] - mu_preds[0]


def cate_huber_dr(X, Y, W):
    """Huber-regression point estimator on DR pseudo-outcomes — the
    cheap trivial-baseline for the MCMC comparison."""
    from sklearn.linear_model import HuberRegressor
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
    mu0_all = mu0.predict(X)
    mu1_all = mu1.predict(X)
    D = np.where(
        W == 1,
        mu1_all - mu0_all + (Y - mu1_all) / pi,
        mu1_all - mu0_all - (Y - mu0_all) / (1.0 - pi),
    )
    X_infer = np.column_stack([np.ones(len(X)), X])
    reg = HuberRegressor(epsilon=1.35, max_iter=200,
                         fit_intercept=False)
    reg.fit(X_infer, D)
    return reg.predict(X_infer)


ESTIMATORS = {
    "S-Learner":             cate_s_learner,
    "T-Learner":             cate_t_learner,
    "X-Learner (std)":       cate_x_learner,
    "EconML Forest":         cate_econml_forest,
    "Causal BART (Hill 2011)": cate_causal_bart,
    "Huber-DR (point)":      cate_huber_dr,
    "RX-Learner (robust)":   lambda X, Y, W: cate_rx_learner(X, Y, W, robust=True),
    "RX-Learner (std)":      lambda X, Y, W: cate_rx_learner(X, Y, W, robust=False),
    "RX-Learner (robust+overlap)": lambda X, Y, W: cate_rx_learner(
        X, Y, W, robust=True, use_overlap=True),
    "RX-Learner (CB-Huber δ=0.5)": lambda X, Y, W: cate_rx_learner(
        X, Y, W, robust=True, nuisance="catboost", huber_delta=0.5),
    "RX-Learner (CB-Huber δ=1.345)": lambda X, Y, W: cate_rx_learner(
        X, Y, W, robust=True, nuisance="catboost", huber_delta=1.345),
}


def _pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def _ate_err(tau_hat, tau):
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def _run(replications):
    rows = []
    for rep in replications:
        X, Y, W, tau = load_ihdp(rep)
        print(f"\n── IHDP replication {rep} (N={len(X)}, treated={int(W.sum())}) ──")
        for name, fn in ESTIMATORS.items():
            t0 = time.time()
            try:
                tau_hat = fn(X, Y, W)
                pehe = _pehe(tau_hat, tau)
                ate_err = _ate_err(tau_hat, tau)
                err = None
            except Exception as e:
                pehe = ate_err = float("nan")
                err = str(e)
            rt = time.time() - t0
            rows.append({
                "estimator": name, "replication": rep,
                "pehe": pehe, "ate_err": ate_err,
                "runtime": rt, "error": err,
            })
            print(f"  {name:<32} √PEHE={pehe:.3f}  ε_ATE={ate_err:.3f}  "
                  f"({rt:.1f}s)" + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["pehe"]).groupby("estimator")
              .agg(mean_pehe=("pehe", "mean"),
                   std_pehe=("pehe", "std"),
                   mean_ate_err=("ate_err", "mean"),
                   mean_rt=("runtime", "mean"),
                   n=("replication", "count"))
              .sort_values("mean_pehe"))


def _write_markdown(df, agg, reps):
    path = RESULTS_DIR / "ihdp_benchmark.md"
    lines = [
        "# IHDP semi-synthetic benchmark",
        "",
        f"Dataset: Hill (2011) IHDP, replications {list(reps)} (from CEVAE preprocessing).",
        "Covariates: 25 real features (N=747 per rep). Outcome simulated per Hill's response surface B.",
        "",
        "Metrics:",
        "- **√ε_PEHE** = √mean((τ̂ − τ)²)   — heterogeneous-effect recovery (lower is better)",
        "- **ε_ATE** = |mean(τ̂) − mean(τ)| — average-effect error",
        "",
        "| Estimator | n | √PEHE | std(√PEHE) | ε_ATE | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in agg.iterrows():
        lines.append(
            f"| {name} | {int(r['n'])} | "
            f"{r['mean_pehe']:.3f} | {r['std_pehe']:.3f} | "
            f"{r['mean_ate_err']:.3f} | {r['mean_rt']:.2f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "ihdp_benchmark_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'ihdp_benchmark_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replications", type=int, default=10)
    args = ap.parse_args()
    reps = list(range(1, args.replications + 1))
    df = _run(reps)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, reps)


if __name__ == "__main__":
    main()
