"""
Nonlinear CATE benchmark.

τ(x) = 2 + sin(2·x₀)  — smooth with curvature, NOT in the [1, x₀] linear basis.

Tests how RX-Learner's parametric Bayesian CATE behaves when the model's
basis is misspecified vs when it includes enough polynomial/Fourier terms.

Variants evaluated:
  - RX-Learner (linear basis, [1, x₀])          — deliberately misspecified
  - RX-Learner (polynomial basis, [1, x₀, x₀²]) — richer
  - RX-Learner (Fourier basis, [1, x₀, sin(2x₀), cos(2x₀)]) — matches DGP
  - S-Learner, T-Learner, X-Learner (std), EconML Forest — nonparametric baselines

Usage:
    python -m benchmarks.run_nonlinear_cate --seeds 8
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

from benchmarks.dgps import nonlinear_cate_dgp


RESULTS_DIR = Path(__file__).parent / "results"


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


# ── Basis functions for RX-Learner's CATE regressors ───────────────────────

def basis_linear(X):
    return np.column_stack([np.ones(len(X)), X[:, 0]])


def basis_polynomial(X):
    x0 = X[:, 0]
    return np.column_stack([np.ones(len(X)), x0, x0 ** 2])


def basis_fourier(X):
    x0 = X[:, 0]
    return np.column_stack([
        np.ones(len(X)), x0, np.sin(2 * x0), np.cos(2 * x0)
    ])


# ── Estimators ──────────────────────────────────────────────────────────────

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


def cate_rx_learner(X, Y, W, basis_fn, robust=True):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=robust, c_whale=1.34,
        use_student_t=robust, use_overlap=False,
        random_state=42,
    )
    X_infer = basis_fn(X)
    model.fit(X, Y, W, X_infer=X_infer)
    cate, _, _ = model.predict(X_new_infer=X_infer)
    return np.asarray(cate).flatten()


ESTIMATORS = {
    "S-Learner":                       cate_s_learner,
    "T-Learner":                       cate_t_learner,
    "X-Learner (std)":                 cate_x_learner,
    "EconML Forest":                   cate_econml_forest,
    "RX-Learner (linear basis)":       lambda X, Y, W: cate_rx_learner(X, Y, W, basis_linear),
    "RX-Learner (polynomial basis)":   lambda X, Y, W: cate_rx_learner(X, Y, W, basis_polynomial),
    "RX-Learner (Fourier basis)":      lambda X, Y, W: cate_rx_learner(X, Y, W, basis_fourier),
}


def _pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def _bias(tau_hat, tau):
    return float(np.mean(tau_hat - tau))


def _run(seeds):
    rows = []
    for seed in seeds:
        X, Y, W, tau = nonlinear_cate_dgp(seed=seed)
        for name, fn in ESTIMATORS.items():
            t0 = time.time()
            try:
                tau_hat = fn(X, Y, W)
                pehe = _pehe(tau_hat, tau)
                bias = _bias(tau_hat, tau)
                corr = float(np.corrcoef(tau_hat, tau)[0, 1])
                err = None
            except Exception as e:
                pehe = bias = corr = float("nan"); err = str(e)
            rt = time.time() - t0
            rows.append({
                "estimator": name, "seed": seed,
                "pehe": pehe, "bias": bias, "corr_tau_hat_tau": corr,
                "runtime": rt, "error": err,
            })
            print(f"  seed={seed}  {name:<34}  PEHE={pehe:.3f}  "
                  f"corr={corr:+.3f}  bias={bias:+.3f}  ({rt:.1f}s)"
                  + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["pehe"]).groupby("estimator")
              .agg(mean_pehe=("pehe", "mean"),
                   std_pehe=("pehe", "std"),
                   mean_bias=("bias", "mean"),
                   mean_corr=("corr_tau_hat_tau", "mean"),
                   mean_rt=("runtime", "mean"),
                   n=("seed", "count"))
              .sort_values("mean_pehe"))


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "nonlinear_cate.md"
    lines = [
        "# Nonlinear CATE benchmark",
        "",
        f"DGP: `nonlinear_cate_dgp`, τ(x) = 2 + sin(2·x₀), N=1000.",
        f"Seeds: {list(seeds)}",
        "",
        "The DGP's CATE has curvature *not* in the linear basis [1, x₀]. "
        "RX-Learner variants differ only in the `X_infer` basis passed to the "
        "Bayesian CATE regression — same nuisance, same MCMC, same likelihood.",
        "",
        "| Estimator | n | Mean PEHE | Std PEHE | Mean Bias | Mean Corr | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in agg.iterrows():
        lines.append(
            f"| {name} | {int(r['n'])} | "
            f"{r['mean_pehe']:.3f} | {r['std_pehe']:.3f} | "
            f"{r['mean_bias']:+.3f} | {r['mean_corr']:+.3f} | "
            f"{r['mean_rt']:.2f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "nonlinear_cate_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'nonlinear_cate_raw.csv'}")


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
