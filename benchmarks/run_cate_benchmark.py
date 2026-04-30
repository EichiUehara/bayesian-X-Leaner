"""
CATE (heterogeneous-effect) benchmark.

Measures each estimator's per-unit τ̂(x) vs ground truth τ(x) = 2 + x₀:

  PEHE = √E[(τ̂(x) − τ(x))²]   (precision in estimation of heterog. effects)

Only estimators that can produce per-unit CATE are evaluated. ATE-only
methods (R-Learner constant, DoubleML IRM ATE mode, CausalML average) are
skipped.

Usage:
    python -m benchmarks.run_cate_benchmark --seeds 8
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from benchmarks.dgps import heterogeneous_cate_dgp

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


# ── CATE-capable estimators ─────────────────────────────────────────────────

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


def cate_rx_learner(X, Y, W, robust=False, use_student_t=False, use_overlap=False):
    """RX-Learner with X_infer=X to estimate per-unit CATE."""
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=robust, c_whale=1.34,
        use_student_t=use_student_t, use_overlap=use_overlap,
        random_state=42,
    )
    # Intercept + x₀ as CATE regressors — τ(x) = 2 + x₀
    X_infer = np.column_stack([np.ones(len(X)), X[:, 0]])
    model.fit(X, Y, W, X_infer=X_infer)
    cate, _, _ = model.predict(X_new_infer=X_infer)
    return np.asarray(cate).flatten()


ESTIMATORS = {
    "S-Learner":            cate_s_learner,
    "T-Learner":            cate_t_learner,
    "X-Learner (std)":      cate_x_learner,
    "EconML Forest":        cate_econml_forest,
    "RX-Learner (robust)":  lambda X, Y, W: cate_rx_learner(
        X, Y, W, robust=True, use_student_t=True),
    "RX-Learner (std)":     lambda X, Y, W: cate_rx_learner(X, Y, W),
}


def _pehe(tau_hat, tau_true):
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def _bias(tau_hat, tau_true):
    return float(np.mean(tau_hat - tau_true))


def _run(seeds):
    rows = []
    for seed in seeds:
        X, Y, W, tau = heterogeneous_cate_dgp(seed=seed)
        for name, fn in ESTIMATORS.items():
            t0 = time.time()
            try:
                tau_hat = fn(X, Y, W)
                err = None
                pehe = _pehe(tau_hat, tau)
                bias = _bias(tau_hat, tau)
                corr = float(np.corrcoef(tau_hat, tau)[0, 1])
            except Exception as e:
                pehe = bias = corr = float("nan")
                err = str(e)
            rt = time.time() - t0
            rows.append({
                "estimator": name, "seed": seed,
                "pehe": pehe, "bias": bias, "corr_tau_hat_tau": corr,
                "runtime": rt, "error": err,
            })
            print(f"  seed={seed}  {name:<26}  PEHE={pehe:.3f}  "
                  f"corr={corr:+.3f}  bias={bias:+.3f}  ({rt:.1f}s)"
                  + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    agg = (df.groupby("estimator")
             .agg(mean_pehe=("pehe", "mean"),
                  std_pehe=("pehe", "std"),
                  mean_bias=("bias", "mean"),
                  mean_corr=("corr_tau_hat_tau", "mean"),
                  mean_rt=("runtime", "mean"),
                  n=("seed", "count"))
             .sort_values("mean_pehe"))
    return agg


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "cate_benchmark.md"
    lines = [
        "# CATE (heterogeneous-effect) benchmark",
        "",
        f"DGP: `heterogeneous_cate_dgp`, τ(x) = 2 + x₀, N=1000.",
        f"Seeds: {list(seeds)}",
        "",
        "**PEHE** (Precision in Estimating Heterogeneous Effects) = √E[(τ̂ − τ)²] — lower is better.",
        "**corr** = Pearson correlation between τ̂(x) and τ(x) — 1.0 is best.",
        "",
        "| Estimator | n | Mean PEHE | Std PEHE | Mean Bias | Mean Corr | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in agg.iterrows():
        lines.append(
            f"| {name} | {int(row['n'])} | "
            f"{row['mean_pehe']:.3f} | {row['std_pehe']:.3f} | "
            f"{row['mean_bias']:+.3f} | {row['mean_corr']:+.3f} | "
            f"{row['mean_rt']:.2f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    csv = RESULTS_DIR / "cate_benchmark_raw.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")


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
