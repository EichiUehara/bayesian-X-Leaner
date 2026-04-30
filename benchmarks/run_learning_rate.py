"""Bootstrap-calibrated learning-rate selector for the generalised
Welsch posterior.

Reviewer concern: §4.4 acknowledges the Welsch pseudo-likelihood
yields a generalised Bayes posterior whose calibration is empirical,
not theorem-derived. A practical recipe would be a learning-rate
(temperature) selector η > 0 that down-weights the pseudo-likelihood
to match the variance of an independent (nuisance) bootstrap.

Implements the Lyddon-Holmes-Walker (2019) loss-likelihood-bootstrap
calibrator, adapted to the Welsch pseudo-loss:

  η_hat = argmin_η  | Var_posterior(β; η) - Var_LLB(β) |

where Var_LLB is estimated by repeated cross-fitting + Huber-DR
point estimation. The script then refits the RX-Learner with
log_factor scaled by η_hat and reports posterior calibration.

Usage:  python -u -m benchmarks.run_learning_rate --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0
DENSITIES = [0.00, 0.05, 0.20]
N_LLB = 50
ETA_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def _huber_dr_ate(X, Y, W, seed):
    rng = np.random.default_rng(seed)
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
    mu0_a = mu0.predict(X); mu1_a = mu1.predict(X)
    D = np.where(W == 1,
                 mu1_a - mu0_a + (Y - mu1_a) / pi,
                 mu1_a - mu0_a - (Y - mu0_a) / (1.0 - pi))
    Xones = np.ones((len(X), 1))
    reg = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
    reg.fit(Xones, D)
    return float(reg.coef_[0])


def llb_variance(X, Y, W, B=N_LLB, seed=0):
    """Loss-likelihood bootstrap: refit nuisance + Huber-DR on
    bootstrap resamples; variance of the resulting ATE point estimates
    is the calibration target."""
    rng = np.random.default_rng(seed)
    Nfull = len(X)
    boots = []
    for b in range(B):
        idx = rng.integers(0, Nfull, size=Nfull)
        try:
            ate_b = _huber_dr_ate(X[idx], Y[idx], W[idx], seed + b + 1)
            boots.append(ate_b)
        except Exception:
            pass
    if len(boots) < 5:
        return float("nan")
    return float(np.var(boots, ddof=1))


def fit_rx_with_eta(X, Y, W, eta, seed):
    """Run RX-Learner with the Welsch c_whale rescaled to absorb a
    pseudo-likelihood temperature: scaling -ρ_W(r; c) by η is
    equivalent to using c' = c / sqrt(η) in the Welsch shape.
    Returns posterior var(beta_intercept)."""
    c_eta = 1.34 / np.sqrt(eta)
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        nuisance_method="xgboost",
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=c_eta, use_student_t=True,
        mad_rescale=False, random_state=seed,
    )
    model.fit(X, Y, W, X_infer=np.ones((N, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    return float(np.var(beta, ddof=1)), float(np.mean(beta)), \
        float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def select_eta(X, Y, W, seed):
    """Select η minimising |Var_posterior(η) - Var_LLB|."""
    target = llb_variance(X, Y, W, seed=seed)
    if np.isnan(target):
        return 1.0, target, []
    candidates = []
    for eta in ETA_GRID:
        try:
            var_post, _, _, _ = fit_rx_with_eta(X, Y, W, eta, seed)
            candidates.append((eta, var_post, abs(var_post - target)))
        except Exception as e:
            print(f"    eta={eta} failed: {e}")
    if not candidates:
        return 1.0, target, []
    eta_best = min(candidates, key=lambda t: t[2])[0]
    return eta_best, target, candidates


def _run(seeds):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_whales = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            t0 = time.time()
            eta_hat, var_target, candidates = select_eta(X, Y, W, seed)
            # Final fit at calibrated η
            var_post, ate, lo, hi = fit_rx_with_eta(X, Y, W, eta_hat, seed)
            rt = time.time() - t0
            cov = int(lo <= TRUE_ATE <= hi)
            rows.append({
                "seed": seed, "density": density,
                "eta_hat": eta_hat, "var_llb": var_target, "var_post": var_post,
                "ate_hat": ate, "ci_lo": lo, "ci_hi": hi, "cov": cov,
                "ci_width": hi - lo, "runtime": rt,
            })
            print(f"  seed={seed} p={density:.2f} η̂={eta_hat:.2f} "
                  f"var_llb={var_target:.4g} var_post={var_post:.4g} "
                  f"ate={ate:+.3f} CI=[{lo:+.3f},{hi:+.3f}] cov={cov} ({rt:.0f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby("density").agg(
        n=("seed", "count"),
        eta_mean=("eta_hat", "mean"),
        ate_mean=("ate_hat", "mean"),
        coverage=("cov", "mean"),
        ci_width=("ci_width", "mean"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "learning_rate_calibration.md"
    lines = [
        "# Generalised-Bayes learning-rate calibration for the Welsch posterior",
        "",
        f"Bootstrap-calibrated η-selector (Lyddon-Holmes-Walker style).",
        f"Whale DGP, N = {N}, true ATE = {TRUE_ATE}.",
        f"η ∈ {ETA_GRID}; each candidate refits the posterior with c → c/√η.",
        f"LLB target variance from {N_LLB} bootstrap replicates of Huber-DR.",
        "",
        "| density | n | mean η̂ | ATE | 95% coverage | CI width |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {int(r['n'])} | "
            f"{r['eta_mean']:.2f} | {r['ate_mean']:+.3f} | "
            f"{r['coverage']:.2f} | {r['ci_width']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "learning_rate_calibration_raw.csv", index=False)


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
