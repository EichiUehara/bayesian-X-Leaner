"""Coverage comparison: RX-Learner MCMC posterior vs Huber-DR sandwich/bootstrap CI
on the whale DGP.

Reviewer concern: the IHDP table shows Huber-DR is within noise of
RX-Learner on point PEHE. The paper's defence is that MCMC buys a
calibrated posterior; that defence lands only if we can show the
Huber-DR sandwich/bootstrap CI fails calibration where RX-Learner's
credible interval holds. This script does that head-to-head.

DGP: whale_dgp from benchmarks/dgps.py, contamination densities
{0 %, 1 %, 5 %, 20 %}, N=1000, 3 seeds. True ATE = 2.0 (known).

Estimators (all targeting the scalar ATE):
  - RX-Learner (robust=True, default XGB-MSE nuisance): MCMC 95 % CI.
  - RX-Learner (severity="severe"):                     MCMC 95 % CI.
  - Huber-DR (sklearn HuberRegressor + bootstrap 200):  95 % percentile CI.

Metrics: ATE point estimate, 95 % CI coverage of true ATE=2.0, CI width.

Usage:  python -u -m benchmarks.run_coverage_comparison --seeds 3
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
DENSITIES = [0.00, 0.01, 0.05, 0.20]
N_BOOTSTRAP = 200


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def _dr_pseudo_outcomes(X, Y, W):
    """Build DR pseudo-outcomes the same way the RX-Learner Phase 2 does."""
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
    return D


def huber_dr_ate_with_bootstrap(X, Y, W, n_boot=N_BOOTSTRAP, rng_seed=0):
    """Fit Huber-DR once for the point estimate, then bootstrap the
    entire (nuisance + Huber) pipeline for a 95 % percentile CI."""
    rng = np.random.default_rng(rng_seed)
    # Point estimate
    # NOTE: sklearn HuberRegressor needs >=1 feature column; we use a
    # ones column with fit_intercept=False so coef_[0] is the ATE
    # (equivalent to a Huber location estimate of the DR pseudo-outcomes).
    D = _dr_pseudo_outcomes(X, Y, W)
    Xones = np.ones((len(X), 1))
    reg = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
    reg.fit(Xones, D)
    ate_hat = float(reg.coef_[0])

    # Bootstrap
    boots = np.empty(n_boot)
    N_ = len(X)
    for b in range(n_boot):
        idx = rng.integers(0, N_, size=N_)
        Xb, Yb, Wb = X[idx], Y[idx], W[idx]
        try:
            Db = _dr_pseudo_outcomes(Xb, Yb, Wb)
            rb = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
            rb.fit(np.ones((N_, 1)), Db)
            boots[b] = rb.coef_[0]
        except Exception:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return ate_hat, float(lo), float(hi)


def rx_learner_ate(X, Y, W, severity, seed, likelihood="welsch"):
    """Fit an RX-Learner with an intercept-only basis; return (mean, 2.5%, 97.5%) of posterior.

    likelihood ∈ {welsch, gaussian, student_t}: which Phase-3 likelihood to use.
    """
    kwargs = dict(
        n_splits=2,
        num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, mad_rescale=False, random_state=seed,
    )
    if likelihood == "welsch":
        kwargs["robust"] = True
        kwargs["use_student_t"] = True
    elif likelihood == "gaussian":
        kwargs["robust"] = False
        kwargs["use_student_t"] = False
    elif likelihood == "student_t":
        kwargs["robust"] = False
        kwargs["use_student_t"] = True
    else:
        raise ValueError(likelihood)

    if severity == "none":
        kwargs["nuisance_method"] = "xgboost"
        kwargs["outcome_model_params"] = {
            "max_depth": 4, "n_estimators": 150, "verbosity": 0}
        kwargs["propensity_model_params"] = {
            "max_depth": 4, "n_estimators": 150, "verbosity": 0}
    elif severity == "severe":
        kwargs["contamination_severity"] = "severe"
    else:
        raise ValueError(severity)
    model = TargetedBayesianXLearner(**kwargs)
    model.fit(X, Y, W, X_infer=np.ones((N, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    ate = float(np.mean(beta))
    lo, hi = np.percentile(beta, [2.5, 97.5])
    return ate, float(lo), float(hi)


def conformal_dr_ate(X, Y, W, alpha=0.05, seed=0):
    """Split-conformal interval for the ATE using DR pseudo-outcomes.

    Lei & Candès (2021) inspired: split data 50/50; fit nuisances on
    half; build pseudo-outcomes on the other half; the (1 - alpha)
    quantile of |D - mean(D)| residuals over the calibration half
    yields a finite-sample conformal interval for the ATE under
    exchangeability.
    """
    rng = np.random.default_rng(seed)
    N_ = len(X)
    perm = rng.permutation(N_)
    train_idx, cal_idx = perm[:N_ // 2], perm[N_ // 2:]
    Xt, Yt, Wt = X[train_idx], Y[train_idx], W[train_idx]
    Xc, Yc, Wc = X[cal_idx], Y[cal_idx], W[cal_idx]

    # Fit nuisances on training half
    mu0 = _make_reg(); mu0.fit(Xt[Wt == 0], Yt[Wt == 0])
    mu1 = _make_reg(); mu1.fit(Xt[Wt == 1], Yt[Wt == 1])
    pi_m = _make_clf(); pi_m.fit(Xt, Wt)

    # Build DR pseudo-outcomes on calibration half (out-of-fold)
    pi_c = np.clip(pi_m.predict_proba(Xc)[:, 1], 0.05, 0.95)
    mu0_c = mu0.predict(Xc); mu1_c = mu1.predict(Xc)
    Dc = np.where(
        Wc == 1,
        mu1_c - mu0_c + (Yc - mu1_c) / pi_c,
        mu1_c - mu0_c - (Yc - mu0_c) / (1.0 - pi_c),
    )

    # Conformal interval for the mean: ATE_hat ± q_(1-alpha)(|D - mean(D)|)
    ate_hat = float(np.mean(Dc))
    residuals = np.abs(Dc - ate_hat)
    q = float(np.quantile(residuals, 1 - alpha))
    return ate_hat, ate_hat - q, ate_hat + q


def _run(seeds):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_whales = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            for label, fit in [
                ("RX-Welsch (severity=none)",      lambda: rx_learner_ate(X, Y, W, "none",   seed, "welsch")),
                ("RX-Welsch (severity=severe)",    lambda: rx_learner_ate(X, Y, W, "severe", seed, "welsch")),
                ("RX-Gaussian (severity=none)",    lambda: rx_learner_ate(X, Y, W, "none",   seed, "gaussian")),
                ("RX-StudentT (severity=none)",    lambda: rx_learner_ate(X, Y, W, "none",   seed, "student_t")),
                ("Huber-DR (bootstrap CI)",        lambda: huber_dr_ate_with_bootstrap(
                    X, Y, W, rng_seed=seed)),
                ("Conformal-DR (split-CP)",        lambda: conformal_dr_ate(X, Y, W, alpha=0.05, seed=seed)),
            ]:
                t0 = time.time()
                try:
                    ate, lo, hi = fit()
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); err = str(e)
                rt = time.time() - t0
                cov = int(lo <= TRUE_ATE <= hi) if not np.isnan(ate) else 0
                width = hi - lo if not np.isnan(ate) else float("nan")
                rows.append({
                    "seed": seed, "density": density, "n_whales": n_whales,
                    "estimator": label,
                    "ate_hat": ate, "ci_lo": lo, "ci_hi": hi,
                    "cov": cov, "ci_width": width,
                    "err": err, "runtime": rt,
                })
                print(f"  seed={seed} p={density:.2f} {label:32s}"
                      f" ate={ate:+.3f} CI=[{lo:+.3f},{hi:+.3f}] "
                      f"cov={cov} ({rt:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby(["density", "estimator"]).agg(
        n=("seed", "count"),
        ate=("ate_hat", "mean"),
        bias=("ate_hat", lambda s: float(np.mean(s - TRUE_ATE))),
        coverage=("cov", "mean"),
        ci_width=("ci_width", "mean"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "coverage_comparison.md"
    lines = [
        "# Coverage comparison: RX-Learner MCMC posterior vs Huber-DR bootstrap CI",
        "",
        f"Whale DGP, N = {N}, true ATE = {TRUE_ATE}, 3 seeds. 95 % CI level.",
        "Huber-DR bootstrap is on the full (nuisance + Huber) pipeline (200 replicates).",
        "",
        "| density | estimator | n | mean ATE | bias | 95% coverage | CI width |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {r['estimator']} | {int(r['n'])} | "
            f"{r['ate']:+.3f} | {r['bias']:+.3f} | "
            f"{r['coverage']:.2f} | {r['ci_width']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "coverage_comparison_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'coverage_comparison_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string(index=False))
    _write_markdown(df, agg)


if __name__ == "__main__":
    main()
