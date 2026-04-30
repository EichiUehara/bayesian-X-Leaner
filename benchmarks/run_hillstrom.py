"""Hillstrom email marketing — real-world heavy-tailed RCT.

Public dataset (Kevin Hillstrom 2008): 64,000 customers, randomised
to (no email, men's email, women's email). Outcomes are visit/spend.
Spend is heavy-tailed (most zeros, a long right tail).

We restrict to two arms (no-email control vs men's-email treatment)
and use spend as the outcome. The dataset has known random treatment
assignment, so the ATE is identifiable; per-unit τ(x) is unobserved
(no ground truth for PEHE), so this experiment reports point ATE
and 95% credible/confidence interval, comparing methods on real
heavy-tailed data.

Source URL: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
We auto-download the CSV on first run.

Usage: python -u -m benchmarks.run_hillstrom --seeds 1
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)

from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"
HILLSTROM_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
HILLSTROM_PATH = DATA_DIR / "hillstrom.csv"


def _download_hillstrom():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if HILLSTROM_PATH.exists():
        return
    print(f"Downloading Hillstrom from {HILLSTROM_URL}")
    urllib.request.urlretrieve(HILLSTROM_URL, HILLSTROM_PATH)


def load_hillstrom():
    _download_hillstrom()
    df = pd.read_csv(HILLSTROM_PATH)
    # Two arms: No E-Mail (control) vs Mens E-Mail (treatment)
    df = df[df["segment"].isin(["No E-Mail", "Mens E-Mail"])].copy()
    df["W"] = (df["segment"] == "Mens E-Mail").astype(int)
    Y = df["spend"].to_numpy(dtype=float)
    W = df["W"].to_numpy(dtype=int)
    cov_cols = ["recency", "history", "mens", "womens", "newbie"]
    # Encode zip_code and channel as numeric
    df["zip_code_num"] = df["zip_code"].astype("category").cat.codes
    df["channel_num"] = df["channel"].astype("category").cat.codes
    cov_cols += ["zip_code_num", "channel_num"]
    X = df[cov_cols].to_numpy(dtype=float)
    return X, Y, W


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def naive_ate(Y, W):
    """Difference of means + asymptotic 95% CI."""
    m1 = float(np.mean(Y[W == 1])); m0 = float(np.mean(Y[W == 0]))
    s1 = float(np.var(Y[W == 1], ddof=1) / W.sum())
    s0 = float(np.var(Y[W == 0], ddof=1) / (1 - W).sum())
    se = np.sqrt(s1 + s0)
    ate = m1 - m0
    return ate, ate - 1.96 * se, ate + 1.96 * se


def huber_dr(X, Y, W):
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
    mu0_all = mu0.predict(X); mu1_all = mu1.predict(X)
    D = np.where(
        W == 1,
        mu1_all - mu0_all + (Y - mu1_all) / pi,
        mu1_all - mu0_all - (Y - mu0_all) / (1.0 - pi),
    )
    Xones = np.ones((len(X), 1))
    reg = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
    reg.fit(Xones, D)
    ate = float(reg.coef_[0])
    # Bootstrap CI (50 reps to keep it fast on full N)
    rng = np.random.default_rng(0)
    boots = []
    Nfull = len(X)
    for _ in range(50):
        idx = rng.integers(0, Nfull, size=Nfull)
        Xb, Yb, Wb = X[idx], Y[idx], W[idx]
        try:
            mu0 = _make_reg(); mu0.fit(Xb[Wb == 0], Yb[Wb == 0])
            mu1 = _make_reg(); mu1.fit(Xb[Wb == 1], Yb[Wb == 1])
            pi_m = _make_clf(); pi_m.fit(Xb, Wb)
            pi = np.clip(pi_m.predict_proba(Xb)[:, 1], 0.05, 0.95)
            mu0_b = mu0.predict(Xb); mu1_b = mu1.predict(Xb)
            Db = np.where(
                Wb == 1,
                mu1_b - mu0_b + (Yb - mu1_b) / pi,
                mu1_b - mu0_b - (Yb - mu0_b) / (1.0 - pi),
            )
            r = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
            r.fit(np.ones((Nfull, 1)), Db)
            boots.append(r.coef_[0])
        except Exception:
            pass
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return ate, float(lo), float(hi)
    return ate, float("nan"), float("nan")


def rx_learner(X, Y, W, severity, seed=0):
    n = len(X)
    kwargs = dict(
        n_splits=2,
        num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, mad_rescale=False, random_state=seed,
        robust=True, use_student_t=True,
    )
    if severity == "none":
        kwargs["nuisance_method"] = "xgboost"
        kwargs["outcome_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        kwargs["propensity_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
    elif severity == "severe":
        kwargs["contamination_severity"] = "severe"
    model = TargetedBayesianXLearner(**kwargs)
    model.fit(X, Y, W, X_infer=np.ones((n, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    ate = float(np.mean(beta))
    lo, hi = np.percentile(beta, [2.5, 97.5])
    return ate, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=1,
                    help="Sub-sample replicates (the data is fixed; this is for stochastic estimators).")
    args = ap.parse_args()

    print("Loading Hillstrom...")
    X, Y, W = load_hillstrom()
    print(f"  N={len(X)}, treated={int(W.sum())}, p_treated={float(W.mean()):.3f}")
    print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}], mean={Y.mean():.4f}, "
          f"99th-pct={np.percentile(Y, 99):.2f}, max non-zero seen "
          f"in {(Y > 0).sum()} of {len(Y)} units")

    rows = []
    for seed in range(args.seeds):
        for label, fit in [
            ("Naive (difference of means)",  lambda: naive_ate(Y, W)),
            ("Huber-DR (bootstrap CI)",      lambda: huber_dr(X, Y, W)),
            ("RX-Welsch (severity=none)",    lambda: rx_learner(X, Y, W, "none",   seed)),
            ("RX-Welsch (severity=severe)",  lambda: rx_learner(X, Y, W, "severe", seed)),
        ]:
            t0 = time.time()
            try:
                ate, lo, hi = fit()
                err = None
            except Exception as e:
                ate = lo = hi = float("nan"); err = str(e)
            rt = time.time() - t0
            rows.append({
                "seed": seed, "estimator": label, "ate_hat": ate,
                "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo, "runtime": rt, "err": err,
            })
            print(f"  seed={seed} {label:36s} ate={ate:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                  f"({rt:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hillstrom_raw.csv", index=False)

    md = ["# Hillstrom email marketing — real heavy-tailed RCT",
          "",
          f"Dataset: Kevin Hillstrom MineThatData (2008). Two arms: No E-Mail "
          f"(control) vs Mens E-Mail (treatment). N={len(X)}, treated={int(W.sum())}.",
          f"Outcome: 2-week spend ($). Heavy right tail; "
          f"mean = {Y.mean():.4f}, 99th pct = {np.percentile(Y, 99):.2f}.",
          "Ground-truth ATE is identifiable via random assignment but no per-unit τ(x).",
          "",
          "| Estimator | ATE | 95% CI | width | runtime |",
          "|---|---:|---|---:|---:|"]
    for _, r in df.iterrows():
        md.append(
            f"| {r['estimator']} | {r['ate_hat']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{r['ci_width']:.4f} | {r['runtime']:.1f}s |"
        )
    (RESULTS_DIR / "hillstrom.md").write_text("\n".join(md))
    print(f"\nwrote {RESULTS_DIR / 'hillstrom.md'}")


if __name__ == "__main__":
    main()
