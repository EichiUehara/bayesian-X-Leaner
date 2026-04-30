"""Round-9 reviewer-response experiments — items 7-10.

  7. RBCI-style ω-tuning by bootstrap interval-score minimisation
     (vs our trace-formula η-calibration).
  8. Hillstrom propensity-stratified placebo: stratify on baseline
     spend propensity, permute W within strata, verify near-zero ATE.
  9. Cross-fit dispersion diagnostic: a simple statistic for "when
     should I turn on modular pooling?"
 10. Permutation-based ATE intervals as the aligned baseline
     replacing/supplementing conformal-DR for ATE inference.

Usage: python -u -m benchmarks.run_round9_experiments
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import HuberRegressor

from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


# ============== Item 7: RBCI-style ω by interval-score minimisation ==============

def interval_score(lo, hi, y, alpha=0.05):
    """Winkler interval score (negatively oriented; lower is better).
    For a (1-α) interval [lo, hi] and observation y."""
    width = hi - lo
    pen = (2 / alpha) * (np.maximum(0, lo - y) + np.maximum(0, y - hi))
    return float(np.mean(width + pen))


def fit_rx_at_eta(X, Y, W, eta, severity, seed):
    """Refit RX-Welsch with c → c/sqrt(eta), return posterior."""
    c_eta = 1.34 / np.sqrt(eta)
    kwargs = dict(
        n_splits=2, num_warmup=300, num_samples=500, num_chains=2,
        c_whale=c_eta, mad_rescale=False, random_state=seed,
        robust=True, use_student_t=True,
    )
    if severity == "none":
        kwargs["nuisance_method"] = "xgboost"
        kwargs["outcome_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        kwargs["propensity_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
    else:
        kwargs["contamination_severity"] = "severe"
    model = TargetedBayesianXLearner(**kwargs)
    model.fit(X, Y, W, X_infer=np.ones((N, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    return beta


def rbci_omega_select(X, Y, W, severity, seed, eta_grid=(0.5, 1.0, 2.0), B=20):
    """RBCI-style: pick ω to minimise the interval score on bootstrap
    pseudo-truth values. The bootstrap samples give a target distribution
    of ATE values; for each ω the interval score is computed across
    bootstrap pseudo-truths, summed."""
    rng = np.random.default_rng(seed)
    # Bootstrap distribution of Huber-DR ATE estimates (proxy for the
    # sampling distribution of the true ATE under the model)
    Nfull = len(X)
    boot_ates = []
    for _ in range(B):
        idx = rng.integers(0, Nfull, size=Nfull)
        try:
            mu0 = _make_reg(); mu0.fit(X[idx][W[idx] == 0], Y[idx][W[idx] == 0])
            mu1 = _make_reg(); mu1.fit(X[idx][W[idx] == 1], Y[idx][W[idx] == 1])
            pi_m = _make_clf(); pi_m.fit(X[idx], W[idx])
            pi = np.clip(pi_m.predict_proba(X[idx])[:, 1], 0.05, 0.95)
            mu0_a = mu0.predict(X[idx]); mu1_a = mu1.predict(X[idx])
            D = np.where(W[idx] == 1,
                         mu1_a - mu0_a + (Y[idx] - mu1_a) / pi,
                         mu1_a - mu0_a - (Y[idx] - mu0_a) / (1 - pi))
            r = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
            r.fit(np.ones((Nfull, 1)), D)
            boot_ates.append(r.coef_[0])
        except Exception:
            pass
    boot_ates = np.array(boot_ates)
    # Score each candidate η by the interval score over bootstrap targets
    best_score = float("inf"); best_eta = 1.0
    for eta in eta_grid:
        beta = fit_rx_at_eta(X, Y, W, eta, severity, seed)
        lo, hi = np.percentile(beta, [2.5, 97.5])
        # Average interval score across bootstrap pseudo-truths
        score = float(np.mean(
            (hi - lo) + (2 / 0.05) * (np.maximum(0, lo - boot_ates) + np.maximum(0, boot_ates - hi))
        ))
        if score < best_score:
            best_score = score; best_eta = eta
    # Final fit at selected ω
    beta = fit_rx_at_eta(X, Y, W, best_eta, severity, seed)
    return best_eta, float(np.mean(beta)), float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def run_item7_rbci():
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            for severity in ["none", "severe"]:
                eta_hat, ate, lo, hi = rbci_omega_select(X, Y, W, severity, seed)
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density, "severity": severity,
                             "eta_rbci": eta_hat, "ate": ate, "lo": lo, "hi": hi,
                             "cov": cov, "ci_width": hi - lo})
                print(f"  s={seed} p={density:.2f} sev={severity:7s} ω̂={eta_hat:.2f} "
                      f"ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 8: Hillstrom propensity-stratified placebo ==============

def run_item8_hillstrom_strat_placebo():
    from benchmarks.run_hillstrom import load_hillstrom
    X, Y, W = load_hillstrom()
    n = len(X)
    rng = np.random.default_rng(0)
    # Stratify by pre-treatment spend propensity (proxied by history)
    # Quintiles
    history = X[:, 1]
    quintiles = np.digitize(history, np.percentile(history, [20, 40, 60, 80]))
    # Permute W within each stratum
    W_perm = np.copy(W)
    for q in np.unique(quintiles):
        mask = quintiles == q
        W_perm[mask] = rng.permutation(W[mask])
    rows = []
    for label, W_use in [("Original", W), ("Stratified-permuted", W_perm)]:
        model = TargetedBayesianXLearner(
            outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            nuisance_method="xgboost", n_splits=2,
            num_warmup=400, num_samples=800, num_chains=2,
            robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
            contamination_severity="severe", random_state=0,
        )
        model.fit(X, Y, W_use, X_infer=np.ones((n, 1)))
        beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
        ate = float(np.mean(beta))
        lo, hi = np.percentile(beta, [2.5, 97.5])
        rows.append({"label": label, "ate": ate, "lo": lo, "hi": hi,
                     "ci_width": hi - lo})
        print(f"  {label:25s} ate={ate:+.5f} CI=[{lo:+.5f},{hi:+.5f}]")
    return pd.DataFrame(rows)


# ============== Item 9: Cross-fit dispersion diagnostic ==============

def run_item9_dispersion():
    """For each (seed, density, severity), fit RX-Welsch K times with
    different cross-fitting seeds, report the dispersion of posterior
    means as a diagnostic for whether modular pooling is needed."""
    rows = []
    for seed in range(5):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            for severity in ["none", "severe"]:
                ates = []; widths = []
                for cf_seed in range(5):
                    kwargs = dict(
                        n_splits=2, num_warmup=200, num_samples=400, num_chains=2,
                        c_whale=1.34, mad_rescale=False,
                        random_state=seed * 100 + cf_seed,
                        robust=True, use_student_t=True,
                    )
                    if severity == "none":
                        kwargs["nuisance_method"] = "xgboost"
                        kwargs["outcome_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
                        kwargs["propensity_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
                    else:
                        kwargs["contamination_severity"] = "severe"
                    try:
                        m = TargetedBayesianXLearner(**kwargs)
                        m.fit(X, Y, W, X_infer=np.ones((N, 1)))
                        beta = np.asarray(m.mcmc_samples["beta"]).squeeze()
                        ates.append(float(np.mean(beta)))
                        widths.append(float(np.percentile(beta, 97.5) - np.percentile(beta, 2.5)))
                    except Exception:
                        pass
                if not ates:
                    continue
                disp = float(np.std(ates))
                mean_w = float(np.mean(widths))
                # Simple ratio: dispersion / typical width
                ratio = disp / mean_w if mean_w > 0 else float("nan")
                rows.append({"seed": seed, "density": density, "severity": severity,
                             "cross_fit_dispersion": disp,
                             "mean_ci_width": mean_w,
                             "dispersion_ratio": ratio,
                             "n_cf": len(ates)})
                print(f"  s={seed} p={density:.2f} sev={severity:7s} "
                      f"disp={disp:.3f} w={mean_w:.3f} ratio={ratio:.3f}")
    return pd.DataFrame(rows)


# ============== Item 10: Permutation-based ATE intervals ==============

def run_item10_permutation_ate():
    """Permutation-based ATE confidence intervals: under the sharp null
    H0: τ(x) = 0, the distribution of the test statistic under permuted W
    gives a null reference. We invert this to a confidence interval by
    finding the range of τ values for which the observed minus τ shifted
    statistic does not reject."""
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            # Simple ATE estimator: difference of Huber means
            def stat(Y_, W_):
                m1 = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=True)
                m1.fit(np.ones((W_.sum(), 1)), Y_[W_ == 1])
                m0 = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=True)
                m0.fit(np.ones(((1 - W_).sum(), 1)), Y_[W_ == 0])
                return float(m1.intercept_) - float(m0.intercept_)
            ate_obs = stat(Y, W)
            # Permutation null
            rng = np.random.default_rng(seed)
            null_stats = []
            for _ in range(200):
                W_perm = rng.permutation(W)
                try:
                    null_stats.append(stat(Y, W_perm))
                except Exception:
                    pass
            null_stats = np.array(null_stats)
            # 95% interval inversion: τ values for which (ate_obs - τ) is in
            # the central 95% of the null
            null_lo, null_hi = np.percentile(null_stats, [2.5, 97.5])
            ci_lo = ate_obs - null_hi
            ci_hi = ate_obs - null_lo
            cov = int(ci_lo <= TRUE_ATE <= ci_hi)
            rows.append({"seed": seed, "density": density, "ate_obs": ate_obs,
                         "ci_lo": ci_lo, "ci_hi": ci_hi, "cov": cov,
                         "ci_width": ci_hi - ci_lo, "n_perm": len(null_stats)})
            print(f"  s={seed} p={density:.2f} ate={ate_obs:+.3f} "
                  f"CI=[{ci_lo:+.3f},{ci_hi:+.3f}] cov={cov} w={ci_hi-ci_lo:.3f}")
    return pd.DataFrame(rows)


# ============== Main ==============

def main():
    runs = [
        ("rbci_omega",       run_item7_rbci,                   "rbci_omega"),
        ("hillstrom_strat",  run_item8_hillstrom_strat_placebo, "hillstrom_strat_placebo"),
        ("dispersion_diag",  run_item9_dispersion,             "crossfit_dispersion"),
        ("permutation_ate",  run_item10_permutation_ate,        "permutation_ate"),
    ]
    for label, fn, fname in runs:
        print(f"\n=== {label} ===")
        try:
            df = fn()
            df.to_csv(RESULTS_DIR / f"{fname}_raw.csv", index=False)
            print(f"  wrote {fname}_raw.csv")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {label} FAILED: {e}")


if __name__ == "__main__":
    main()
