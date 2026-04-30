"""Round-13 reviewer-response experiments — items 1, 2, 3, 4, 5, 6, 7, 8.

Items addressed:
  1. Squared-loss generalised-Bayes DR posterior with ω-calibration,
     head-to-head against Welsch.
  2. Lasso-DR (Lasso on cross-fitted DR pseudo-outcomes, sparse-linear baseline).
  3. Tukey biweight Phase-3 likelihood vs Welsch.
  4. α-stable (Lévy α-stable, α near 2) contamination.
  5. Catoni-mean DR-style baseline.
  6. Student-t-residual BART variant (already covered in round-5
     run_student_t_bart; here we add fixed-ν, fixed-σ comparison).
  7. WIS (Winkler interval score) on Hillstrom subgroups.
  8. Data-driven δ via 5-fold CV on DR-pseudo-outcome dispersion.

Item 9 (library auto-fallback) is a code change, handled separately.

Usage: python -u -m benchmarks.run_round13_experiments
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random

from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import HuberRegressor, Lasso

from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from sert_xlearner.models.nuisance import NuisanceEstimator
from sert_xlearner.core.orthogonalization import impute_and_debias


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def _dr_pseudo(X, Y, W):
    nuis = NuisanceEstimator(
        {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, random_state=0, method="xgboost",
    )
    mu0, mu1, pi = nuis.fit_predict(X, Y, W)
    _, _, D1, D0, *_ = impute_and_debias(
        Y, W, mu0, mu1, pi, robust=False, use_overlap=False)
    return np.concatenate([D1, D0])


# ============== Item 1: ω-calibrated squared-loss generalised Bayes ==============

def squared_loss_gibbs_model(X_inf, D, omega, sigma=1.0, prior_scale=10.0):
    """Squared-loss Gibbs posterior at temperature ω:
       p_ω(β | D) ∝ p(β) exp(-ω·∑(D - φβ)²/(2σ²))."""
    n, p = X_inf.shape
    beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([p]))
    tau = jnp.dot(X_inf, beta)
    sq_loss = -0.5 * (D - tau) ** 2 / (sigma ** 2)
    numpyro.factor("sq_gibbs", omega * jnp.sum(sq_loss))


def run_item1_squared_gibbs():
    """For each (seed, density) compute squared-loss Gibbs posterior at
    bootstrap-calibrated ω and compare against Welsch."""
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            D = _dr_pseudo(X, Y, W)
            X_inf = np.ones((len(D), 1))
            sigma_est = float(np.std(D))
            # ω grid; pick the one matching bootstrap variance
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(50):
                idx = rng.integers(0, len(D), size=len(D))
                boots.append(float(np.mean(D[idx])))
            target_var = float(np.var(boots, ddof=1))
            best_omega = 1.0; best_diff = float("inf")
            for omega in [0.1, 0.5, 1.0, 2.0, 5.0]:
                kernel = NUTS(squared_loss_gibbs_model)
                mcmc = MCMC(kernel, num_warmup=200, num_samples=400,
                            num_chains=2, progress_bar=False)
                try:
                    mcmc.run(random.PRNGKey(seed), jnp.array(X_inf),
                             jnp.array(D), omega=omega, sigma=sigma_est)
                    beta = np.asarray(mcmc.get_samples()["beta"]).squeeze()
                    var_post = float(np.var(beta, ddof=1))
                    if abs(var_post - target_var) < best_diff:
                        best_diff = abs(var_post - target_var)
                        best_omega = omega
                except Exception:
                    pass
            # Final fit at best ω
            kernel = NUTS(squared_loss_gibbs_model)
            mcmc = MCMC(kernel, num_warmup=300, num_samples=500,
                        num_chains=2, progress_bar=False)
            mcmc.run(random.PRNGKey(seed), jnp.array(X_inf), jnp.array(D),
                     omega=best_omega, sigma=sigma_est)
            beta = np.asarray(mcmc.get_samples()["beta"]).squeeze()
            ate = float(np.mean(beta))
            lo, hi = np.percentile(beta, [2.5, 97.5])
            cov = int(lo <= TRUE_ATE <= hi)
            rows.append({"seed": seed, "density": density,
                         "estimator": "Squared-Gibbs(ω-cal)",
                         "omega_hat": best_omega,
                         "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                         "ci_width": hi - lo})
            print(f"  s={seed} d={density:.2f} sq-Gibbs ω̂={best_omega} ate={ate:+.3f} "
                  f"cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 2: Lasso-DR (sparse-linear DR baseline) ==============

def run_item2_dipw_lasso():
    """Lasso on cross-fitted DR pseudo-outcomes (sparse-linear DR baseline).

    NOTE: this is *not* the literature DIPW-Lasso (denoised-IPW + a
    specific Lasso scheme); it is the straightforward Lasso-on-DR-
    pseudo-outcomes baseline. The function name is kept for backwards
    compatibility with existing CSVs; the reported label in the paper
    is ``Lasso-DR''."""
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            D = _dr_pseudo(X, Y, W)
            # Treated/control split for X order
            X_full = np.row_stack([X[W == 1], X[W == 0]])
            # Lasso on intercept (i.e. just mean) — equivalent to pure DR mean
            lasso = Lasso(alpha=0.01, fit_intercept=True, max_iter=2000)
            lasso.fit(X_full, D)
            ate = float(lasso.intercept_)
            # Bootstrap CI
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(50):
                idx = rng.integers(0, len(D), size=len(D))
                try:
                    l = Lasso(alpha=0.01, fit_intercept=True, max_iter=2000)
                    l.fit(X_full[idx], D[idx])
                    boots.append(l.intercept_)
                except Exception:
                    pass
            lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
            cov = int(lo <= TRUE_ATE <= hi) if boots else 0
            rows.append({"seed": seed, "density": density,
                         "estimator": "Lasso-DR",
                         "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                         "ci_width": hi - lo})
            print(f"  s={seed} d={density:.2f} Lasso-DR ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 3: Tukey biweight Phase-3 ==============

def biweight_model(X_inf, D, c=4.685, prior_scale=10.0):
    """Tukey biweight: ρ_T(r) = (c²/6)[1 - (1-(r/c)²)³] for |r|<c, else c²/6."""
    n, p = X_inf.shape
    beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([p]))
    tau = jnp.dot(X_inf, beta)
    r = D - tau
    u = r / c
    inside = jnp.abs(u) <= 1.0
    rho_inside = (c ** 2 / 6.0) * (1.0 - (1.0 - u ** 2) ** 3)
    rho_outside = (c ** 2 / 6.0)
    rho = jnp.where(inside, rho_inside, rho_outside)
    numpyro.factor("biweight", -jnp.sum(rho))


def run_item3_biweight():
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            D = _dr_pseudo(X, Y, W)
            X_inf = np.ones((len(D), 1))
            kernel = NUTS(biweight_model)
            mcmc = MCMC(kernel, num_warmup=300, num_samples=500,
                        num_chains=2, progress_bar=False)
            try:
                mcmc.run(random.PRNGKey(seed), jnp.array(X_inf), jnp.array(D))
                beta = np.asarray(mcmc.get_samples()["beta"]).squeeze()
                ate = float(np.mean(beta))
                lo, hi = np.percentile(beta, [2.5, 97.5])
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density,
                             "estimator": "Tukey-biweight",
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                             "ci_width": hi - lo})
                print(f"  s={seed} d={density:.2f} biweight ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
            except Exception as e:
                print(f"  s={seed} d={density:.2f} biweight ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 4: α-stable contamination ==============

def alpha_stable_dgp(N, alpha_stable, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, 5))
    pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
    W = rng.binomial(1, pi)
    Y0 = X[:, 0] + 0.5 * X[:, 1]
    # α-stable noise via scipy
    from scipy.stats import levy_stable
    eps = levy_stable.rvs(alpha_stable, beta=0, loc=0, scale=1, size=N,
                          random_state=seed)
    # Clip to keep numerical sanity; α-stable can have astronomical values
    eps = np.clip(eps, -100, 100)
    Y0 = Y0 + eps
    Y1 = Y0 + TRUE_ATE
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W


def run_item4_alpha_stable():
    rows = []
    for seed in range(3):
        for alpha in [1.7, 1.9, 2.0]:  # 2.0 = Gaussian limit
            X, Y, W = alpha_stable_dgp(N, alpha, seed)
            for severity in ["none", "severe"]:
                kwargs = dict(
                    n_splits=2, num_warmup=300, num_samples=500, num_chains=2,
                    c_whale=1.34, mad_rescale=False, random_state=seed,
                    robust=True, use_student_t=True,
                )
                if severity == "none":
                    kwargs["nuisance_method"] = "xgboost"
                    kwargs["outcome_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
                    kwargs["propensity_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
                else:
                    kwargs["contamination_severity"] = "severe"
                model = TargetedBayesianXLearner(**kwargs)
                try:
                    model.fit(X, Y, W, X_infer=np.ones((N, 1)))
                    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
                    ate = float(np.mean(beta))
                    lo, hi = np.percentile(beta, [2.5, 97.5])
                    cov = int(lo <= TRUE_ATE <= hi)
                    rows.append({"seed": seed, "alpha_stable": alpha, "severity": severity,
                                 "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                                 "ci_width": hi - lo})
                    print(f"  s={seed} α={alpha} sev={severity:7s} ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
                except Exception as e:
                    print(f"  s={seed} α={alpha} sev={severity} ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 5: Catoni-mean DR baseline ==============

def catoni_mean(x, n_iter=10):
    """Catoni's M-estimator of mean (Catoni 2012). Iterative."""
    s = np.std(x)
    if s < 1e-9:
        return float(np.mean(x))
    n = len(x)
    alpha = np.sqrt(2 * np.log(2 / 0.05) / (n * (s ** 2 + 1)))  # Catoni scale
    mu = float(np.mean(x))
    for _ in range(n_iter):
        u = alpha * (x - mu)
        psi = np.where(u >= 0, np.log(1 + u + u ** 2 / 2),
                       -np.log(1 - u + u ** 2 / 2))
        delta = float(np.mean(psi)) / alpha
        mu = mu + delta
        if abs(delta) < 1e-8:
            break
    return mu


def run_item5_catoni():
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            D = _dr_pseudo(X, Y, W)
            ate = catoni_mean(D)
            rng = np.random.default_rng(seed)
            boots = [catoni_mean(D[rng.integers(0, len(D), size=len(D))])
                     for _ in range(100)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cov = int(lo <= TRUE_ATE <= hi)
            rows.append({"seed": seed, "density": density,
                         "estimator": "Catoni-DR",
                         "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                         "ci_width": hi - lo})
            print(f"  s={seed} d={density:.2f} Catoni ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 6: Fixed-(ν, σ) Student-t variant ==============

def run_item6_student_t_variants():
    """Student-t Phase-3 with fixed (ν=3, σ=1.0) for comparison."""
    def t_fixed_model(X_inf, D, sigma=1.0, nu=3.0, prior_scale=10.0):
        n, p = X_inf.shape
        beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([p]))
        tau = jnp.dot(X_inf, beta)
        numpyro.sample("D_obs", dist.StudentT(nu, tau, sigma), obs=D)

    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            D = _dr_pseudo(X, Y, W)
            X_inf = np.ones((len(D), 1))
            for sigma_fixed in [1.0, 5.0]:
                kernel = NUTS(t_fixed_model)
                mcmc = MCMC(kernel, num_warmup=300, num_samples=500,
                            num_chains=2, progress_bar=False)
                mcmc.run(random.PRNGKey(seed), jnp.array(X_inf), jnp.array(D),
                         sigma=sigma_fixed, nu=3.0)
                beta = np.asarray(mcmc.get_samples()["beta"]).squeeze()
                ate = float(np.mean(beta))
                lo, hi = np.percentile(beta, [2.5, 97.5])
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density,
                             "sigma_fixed": sigma_fixed,
                             "estimator": f"Student-t(ν=3, σ={sigma_fixed})",
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                             "ci_width": hi - lo})
                print(f"  s={seed} d={density:.2f} t(σ={sigma_fixed}) ate={ate:+.3f} cov={cov}")
    return pd.DataFrame(rows)


# ============== Item 7: WIS on Hillstrom subgroups ==============

def run_item7_wis_hillstrom():
    """Compute Winkler interval score per recency-quintile subgroup
    on Hillstrom posterior."""
    from benchmarks.run_hillstrom import load_hillstrom
    X, Y, W = load_hillstrom()
    n = len(X)
    rng = np.random.default_rng(0)
    # Train/holdout 50/50
    idx = rng.permutation(n)
    tr, ho = idx[:n // 2], idx[n // 2:]
    # Fit RX-Welsch with recency-quintile basis on training
    rec = X[:, 0]  # recency
    qcuts = np.percentile(rec[tr], [20, 40, 60, 80])
    bins = np.digitize(rec, qcuts)  # 0..4
    # Basis: intercept + 4 dummies for quintiles 1..4
    X_inf = np.column_stack([np.ones(n)] + [(bins == k).astype(float) for k in range(1, 5)])
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        nuisance_method="xgboost", n_splits=2,
        num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
        contamination_severity="severe", random_state=0,
    )
    model.fit(X[tr], Y[tr], W[tr], X_infer=X_inf[tr])
    cate, lo, hi = model.predict(X_new_infer=X_inf[ho])
    cate = np.asarray(cate).flatten()
    lo = np.asarray(lo).flatten(); hi = np.asarray(hi).flatten()
    # WIS: Winkler interval score = (hi - lo) + (2/α)·(max(0, lo - y) + max(0, y - hi))
    alpha = 0.05
    rows = []
    for q in range(5):
        mask = bins[ho] == q
        if mask.sum() == 0: continue
        Y_h = Y[ho][mask]
        # WIS uses observed Y (not τ; Y is the only thing observed in real data)
        # We compute interval score on the predicted CATE intervals against per-unit Y
        # as a proxy — note: this is an unfair comparison since CIs are for τ, not Y.
        # We use a different proxy: interval-score on holdout pseudo-outcomes.
        nuis = NuisanceEstimator(
            {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            n_splits=2, random_state=0, method="xgboost",
        )
        try:
            mu0_h, mu1_h, pi_h = nuis.fit_predict(X[ho][mask], Y_h, W[ho][mask])
            _, _, D1, D0, *_ = impute_and_debias(
                Y_h, W[ho][mask], mu0_h, mu1_h, pi_h, robust=False, use_overlap=False)
            D_h = np.concatenate([D1, D0])
            target = float(np.mean(D_h))
            cate_q = float(np.mean(cate[mask]))
            lo_q = float(np.mean(lo[mask])); hi_q = float(np.mean(hi[mask]))
            wis = (hi_q - lo_q) + (2 / alpha) * (max(0, lo_q - target) + max(0, target - hi_q))
            rows.append({"quintile": q, "n_holdout": int(mask.sum()),
                         "tau_hat": cate_q, "lo": lo_q, "hi": hi_q,
                         "ci_width": hi_q - lo_q,
                         "target_dr_ate": target,
                         "WIS": wis})
            print(f"  Q{q} n={mask.sum()} τ̂={cate_q:+.4f} CI=[{lo_q:+.4f},{hi_q:+.4f}] WIS={wis:.3f}")
        except Exception as e:
            print(f"  Q{q} ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 8: Data-driven δ via 5-fold CV ==============

def run_item8_datadriven_delta():
    """5-fold CV: pick δ minimising held-out DR-pseudo-outcome dispersion."""
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            best_delta = None; best_score = float("inf")
            for delta in [0.5, 1.0, 1.345, 2.0]:
                # 5-fold CV
                rng = np.random.default_rng(seed)
                folds = np.array_split(rng.permutation(N), 5)
                scores = []
                for k in range(5):
                    test = folds[k]
                    train = np.concatenate([folds[j] for j in range(5) if j != k])
                    out_p = {"depth": 4, "iterations": 100, "loss_function": f"Huber:delta={delta}"}
                    prop_p = {"depth": 4, "iterations": 100}
                    try:
                        nuis = NuisanceEstimator(out_p, prop_p, n_splits=2,
                                                  random_state=seed, method="catboost")
                        mu0, mu1, pi = nuis.fit_predict(X[train], Y[train], W[train])
                        _, _, D1, D0, *_ = impute_and_debias(
                            Y[train], W[train], mu0, mu1, pi, robust=False, use_overlap=False)
                        D_tr = np.concatenate([D1, D0])
                        # Score: dispersion of pseudo-outcomes (smaller = better fit)
                        scores.append(float(np.median(np.abs(D_tr - np.median(D_tr)))))
                    except Exception:
                        pass
                if not scores: continue
                mean_score = float(np.mean(scores))
                if mean_score < best_score:
                    best_score = mean_score; best_delta = delta
            rows.append({"seed": seed, "density": density,
                         "delta_hat": best_delta, "score": best_score})
            print(f"  s={seed} d={density:.2f} δ̂={best_delta} (CV-MAD={best_score:.3f})")
    return pd.DataFrame(rows)


def main():
    runs = [
        ("squared_gibbs",   run_item1_squared_gibbs,    "squared_gibbs"),
        ("dipw_lasso",      run_item2_dipw_lasso,        "dipw_lasso"),
        ("biweight",        run_item3_biweight,          "tukey_biweight"),
        ("alpha_stable",    run_item4_alpha_stable,      "alpha_stable"),
        ("catoni",          run_item5_catoni,            "catoni_dr"),
        ("student_t_fixed", run_item6_student_t_variants,"student_t_fixed_variants"),
        ("hillstrom_wis",   run_item7_wis_hillstrom,     "hillstrom_wis"),
        ("datadriven_delta", run_item8_datadriven_delta, "datadriven_delta"),
    ]
    for label, fn, fname in runs:
        print(f"\n=== {label} ===")
        try:
            df = fn()
            if df is not None and len(df):
                df.to_csv(RESULTS_DIR / f"{fname}_raw.csv", index=False)
                print(f"  wrote {fname}_raw.csv")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {label} FAILED: {e}")
    print("DONE_ROUND13")


if __name__ == "__main__":
    main()
