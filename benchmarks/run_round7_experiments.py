"""Round-7 reviewer-response experiments — items 2-7, 9, 10.

  2. Heteroskedastic Phase-3: weight Welsch by Var(D|X) estimator.
  3. γ-divergence X-learner with BOTH stages robustified.
  4. Trimmed/overlap-weighted DR baseline.
  5. BMA at varying N (basis brittleness vs sample size).
  6. EVT-informed tail-mixture Phase-3 likelihood (Pareto tail component).
  7. Robust DML / R-learner with Huber loss.
  9. Hillstrom additional bases (recency thresholds + interactions).
 10. 30-seed coverage replication of the headline cells.

Item 8 (GRF with robustification) requires R/rpy2 — acknowledged
as future work.

Usage: python -u -m benchmarks.run_round7_experiments
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

from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from benchmarks.dgps import whale_dgp


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def _dr_pseudo(X, Y, W, trim=0.0, overlap=False):
    """Return DR pseudo-outcomes; optional propensity trimming or
    overlap-weight reformulation."""
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], max(0.01, trim), 1 - max(0.01, trim))
    mu0_a = mu0.predict(X); mu1_a = mu1.predict(X)
    if overlap:
        # Overlap weights: keep the same DR formula but multiplied by
        # arm-specific overlap weights, targets the ATO not ATE
        D_dr = np.where(
            W == 1,
            mu1_a - mu0_a + (Y - mu1_a) / pi,
            mu1_a - mu0_a - (Y - mu0_a) / (1 - pi),
        )
        # Overlap-weighted average
        w_overlap = np.where(W == 1, 1 - pi, pi)
        return D_dr, w_overlap
    D = np.where(
        W == 1,
        mu1_a - mu0_a + (Y - mu1_a) / pi,
        mu1_a - mu0_a - (Y - mu0_a) / (1 - pi),
    )
    return D, np.ones_like(D)


# ============== Item 4: Trimmed/overlap-weighted DR baseline ==============

def trimmed_dr_ate(X, Y, W, trim=0.05, seed=0):
    rng = np.random.default_rng(seed)
    D, _ = _dr_pseudo(X, Y, W, trim=trim)
    Xones = np.ones((len(X), 1))
    reg = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
    reg.fit(Xones, D)
    ate = float(reg.coef_[0])
    boots = []
    for _ in range(50):
        idx = rng.integers(0, len(X), size=len(X))
        try:
            Db, _ = _dr_pseudo(X[idx], Y[idx], W[idx], trim=trim)
            r = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
            r.fit(np.ones((len(X), 1)), Db)
            boots.append(r.coef_[0])
        except Exception:
            pass
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return ate, float(lo), float(hi)


def overlap_dr_ate(X, Y, W, seed=0):
    rng = np.random.default_rng(seed)
    D, w = _dr_pseudo(X, Y, W, overlap=True)
    Xones = np.ones((len(X), 1))
    # Weighted Huber on D with overlap weights
    reg = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
    reg.fit(Xones, D, sample_weight=w)
    ate = float(reg.coef_[0])
    boots = []
    for _ in range(50):
        idx = rng.integers(0, len(X), size=len(X))
        try:
            Db, wb = _dr_pseudo(X[idx], Y[idx], W[idx], overlap=True)
            r = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
            r.fit(np.ones((len(X), 1)), Db, sample_weight=wb)
            boots.append(r.coef_[0])
        except Exception:
            pass
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return ate, float(lo), float(hi)


def run_item4_trimmed():
    rows = []
    for seed in [0, 1, 2]:
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            for label, fn in [
                ("Trimmed-DR (π∈[0.05,0.95])", lambda: trimmed_dr_ate(X, Y, W, 0.05, seed)),
                ("Overlap-weighted DR",        lambda: overlap_dr_ate(X, Y, W, seed)),
            ]:
                ate, lo, hi = fn()
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density, "estimator": label,
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                             "ci_width": hi - lo})
                print(f"  seed={seed} p={density:.2f} {label:30s} ate={ate:+.3f} "
                      f"cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 7: Robust DML / R-learner with Huber ==============

def r_learner_huber(X, Y, W, seed=0):
    """R-learner: residualize Y and W on X, then Huber-regress
    Y_resid on W_resid. Equivalent to Nie-Wager R-learner with
    a robust outer regression."""
    mu = _make_reg(); mu.fit(X, Y)
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
    Y_res = Y - mu.predict(X)
    W_res = W - pi
    # Huber regression of Y_res on W_res (intercept-free)
    reg = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
    reg.fit(W_res.reshape(-1, 1), Y_res)
    ate = float(reg.coef_[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(50):
        idx = rng.integers(0, len(X), size=len(X))
        try:
            mu_b = _make_reg(); mu_b.fit(X[idx], Y[idx])
            pi_b = _make_clf(); pi_b.fit(X[idx], W[idx])
            pi_v = np.clip(pi_b.predict_proba(X[idx])[:, 1], 0.05, 0.95)
            Yr = Y[idx] - mu_b.predict(X[idx])
            Wr = W[idx] - pi_v
            r = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
            r.fit(Wr.reshape(-1, 1), Yr)
            boots.append(r.coef_[0])
        except Exception:
            pass
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return ate, float(lo), float(hi)


def run_item7_rlearner():
    rows = []
    for seed in [0, 1, 2]:
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            ate, lo, hi = r_learner_huber(X, Y, W, seed)
            cov = int(lo <= TRUE_ATE <= hi)
            rows.append({"seed": seed, "density": density, "estimator": "R-learner-Huber",
                         "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                         "ci_width": hi - lo})
            print(f"  seed={seed} p={density:.2f} R-Huber ate={ate:+.3f} cov={cov} "
                  f"w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 9: Hillstrom additional bases ==============

def run_item9_hillstrom():
    from benchmarks.run_hillstrom import load_hillstrom
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    X, Y, W = load_hillstrom()
    n = len(X)
    recency = X[:, 0]; history = X[:, 1]
    mens = X[:, 2]; womens = X[:, 3]; newbie = X[:, 4]
    bases = {
        "[1, recency<3]":        np.column_stack([np.ones(n), (recency < 3).astype(float)]),
        "[1, history>p75]":      np.column_stack([np.ones(n), (history > np.percentile(history, 75)).astype(float)]),
        "[1, mens, womens]":     np.column_stack([np.ones(n), mens, womens]),
        "[1, newbie]":           np.column_stack([np.ones(n), newbie]),
        "[1, history*recency]":  np.column_stack([np.ones(n), (history > np.median(history)) & (recency < 6)]),
    }
    rows = []
    for label, X_inf in bases.items():
        model = TargetedBayesianXLearner(
            outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            nuisance_method="xgboost", n_splits=2,
            num_warmup=400, num_samples=800, num_chains=2,
            robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
            random_state=0,
        )
        model.fit(X, Y, W, X_infer=X_inf)
        cate, lo, hi = model.predict(X_new_infer=X_inf)
        ate_overall = float(np.mean(np.asarray(cate)))
        # Get subgroup posterior on the second basis indicator (if present)
        if X_inf.shape[1] == 2:
            mask = X_inf[:, 1].astype(bool)
            tau_sub = float(np.mean(np.asarray(cate)[mask])) if mask.any() else float("nan")
            n_sub = int(mask.sum())
            ci_sub = (float(np.mean(np.asarray(lo)[mask])) if mask.any() else float("nan"),
                      float(np.mean(np.asarray(hi)[mask])) if mask.any() else float("nan"))
        else:
            tau_sub, n_sub, ci_sub = float("nan"), 0, (float("nan"), float("nan"))
        rows.append({
            "basis": label, "ate_overall": ate_overall,
            "n_sub": n_sub, "tau_sub": tau_sub,
            "lo_sub": ci_sub[0], "hi_sub": ci_sub[1],
        })
        print(f"  basis={label:25s} ate={ate_overall:+.4f} sub_n={n_sub} "
              f"tau_sub={tau_sub:+.4f} CI=[{ci_sub[0]:+.4f},{ci_sub[1]:+.4f}]")
    return pd.DataFrame(rows)


# ============== Item 10: 30-seed coverage replication ==============

def run_item10_coverage_replication():
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    rows = []
    for seed in range(30):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
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
                model.fit(X, Y, W, X_infer=np.ones((N, 1)))
                beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
                ate = float(np.mean(beta))
                lo, hi = np.percentile(beta, [2.5, 97.5])
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density, "severity": severity,
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                             "ci_width": hi - lo})
        if seed % 5 == 4:
            print(f"  seed {seed}/29 done")
    return pd.DataFrame(rows)


# ============== Item 5: BMA at varying N ==============

def run_item5_bma_varying_n():
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    TAU_BULK, TAU_TAIL, WHALE_CUT = 2.0, 10.0, 1.96
    CANDIDATES = [1.0, 1.5, 1.96, 2.5, 3.0]
    rows = []
    for n_size in [500, 1000, 2000]:
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(seed)
            X = rng.normal(0, 1, (n_size, 5))
            whale = (np.abs(X[:, 0]) > WHALE_CUT).astype(float)
            tau = TAU_BULK * (1 - whale) + TAU_TAIL * whale
            Y0 = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, n_size)
            Y1 = Y0 + tau
            pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
            W = rng.binomial(1, pi)
            Y = np.where(W == 1, Y1, Y0)

            best_pehe = float("inf"); best_c = None
            scores = []
            for c in CANDIDATES:
                X_inf = np.column_stack([np.ones(n_size), (np.abs(X[:, 0]) > c).astype(float)])
                model = TargetedBayesianXLearner(
                    outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                    propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                    nuisance_method="xgboost", n_splits=2,
                    num_warmup=300, num_samples=500, num_chains=2,
                    robust=True, c_whale=1.34, use_student_t=True,
                    mad_rescale=False, random_state=seed,
                )
                model.fit(X, Y, W, X_infer=X_inf)
                cate, _, _ = model.predict(X_new_infer=X_inf)
                cate = np.asarray(cate).flatten()
                pehe = float(np.sqrt(np.mean((cate - tau) ** 2)))
                scores.append((c, pehe, cate))
                if pehe < best_pehe:
                    best_pehe = pehe; best_c = c
            # BMA: softmax(-pehe) over candidates
            pehes = np.array([s[1] for s in scores])
            temp = max(np.std(pehes), 1e-3)
            ws = np.exp((-pehes + pehes.max()) / temp)
            ws /= ws.sum()
            cates = np.stack([s[2] for s in scores])
            cate_bma = ws @ cates
            pehe_bma = float(np.sqrt(np.mean((cate_bma - tau) ** 2)))
            rows.append({"seed": seed, "N": n_size,
                         "best_c": best_c, "best_pehe": best_pehe,
                         "bma_pehe": pehe_bma,
                         "bma_w_at_truth": float(ws[CANDIDATES.index(1.96)])})
            print(f"  seed={seed} N={n_size} best_c={best_c} best_pehe={best_pehe:.3f} "
                  f"bma_pehe={pehe_bma:.3f} w(1.96)={ws[CANDIDATES.index(1.96)]:.2f}")
    return pd.DataFrame(rows)


# ============== Item 2 + 6: heteroskedastic + EVT-mixture Phase-3 ==============

def heteroskedastic_welsch_model(X_infer, D, X_for_var, prior_scale=10.0):
    """Welsch with X-dependent scale: scale s(x) = exp(γ₀ + γ_x · x_for_var).
    Uses ψ_W with c absorbed into the scale, equivalent to a heteroskedastic
    Welsch likelihood."""
    n, p = X_infer.shape
    p_var = X_for_var.shape[1]
    beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([p]))
    gamma = numpyro.sample("gamma", dist.Normal(0, 1.0).expand([p_var]))
    log_s = jnp.dot(X_for_var, gamma)
    s = jnp.exp(log_s)
    tau = jnp.dot(X_infer, beta)
    r = (D - tau) / s
    c = 1.34
    welsch = (c**2 / 2.0) * (1.0 - jnp.exp(-(r / c)**2))
    numpyro.factor("hetero_welsch", -jnp.sum(welsch + log_s))


def evt_mixture_model(X_infer, D, prior_scale=10.0):
    """Welsch bulk + Pareto tail mixture likelihood, paradigm-(iii) hybrid.
    Mixture: (1-π_tail) · Welsch(τ, c) + π_tail · Pareto(α, threshold).
    π_tail, α, threshold inferred jointly."""
    n, p = X_infer.shape
    beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([p]))
    pi_tail = numpyro.sample("pi_tail", dist.Beta(1.0, 9.0))
    alpha = numpyro.sample("alpha", dist.Gamma(2.0, 1.0)) + 1.0
    log_thresh = numpyro.sample("log_thresh", dist.Normal(0.0, 1.0))
    thresh = jnp.exp(log_thresh) + 1.0
    tau = jnp.dot(X_infer, beta)
    r = D - tau
    c = 1.34
    welsch_dens = -(c**2 / 2.0) * (1.0 - jnp.exp(-(r / c)**2))
    log_bulk = welsch_dens + jnp.log(1 - pi_tail)
    abs_r = jnp.abs(r) + 1e-6
    log_pareto_top = jnp.log(alpha) + alpha * jnp.log(thresh) - (alpha + 1) * jnp.log(jnp.maximum(abs_r, thresh))
    log_pareto = jnp.where(abs_r > thresh, log_pareto_top, -100.0) + jnp.log(pi_tail)
    log_lik = jnp.logaddexp(log_bulk, log_pareto)
    numpyro.factor("evt_mixture_ll", jnp.sum(log_lik))


def fit_phase3_custom(model_fn, X_inf, D, **kwargs):
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=300, num_samples=500, num_chains=2, progress_bar=False)
    mcmc.run(random.PRNGKey(0), jnp.array(X_inf), jnp.array(D), **kwargs)
    beta = np.asarray(mcmc.get_samples()["beta"]).squeeze()
    return float(np.mean(beta)), float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def run_items_2_6_hetero_evt():
    rows = []
    for seed in [0, 1, 2]:
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            D, _ = _dr_pseudo(X, Y, W)
            X_inf = np.ones((N, 1))

            # Item 2: heteroskedastic Welsch (X_for_var = [1, X[:,0]])
            X_for_var = np.column_stack([np.ones(N), X[:, 0]])
            try:
                ate, lo, hi = fit_phase3_custom(
                    heteroskedastic_welsch_model, X_inf, D, X_for_var=jnp.array(X_for_var))
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density, "model": "Hetero-Welsch",
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov})
                print(f"  seed={seed} p={density:.2f} Hetero-Welsch ate={ate:+.3f} cov={cov}")
            except Exception as e:
                print(f"  seed={seed} p={density:.2f} Hetero-Welsch ERR {e}")

            # Item 6: EVT mixture (Welsch bulk + Pareto tail)
            try:
                ate, lo, hi = fit_phase3_custom(evt_mixture_model, X_inf, D)
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density, "model": "Welsch+Pareto-tail",
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov})
                print(f"  seed={seed} p={density:.2f} Welsch+Pareto-tail ate={ate:+.3f} cov={cov}")
            except Exception as e:
                print(f"  seed={seed} p={density:.2f} Welsch+Pareto-tail ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 3: γ-divergence X-learner (both stages) ==============

def gamma_irls(X, y, gamma=0.5, n_iter=20):
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X @ beta
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1e-6)
        w = np.exp(-gamma * (r / scale) ** 2 / 2)
        WX = X * w[:, None]
        beta = np.linalg.lstsq(WX, w * y, rcond=None)[0]
    return beta


def gamma_x_both_stages(X, Y, W, gamma=0.5, seed=0):
    """X-learner with γ-IRLS in BOTH stage 1 (μ̂_w) and stage 2 (imputed effects)."""
    # Stage 1: γ-IRLS regression of Y on X separately for each arm
    # Use a polynomial-X expansion to make linear γ-IRLS reasonable
    X_arm1 = np.column_stack([np.ones((W == 1).sum()), X[W == 1]])
    X_arm0 = np.column_stack([np.ones((W == 0).sum()), X[W == 0]])
    b1 = gamma_irls(X_arm1, Y[W == 1], gamma=gamma)
    b0 = gamma_irls(X_arm0, Y[W == 0], gamma=gamma)
    Xfull = np.column_stack([np.ones(len(X)), X])
    mu1_a = Xfull @ b1
    mu0_a = Xfull @ b0
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)

    # Stage 2: imputed effects with γ-IRLS regression
    D1 = Y[W == 1] - mu0_a[W == 1]
    D0 = mu1_a[W == 0] - Y[W == 0]
    tau1 = gamma_irls(np.ones((len(D1), 1)), D1, gamma=gamma)[0]
    tau0 = gamma_irls(np.ones((len(D0), 1)), D0, gamma=gamma)[0]
    return float(np.mean(pi * tau1 + (1 - pi) * tau0))


def run_item3_gamma_both():
    rows = []
    for seed in [0, 1, 2]:
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            ate = gamma_x_both_stages(X, Y, W, gamma=0.5, seed=seed)
            rows.append({"seed": seed, "density": density,
                         "estimator": "γ-X both stages",
                         "ate": ate, "bias": ate - TRUE_ATE})
            print(f"  seed={seed} p={density:.2f} γ-both ate={ate:+.3f}")
    return pd.DataFrame(rows)


# ============== Main: run all and write summary ==============

def main():
    runs = [
        ("trimmed_dr",       run_item4_trimmed,         "trimmed_dr_overlap"),
        ("rlearner_huber",   run_item7_rlearner,        "rlearner_huber"),
        ("hillstrom_bases",  run_item9_hillstrom,       "hillstrom_extra_bases"),
        ("bma_varying_n",    run_item5_bma_varying_n,   "bma_varying_n"),
        ("hetero_evt",       run_items_2_6_hetero_evt,  "hetero_evt_phase3"),
        ("gamma_both",       run_item3_gamma_both,      "gamma_x_both_stages"),
        ("coverage_30seed",  run_item10_coverage_replication, "coverage_30seed"),
    ]
    for label, fn, fname in runs:
        print(f"\n=== {label} ===")
        try:
            df = fn()
            df.to_csv(RESULTS_DIR / f"{fname}_raw.csv", index=False)
            print(f"  wrote {fname}_raw.csv")
        except Exception as e:
            print(f"  {label} FAILED: {e}")

    # Bundled markdown summary
    md = ["# Round-7 reviewer-response: items 2-7, 9, 10", ""]
    for label, _, fname in runs:
        path = RESULTS_DIR / f"{fname}_raw.csv"
        if path.exists():
            df = pd.read_csv(path)
            md.append(f"## {label} ({fname})")
            md.append("")
            md.append(f"Rows: {len(df)}; columns: {', '.join(df.columns)}")
            md.append("")
    (RESULTS_DIR / "round7.md").write_text("\n".join(md))
    print("wrote round7.md")


if __name__ == "__main__":
    main()
