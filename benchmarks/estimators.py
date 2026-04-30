"""
Unified estimator wrappers for fair Monte Carlo comparison.

Each wrapper returns a standard ``Result`` dict:

    {
      "ate":      float | None,   # point estimate of ATE
      "ci_lo":    float | None,   # 2.5 % posterior / confidence limit
      "ci_hi":    float | None,   # 97.5 % posterior / confidence limit
      "runtime":  float,          # wall-clock seconds
      "error":    str | None,     # exception message on failure
    }

Every non-parametric method is pinned to the same base learner
configuration (``BASE_LEARNER_KIND`` below) so architecture is the only
varying factor across rows in the results table.
"""

from __future__ import annotations
import os
import time
import warnings
import numpy as np
from contextlib import contextmanager
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Parallel MCMC chains — must be set before jax loads
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")


# ═══════════════════════════════════════════════════════════════════════════
# Unified base-learner factory
# ═══════════════════════════════════════════════════════════════════════════

def make_regressor():
    """Shared regressor for ALL methods — so comparisons measure architecture."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=150, max_depth=4, learning_rate=0.1, random_state=42
    )


def make_classifier():
    """Shared classifier for ALL methods."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.1, random_state=42
    )


# Dict form for libraries that need params rather than instances
_XGB_PARAMS = {"n_estimators": 150, "max_depth": 4, "verbosity": 0, "random_state": 42}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def _timed():
    t0 = time.time()
    yield lambda: time.time() - t0


def _result(ate=None, lo=None, hi=None, rt=0.0, err=None):
    return {
        "ate": None if ate is None else float(ate),
        "ci_lo": None if lo is None else float(lo),
        "ci_hi": None if hi is None else float(hi),
        "runtime": float(rt),
        "error": err,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Meta-Learner family — always available (sklearn only)
# ═══════════════════════════════════════════════════════════════════════════

def fit_s_learner(X, Y, W):
    with _timed() as elapsed:
        try:
            XW = np.column_stack([X, W.astype(float)])
            m = make_regressor()
            m.fit(XW, Y)
            X1 = np.column_stack([X, np.ones(len(X))])
            X0 = np.column_stack([X, np.zeros(len(X))])
            ate = float(np.mean(m.predict(X1) - m.predict(X0)))
            return _result(ate=ate, rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_t_learner(X, Y, W):
    with _timed() as elapsed:
        try:
            mu0 = make_regressor()
            mu0.fit(X[W == 0], Y[W == 0])
            mu1 = make_regressor()
            mu1.fit(X[W == 1], Y[W == 1])
            ate = float(np.mean(mu1.predict(X) - mu0.predict(X)))
            return _result(ate=ate, rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_r_learner(X, Y, W):
    """R-Learner (Robinson residualisation, constant ATE via WLS)."""
    with _timed() as elapsed:
        try:
            N = len(X)
            Y_res = np.zeros(N)
            W_res = np.zeros(N)
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            for tr, te in kf.split(X):
                mu = make_regressor(); mu.fit(X[tr], Y[tr])
                Y_res[te] = Y[te] - mu.predict(X[te])
                pi_m = make_classifier(); pi_m.fit(X[tr], W[tr])
                pi_te = np.clip(pi_m.predict_proba(X[te])[:, 1], 0.01, 0.99)
                W_res[te] = W[te] - pi_te
            denom = np.dot(W_res, W_res)
            ate = float(np.dot(W_res, Y_res) / denom) if denom > 1e-8 else float("nan")
            return _result(ate=ate, rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_x_learner(X, Y, W):
    """Standard X-Learner (Künzel et al. 2019)."""
    with _timed() as elapsed:
        try:
            mu0 = make_regressor(); mu0.fit(X[W == 0], Y[W == 0])
            mu1 = make_regressor(); mu1.fit(X[W == 1], Y[W == 1])
            pi_m = make_classifier(); pi_m.fit(X, W)
            pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.01, 0.99)
            D1 = Y[W == 1] - mu0.predict(X[W == 1])
            D0 = mu1.predict(X[W == 0]) - Y[W == 0]
            tau1 = make_regressor(); tau1.fit(X[W == 1], D1)
            tau0 = make_regressor(); tau0.fit(X[W == 0], D0)
            ate = float(np.mean((1 - pi) * tau1.predict(X) + pi * tau0.predict(X)))
            return _result(ate=ate, rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_dr_learner(X, Y, W):
    """DR-Learner (AIPW) with cross-fitted nuisance + normal-approx CI."""
    with _timed() as elapsed:
        try:
            N = len(X)
            pseudo = np.zeros(N)
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            for tr, te in kf.split(X):
                Xtr, Ytr, Wtr = X[tr], Y[tr], W[tr]
                Xte, Wte, Yte = X[te], W[te], Y[te]
                mu0 = make_regressor()
                ctrl = Wtr == 0
                mu0.fit(Xtr[ctrl], Ytr[ctrl]) if ctrl.sum() > 1 else mu0.fit(Xtr, Ytr)
                mu1 = make_regressor(); mu1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
                pi_m = make_classifier(); pi_m.fit(Xtr, Wtr)
                pi = np.clip(pi_m.predict_proba(Xte)[:, 1], 0.01, 0.99)
                mu0_te, mu1_te = mu0.predict(Xte), mu1.predict(Xte)
                pseudo[te] = (
                    mu1_te - mu0_te
                    + Wte / pi * (Yte - mu1_te)
                    - (1 - Wte) / (1 - pi) * (Yte - mu0_te)
                )
            ate = float(np.mean(pseudo))
            se = float(np.std(pseudo, ddof=1) / np.sqrt(N))
            return _result(ate=ate, lo=ate - 1.96 * se, hi=ate + 1.96 * se, rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# RX-Learner (this work)
# ═══════════════════════════════════════════════════════════════════════════

def fit_causal_bart(X, Y, W, draws=400, tune=400, trees=50, chains=2):
    """
    Bayesian T-Learner with PyMC-BART — canonical Bayesian causal baseline.

    μ₀(x), μ₁(x) are each modelled as a BART regression. ATE is posterior
    mean of (μ₁ − μ₀) with 95% HDI as credible interval.
    """
    with _timed() as elapsed:
        try:
            import pymc as pm
            import pymc_bart as pmb
            import arviz as az

            rng = np.random.default_rng(42)

            def _bart_ate(X_sub, Y_sub, Xp):
                with pm.Model():
                    X_shared = pm.Data("X", X_sub.astype(np.float32))
                    sigma = pm.HalfNormal("sigma", 1.0)
                    mu = pmb.BART("mu", X=X_shared, Y=Y_sub.astype(np.float32),
                                  m=trees)
                    pm.Normal("y", mu=mu, sigma=sigma, observed=Y_sub)
                    idata = pm.sample(draws=draws, tune=tune, chains=chains,
                                      random_seed=42, progressbar=False,
                                      compute_convergence_checks=False)
                    pm.set_data({"X": Xp.astype(np.float32)})
                    pp = pm.sample_posterior_predictive(
                        idata, var_names=["mu"], random_seed=42,
                        progressbar=False,
                    )
                return pp.posterior_predictive["mu"].values  # (chains, draws, N)

            treat = W == 1
            ctrl = W == 0
            mu1_samples = _bart_ate(X[treat], Y[treat], X)
            mu0_samples = _bart_ate(X[ctrl], Y[ctrl], X)

            cate = mu1_samples - mu0_samples                 # (chains, draws, N)
            ate_samples = cate.mean(axis=-1).flatten()       # (chains*draws,)
            ate = float(ate_samples.mean())
            lo, hi = np.percentile(ate_samples, [2.5, 97.5])
            return _result(ate=ate, lo=lo, hi=hi, rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_rx_learner(X, Y, W, robust=False, use_student_t=False,
                   use_overlap=False, num_warmup=400, num_samples=800, num_chains=2):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    with _timed() as elapsed:
        try:
            model = TargetedBayesianXLearner(
                outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                n_splits=2,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                robust=robust,
                c_whale=1.34,
                use_student_t=use_student_t,
                use_overlap=use_overlap,
                random_state=42,
            )
            model.fit(X, Y, W)
            cate, lo, hi = model.predict()
            return _result(ate=cate[0], lo=lo[0], hi=hi[0], rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Optional external libraries
# ═══════════════════════════════════════════════════════════════════════════

def fit_doubleml_irm(X, Y, W):
    with _timed() as elapsed:
        try:
            import doubleml as dml
            data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=W.astype(float))
            irm = dml.DoubleMLIRM(
                data,
                ml_g=make_regressor(),
                ml_m=make_classifier(),
                n_folds=2, n_rep=1, score="ATE",
            )
            irm.fit()
            ci = irm.confint()
            return _result(
                ate=float(irm.coef[0]),
                lo=float(ci.iloc[0, 0]),
                hi=float(ci.iloc[0, 1]),
                rt=elapsed(),
            )
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_econml_forest(X, Y, W):
    with _timed() as elapsed:
        try:
            from econml.dml import CausalForestDML
            est = CausalForestDML(
                model_y=make_regressor(),
                model_t=make_classifier(),
                discrete_treatment=True,
                cv=2, n_estimators=200, random_state=42,
            )
            est.fit(Y, W, X=X)
            ate = float(est.ate(X))
            lb, ub = est.ate_interval(X, alpha=0.05)
            return _result(ate=ate, lo=float(lb), hi=float(ub), rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


def fit_causalml_xlearner(X, Y, W):
    with _timed() as elapsed:
        try:
            from causalml.inference.meta import BaseXRegressor
            learner = BaseXRegressor(
                learner=make_regressor(),
                control_name=0,
            )
            te = learner.fit_predict(X=X, treatment=W, y=Y)
            return _result(ate=float(np.mean(te)), rt=elapsed())
        except Exception as e:
            return _result(rt=elapsed(), err=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Registry (ordered for table display)
# ═══════════════════════════════════════════════════════════════════════════

ESTIMATORS = {
    "S-Learner":           fit_s_learner,
    "T-Learner":           fit_t_learner,
    "R-Learner":           fit_r_learner,
    "X-Learner (std)":     fit_x_learner,
    "DR-Learner (AIPW)":   fit_dr_learner,
    "RX-Learner (std)":    lambda X, Y, W: fit_rx_learner(X, Y, W, robust=False),
    "RX-Learner (robust)": lambda X, Y, W: fit_rx_learner(X, Y, W, robust=True, use_student_t=True),
    "RX-Learner (robust+overlap)": lambda X, Y, W: fit_rx_learner(
        X, Y, W, robust=True, use_student_t=True, use_overlap=True),
    "DoubleML IRM":        fit_doubleml_irm,
    "EconML Forest":       fit_econml_forest,
    "CausalML X-Learner":  fit_causalml_xlearner,
    "Causal BART (T-Learner)": fit_causal_bart,
}
