"""Round-12 reviewer-response experiments — items 2, 3, 5, 6, 7, 11.

Items addressed:
  2. Hill-plot of Hillstrom tail heaviness + coverage by tail-index quintile.
  3. Bimodal-contamination DGP stress test.
  5. Quantile-DR / CSQTE-style baseline at q=0.75 on whale data.
  6. Horseshoe-prior spline basis (vs ridge from round 5).
  7. N=10k, p=100 runtime + memory profile.
 11. RBCI ω-tuning vs trace-η side-by-side comparison.

Usage: python -u -m benchmarks.run_round12_experiments
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import HuberRegressor

from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figures"
N = 1000
TRUE_ATE = 2.0


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


# ============== Item 2: Hill plot + coverage by tail-index quintile ==============

def hill_estimator(x_abs, k):
    """Hill estimator using the top-k order statistics."""
    sorted_x = np.sort(x_abs)[::-1]
    if k >= len(sorted_x) or sorted_x[k] <= 0:
        return float("nan")
    return float(1.0 / np.mean(np.log(sorted_x[:k] / sorted_x[k])))


def run_item2_hillstrom_hill():
    from benchmarks.run_hillstrom import load_hillstrom
    X, Y, W = load_hillstrom()
    Y_pos = Y[Y > 0]  # spend > 0 only (zero-spend customers don't have a tail)
    print(f"  N_total={len(Y)}, N_with_spend={len(Y_pos)}")
    # Hill plot: alpha as a function of k (number of upper-order stats)
    k_grid = np.unique(np.round(np.geomspace(5, len(Y_pos) // 2, 20)).astype(int))
    alphas = [hill_estimator(Y_pos, int(k)) for k in k_grid]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(k_grid, alphas, marker="o", lw=1.5)
    ax.axhline(2.0, ls="--", color="grey", alpha=0.5,
               label="α=2 (finite variance threshold)")
    ax.set_xscale("log")
    ax.set_xlabel("k (top order statistics)")
    ax.set_ylabel(r"Hill estimate $\hat\alpha$")
    ax.set_title(f"Hill plot, Hillstrom positive spend (N={len(Y_pos)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_hillstrom_hill.pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote fig7_hillstrom_hill.pdf, alpha at k=median={alphas[len(k_grid)//2]:.2f}")
    return pd.DataFrame({"k": k_grid, "alpha_hat": alphas})


# ============== Item 3: Bimodal contamination DGP ==============

def bimodal_whale_dgp(N, density, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, 5))
    pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
    W = rng.binomial(1, pi)
    Y0 = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, N)
    Y1 = Y0 + TRUE_ATE
    Y = np.where(W == 1, Y1, Y0)
    n_w = int(round(density * N))
    if n_w > 0:
        idx = rng.choice(N, size=n_w, replace=False)
        # Bimodal: half whales shift up by +5000, half down by -5000
        signs = np.where(rng.uniform(size=n_w) < 0.5, +1.0, -1.0)
        Y[idx] += 5000.0 * signs
    return X, Y, W


def run_item3_bimodal():
    rows = []
    for seed in range(3):
        for density in [0.05, 0.20]:
            X, Y, W = bimodal_whale_dgp(N, density, seed)
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
                print(f"  s={seed} p={density:.2f} sev={severity:7s} ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 5: Quantile-DR at q=0.75 on whale data ==============

def run_item5_csqte_q75():
    from sklearn.linear_model import QuantileRegressor
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
            mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
            pi_m = _make_clf(); pi_m.fit(X, W)
            pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
            mu0_a = mu0.predict(X); mu1_a = mu1.predict(X)
            D = np.where(W == 1,
                         mu1_a - mu0_a + (Y - mu1_a) / pi,
                         mu1_a - mu0_a - (Y - mu0_a) / (1 - pi))
            qreg = QuantileRegressor(quantile=0.75, alpha=0.0, solver="highs")
            qreg.fit(np.ones((N, 1)), D)
            qte = float(qreg.intercept_)
            rows.append({"seed": seed, "density": density,
                         "csqte_q75_estimate": qte,
                         "true_ate": TRUE_ATE,
                         "diff_from_ate": qte - TRUE_ATE})
            print(f"  s={seed} p={density:.2f} CSQTE_q75={qte:+.3f} (true ATE 2)")
    return pd.DataFrame(rows)


# ============== Item 6: Horseshoe-prior spline basis ==============

def run_item6_horseshoe_spline():
    """Welsch posterior with horseshoe prior on spline coefficients."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import jax.random as random
    from sert_xlearner.models.nuisance import NuisanceEstimator
    from sert_xlearner.core.orthogonalization import impute_and_debias

    def horseshoe_welsch_model(X_inf, D, c=1.34, prior_scale=1.0):
        n, p = X_inf.shape
        tau_global = numpyro.sample("tau_global", dist.HalfCauchy(1.0))
        lam = numpyro.sample("lam", dist.HalfCauchy(1.0).expand([p]))
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0, 1.0).expand([p]))
        beta = numpyro.deterministic("beta", beta_raw * lam * tau_global)
        tau = jnp.dot(X_inf, beta)
        r = D - tau
        log_lik = -(c**2 / 2.0) * (1.0 - jnp.exp(-(r / c) ** 2))
        numpyro.factor("welsch", jnp.sum(log_lik))

    def spline_basis(x, knots):
        cols = [np.ones(len(x)), x, x * x, x * x * x]
        for k in knots:
            cols.append(np.maximum(0, x - k) ** 3)
        return np.column_stack(cols)

    TAU_BULK, TAU_TAIL, WHALE_CUT = 2.0, 10.0, 1.96
    knots = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (N, 5))
        whale = (np.abs(X[:, 0]) > WHALE_CUT).astype(float)
        tau = TAU_BULK + (TAU_TAIL - TAU_BULK) * whale
        Y0 = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, N)
        Y1 = Y0 + tau
        pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
        W = rng.binomial(1, pi)
        Y = np.where(W == 1, Y1, Y0)
        nuis = NuisanceEstimator(
            {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            n_splits=2, random_state=seed, method="xgboost",
        )
        mu0, mu1, pi_pred = nuis.fit_predict(X, Y, W)
        treated_mask, _, D1, D0, *_ = impute_and_debias(
            Y, W, mu0, mu1, pi_pred, robust=False, use_overlap=False)
        D = np.concatenate([D1, D0])
        X_arr = np.concatenate([X[treated_mask, 0], X[~treated_mask, 0]])
        X_inf = spline_basis(X_arr, knots)
        kernel = NUTS(horseshoe_welsch_model)
        mcmc = MCMC(kernel, num_warmup=400, num_samples=600, num_chains=2,
                    progress_bar=False)
        try:
            mcmc.run(random.PRNGKey(seed), jnp.array(X_inf), jnp.array(D))
            beta = np.asarray(mcmc.get_samples()["beta"])
            # Predicted CATE on full population
            X_full = spline_basis(X[:, 0], knots)
            cate = X_full @ beta.mean(axis=0)
            pehe = float(np.sqrt(np.mean((cate - tau) ** 2)))
            tau_whale = float(np.mean(cate[whale.astype(bool)]))
            tau_bulk = float(np.mean(cate[~whale.astype(bool)]))
            rows.append({"seed": seed, "pehe": pehe,
                         "tau_hat_whale": tau_whale, "tau_hat_bulk": tau_bulk})
            print(f"  s={seed} horseshoe-spline PEHE={pehe:.3f} τ̂_whale={tau_whale:.2f} τ̂_bulk={tau_bulk:.2f}")
        except Exception as e:
            print(f"  s={seed} ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 7: N=10k, p=100 runtime+memory ==============

def run_item7_scaling_large():
    rng = np.random.default_rng(0)
    rows = []
    for (n_size, p_dim) in [(2000, 50), (5000, 50), (10000, 50), (10000, 100)]:
        n_w = int(0.05 * n_size)
        X = rng.normal(0, 1, (n_size, max(5, p_dim)))
        pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
        W = rng.binomial(1, pi)
        Y0 = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, n_size)
        Y1 = Y0 + 2.0
        Y = np.where(W == 1, Y1, Y0)
        idx = rng.choice(n_size, size=n_w, replace=False)
        Y[idx] += 5000.0 * np.sign(rng.normal(size=n_w))
        if p_dim == 1:
            X_inf = np.ones((n_size, 1))
        else:
            X_inf = np.column_stack([np.ones(n_size), rng.normal(0, 1, (n_size, p_dim - 1))])
        prior_scale = 2.0
        kwargs = dict(
            n_splits=2, num_warmup=200, num_samples=400, num_chains=2,
            c_whale=1.34, mad_rescale=False, random_state=0,
            robust=True, use_student_t=True, prior_scale=prior_scale,
            contamination_severity="severe",
        )
        t0 = time.time()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            model = TargetedBayesianXLearner(**kwargs)
            model.fit(X, Y, W, X_infer=X_inf)
            rt = time.time() - t0
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rows.append({"N": n_size, "p": p_dim, "runtime": rt,
                         "peak_rss_kb_delta": int(rss_after - rss_before)})
            print(f"  N={n_size} p={p_dim} runtime={rt:.1f}s rss_delta={(rss_after-rss_before)/1024:.1f} MB")
        except Exception as e:
            print(f"  N={n_size} p={p_dim} ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 11: RBCI ω vs trace-η side-by-side ==============

def run_item11_rbci_vs_trace():
    rows = []
    for seed in range(3):
        for density in [0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            # Bootstrap distribution of Huber-DR ATE estimates
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(50):
                idx = rng.integers(0, N, size=N)
                try:
                    mu0 = _make_reg(); mu0.fit(X[idx][W[idx] == 0], Y[idx][W[idx] == 0])
                    mu1 = _make_reg(); mu1.fit(X[idx][W[idx] == 1], Y[idx][W[idx] == 1])
                    pi_m = _make_clf(); pi_m.fit(X[idx], W[idx])
                    pi = np.clip(pi_m.predict_proba(X[idx])[:, 1], 0.05, 0.95)
                    D = np.where(W[idx] == 1,
                                 mu1.predict(X[idx]) - mu0.predict(X[idx]) +
                                 (Y[idx] - mu1.predict(X[idx])) / pi,
                                 mu1.predict(X[idx]) - mu0.predict(X[idx]) -
                                 (Y[idx] - mu0.predict(X[idx])) / (1 - pi))
                    r = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
                    r.fit(np.ones((N, 1)), D)
                    boots.append(r.coef_[0])
                except Exception:
                    pass
            boot_arr = np.array(boots)

            for omega in [0.5, 1.0, 2.0]:
                c_eta = 1.34 / np.sqrt(omega)
                kwargs = dict(
                    n_splits=2, num_warmup=300, num_samples=500, num_chains=2,
                    c_whale=c_eta, mad_rescale=False, random_state=seed,
                    robust=True, use_student_t=True,
                    contamination_severity="severe",
                )
                model = TargetedBayesianXLearner(**kwargs)
                model.fit(X, Y, W, X_infer=np.ones((N, 1)))
                beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
                lo, hi = np.percentile(beta, [2.5, 97.5])
                # Interval score on bootstrap pseudo-truths
                int_score = float(np.mean((hi - lo) +
                                          (2 / 0.05) * (np.maximum(0, lo - boot_arr) +
                                                        np.maximum(0, boot_arr - hi))))
                cov_truth = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density,
                             "omega": omega,
                             "ate": float(np.mean(beta)),
                             "ci_width": hi - lo,
                             "interval_score": int_score,
                             "cov_truth": cov_truth})
                print(f"  s={seed} p={density:.2f} ω={omega} "
                      f"ate={np.mean(beta):+.3f} w={hi-lo:.3f} score={int_score:.2f}")
    return pd.DataFrame(rows)


def main():
    runs = [
        ("hillstrom_hill",   run_item2_hillstrom_hill,    "hillstrom_hill"),
        ("bimodal",          run_item3_bimodal,            "bimodal_contamination"),
        ("csqte_q75",        run_item5_csqte_q75,          "csqte_q75"),
        ("horseshoe_spline", run_item6_horseshoe_spline,  "horseshoe_spline"),
        ("scaling_large",    run_item7_scaling_large,     "scaling_large"),
        ("rbci_vs_trace",    run_item11_rbci_vs_trace,    "rbci_vs_trace"),
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
    print("DONE_ROUND12")


if __name__ == "__main__":
    main()
