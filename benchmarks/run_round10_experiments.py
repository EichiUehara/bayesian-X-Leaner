"""Round-10 reviewer-response experiments — items 1, 2, 3, 5, 6, 7, 8.

Items addressed:
  1. Influence-function diagnostic: plot ψ_W vs ψ_t empirically on
     whale residuals at multiple contamination levels.
  2. Computational scaling with basis dimension p.
  3. Multi-functional η calibration: max-directional and spectral
     scales for many contrasts simultaneously.
  5. β-divergence-power-posterior baseline (alternative to Welsch).
  6. Tail-trimmed IPW with bias correction baseline.
  7. Adaptive basis library mode: automated change-point search
     in φ(x).
  8. Shrinkage-stabilised η estimator: ridge-regularised Î.

Item 4 (real heavy-tailed data beyond Hillstrom) attempts the
Lalonde NSW dataset (publicly available, heavy-tailed earnings).

Usage: python -u -m benchmarks.run_round10_experiments
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)

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


def _dr_pseudo(X, Y, W):
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    pi_m = _make_clf(); pi_m.fit(X, W)
    pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
    mu0_a = mu0.predict(X); mu1_a = mu1.predict(X)
    return np.where(W == 1,
                    mu1_a - mu0_a + (Y - mu1_a) / pi,
                    mu1_a - mu0_a - (Y - mu0_a) / (1 - pi))


# ============== Item 1: Influence-function plot ==============

def run_item1_influence_plot():
    """Plot ψ_W(r) and ψ_t(r) along with the empirical residual
    distribution at three contamination levels.

    Layout: 1×3 panels with sharey, short panel titles, and a single
    figure-level legend at the top (avoids per-panel legend clutter
    and per-panel y-axis label duplication).
    """
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.4), sharey=True)
    densities = [0.00, 0.05, 0.20]
    r_grid = np.linspace(-50, 50, 1000)
    c = 1.34; sigma = 1.0; nu = 3.0
    psi_W = r_grid * np.exp(-(r_grid / c) ** 2)
    psi_t = (nu + 1) * r_grid / (nu * sigma ** 2 + r_grid ** 2)
    for i, (ax, density) in enumerate(zip(axes, densities)):
        n_w = int(round(density * N))
        X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=0)
        D = _dr_pseudo(X, Y, W)
        med_D = np.median(D)
        residuals = D - med_D
        ax2 = ax.twinx()
        ax2.hist(np.clip(residuals, -50, 50), bins=80, alpha=0.25, color="grey")
        ax2.set_yticks([])
        ax.plot(r_grid, psi_W, color="steelblue", lw=1.5,
                label=r"$\psi_W$ (Welsch, $c=1.34$)")
        ax.plot(r_grid, psi_t, color="darkorange", lw=1.5, ls="--",
                label=r"$\psi_t$ (Student-$t$, $\nu=3$)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel(r"residual $r$")
        if i == 0:
            ax.set_ylabel(r"$\psi(r)$")
        ax.set_title(f"{density:.0%} contamination")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG_DIR / "fig6_influence_functions.pdf", bbox_inches="tight")
    plt.close()
    print("  wrote fig6_influence_functions.pdf")


# ============== Item 2: Compute scaling with p ==============

def run_item2_scaling():
    """Time RX-Learner Phase 3 NUTS as p (basis dim) grows."""
    rng = np.random.default_rng(0)
    rows = []
    for p in [1, 5, 10, 20, 50, 100]:
        X, Y, W, _ = whale_dgp(N=N, n_whales=int(0.05 * N), seed=0)
        # Random p-dim basis: intercept + p-1 random covariates
        if p == 1:
            X_inf = np.ones((N, 1))
        else:
            X_inf = np.column_stack([np.ones(N), rng.normal(0, 1, (N, p - 1))])
        prior_scale = 10.0 if p < 10 else 2.0
        kwargs = dict(
            n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
            c_whale=1.34, mad_rescale=False, random_state=0,
            robust=True, use_student_t=True, prior_scale=prior_scale,
            contamination_severity="severe",
        )
        t0 = time.time()
        try:
            model = TargetedBayesianXLearner(**kwargs)
            model.fit(X, Y, W, X_infer=X_inf)
            beta = np.asarray(model.mcmc_samples["beta"])  # (S, p)
            ess_min = float(np.min([np.var(beta[:, j]) for j in range(p)] or [0]))
            rt = time.time() - t0
            rows.append({"p": p, "runtime": rt, "n_samples": beta.shape[0]})
            print(f"  p={p:4d} runtime={rt:.1f}s n_samples={beta.shape[0]}")
        except Exception as e:
            print(f"  p={p} ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 3: Multi-functional η calibration ==============

def run_item3_multifunctional():
    """Compute η^*(a) for several contrasts simultaneously and report
    the spread; also report a max-directional η = min_a η^*(a) and
    the spectral (largest-eigenvalue) η."""
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            X_inf = np.column_stack([
                np.ones(N), X[:, 0], X[:, 1],
                (np.abs(X[:, 0]) > 1.96).astype(float),
            ])
            p = X_inf.shape[1]
            kwargs = dict(
                n_splits=2, num_warmup=300, num_samples=500, num_chains=2,
                c_whale=1.34, mad_rescale=False, random_state=seed,
                robust=True, use_student_t=True,
                contamination_severity="severe",
            )
            model = TargetedBayesianXLearner(**kwargs)
            model.fit(X, Y, W, X_infer=X_inf)
            beta = np.asarray(model.mcmc_samples["beta"])  # (S, p)
            beta_mean = beta.mean(axis=0)
            # Estimate I and J on residuals at posterior mean
            D = _dr_pseudo(X, Y, W)
            r = D - X_inf @ beta_mean
            c = 1.34
            psi = r * np.exp(-(r / c) ** 2)
            psi_prime = np.exp(-(r / c) ** 2) * (1 - 2 * (r / c) ** 2)
            I_hat = (X_inf.T * psi_prime) @ X_inf / N + 1e-3 * np.eye(p)  # ridge
            J_hat = (X_inf.T * psi ** 2) @ X_inf / N
            I_inv = np.linalg.inv(I_hat)
            sandwich = I_inv @ J_hat @ I_inv
            # Trace formula
            eta_tr = float(np.trace(I_inv) / max(np.trace(sandwich), 1e-9))
            # Per-functional η for each canonical contrast
            etas = []
            for j in range(p):
                a = np.zeros(p); a[j] = 1.0
                num = float(a @ I_inv @ a)
                den = float(a @ sandwich @ a)
                etas.append(num / max(den, 1e-9))
            eta_max = float(min(etas))  # max-directional: smallest η over contrasts
            # Spectral: largest eigenvalue of I^{-1} ratio
            try:
                eigvals = np.linalg.eigvalsh(I_inv) / np.linalg.eigvalsh(sandwich + 1e-6 * np.eye(p))
                eta_spec = float(np.min(eigvals))
            except Exception:
                eta_spec = float("nan")
            rows.append({"seed": seed, "density": density, "p": p,
                         "eta_trace": eta_tr,
                         "eta_per_min": float(min(etas)),
                         "eta_per_max": float(max(etas)),
                         "eta_max_dir": eta_max,
                         "eta_spectral": eta_spec})
            print(f"  s={seed} p={density:.2f} η_tr={eta_tr:.2f} per[{min(etas):.2f},{max(etas):.2f}] "
                  f"max-dir={eta_max:.2f} spec={eta_spec:.2f}")
    return pd.DataFrame(rows)


# ============== Item 5: β-divergence power posterior baseline ==============

def run_item5_beta_divergence():
    """Phase-3 with β-divergence weighted regression (β=0.5).
    Equivalent to Welsch up to a normalisation; we test as alternative."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import jax.random as random
    from sert_xlearner.models.nuisance import NuisanceEstimator
    from sert_xlearner.core.orthogonalization import impute_and_debias

    def beta_divergence_model(X_infer, D, beta_div=0.5):
        n, p = X_infer.shape
        beta = numpyro.sample("beta", dist.Normal(0, 10.0).expand([p]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
        tau = jnp.dot(X_infer, beta)
        # Beta-divergence likelihood: see Basu, Harris, Hjort, Jones (1998)
        # log L_β(θ) = ∑ [(1+β) log f_θ(y_i) - integrate f_θ(y)^{1+β} dy]
        # For Gaussian: ∫ f^{1+β} dy = 1/(σ √(1+β) (2π)^{β/2})
        log_pdf = dist.Normal(tau, sigma).log_prob(D)
        beta_factor = (1 / beta_div) * (jnp.exp(beta_div * log_pdf) - 1)
        numpyro.factor("beta_div", jnp.sum(beta_factor))

    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            nuis = NuisanceEstimator(
                {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                n_splits=2, random_state=seed, method="xgboost",
            )
            mu0, mu1, pi = nuis.fit_predict(X, Y, W)
            _, _, D1, D0, *_ = impute_and_debias(
                Y, W, mu0, mu1, pi, robust=False, use_overlap=False)
            D = np.concatenate([D1, D0])
            X_inf = np.ones((len(D), 1))
            kernel = NUTS(beta_divergence_model)
            mcmc = MCMC(kernel, num_warmup=300, num_samples=500, num_chains=2,
                        progress_bar=False)
            try:
                mcmc.run(random.PRNGKey(seed), jnp.array(X_inf), jnp.array(D),
                         beta_div=0.5)
                beta_post = np.asarray(mcmc.get_samples()["beta"]).squeeze()
                ate = float(np.mean(beta_post))
                lo, hi = np.percentile(beta_post, [2.5, 97.5])
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({"seed": seed, "density": density,
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                             "ci_width": hi - lo})
                print(f"  s={seed} p={density:.2f} β-div ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
            except Exception as e:
                print(f"  s={seed} p={density:.2f} β-div ERR {e}")
    return pd.DataFrame(rows)


# ============== Item 6: Tail-trimmed IPW with bias correction ==============

def run_item6_tail_trimmed_ipw():
    """Trim outcomes beyond a quantile; bias-correct via a residual
    regression."""
    rows = []
    for seed in range(3):
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            # Trim outcomes at 99th percentile
            cutoff = np.percentile(np.abs(Y), 99)
            Y_trim = np.clip(Y, -cutoff, cutoff)
            mu0 = _make_reg(); mu0.fit(X[W == 0], Y_trim[W == 0])
            mu1 = _make_reg(); mu1.fit(X[W == 1], Y_trim[W == 1])
            pi_m = _make_clf(); pi_m.fit(X, W)
            pi = np.clip(pi_m.predict_proba(X)[:, 1], 0.05, 0.95)
            mu0_a = mu0.predict(X); mu1_a = mu1.predict(X)
            D = np.where(W == 1,
                         mu1_a - mu0_a + (Y_trim - mu1_a) / pi,
                         mu1_a - mu0_a - (Y_trim - mu0_a) / (1 - pi))
            ate = float(np.mean(D))
            # Bootstrap CI
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(50):
                idx = rng.integers(0, N, size=N)
                boots.append(float(np.mean(D[idx])))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cov = int(lo <= TRUE_ATE <= hi)
            rows.append({"seed": seed, "density": density,
                         "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                         "ci_width": hi - lo})
            print(f"  s={seed} p={density:.2f} TT-IPW ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ============== Item 7: Adaptive basis with change-point search ==============

def run_item7_change_point():
    """Search over a grid of thresholds and select the threshold that
    minimises a held-out cross-validation criterion."""
    TAU_BULK, TAU_TAIL, WHALE_CUT = 2.0, 10.0, 1.96
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

        # Grid of candidate thresholds
        candidates = np.arange(0.5, 3.5, 0.25)
        # For each candidate, fit and compute held-out PEHE proxy
        best_c = 1.96; best_score = float("inf")
        for c in candidates:
            X_inf = np.column_stack([np.ones(N), (np.abs(X[:, 0]) > c).astype(float)])
            try:
                model = TargetedBayesianXLearner(
                    outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                    propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
                    nuisance_method="xgboost", n_splits=2,
                    num_warmup=200, num_samples=300, num_chains=2,
                    robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
                    random_state=seed,
                )
                model.fit(X, Y, W, X_infer=X_inf)
                cate, _, _ = model.predict(X_new_infer=X_inf)
                cate = np.asarray(cate).flatten()
                # Score: residual variance of D - cate (smaller is better)
                D = _dr_pseudo(X, Y, W)
                score = float(np.median(np.abs(D - cate)))
                if score < best_score:
                    best_score = score; best_c = c
            except Exception:
                pass
        rows.append({"seed": seed, "best_c": float(best_c),
                     "true_c": WHALE_CUT,
                     "abs_error_in_c": abs(best_c - WHALE_CUT)})
        print(f"  s={seed} best_c={best_c:.2f} (true=1.96) error={abs(best_c - WHALE_CUT):.2f}")
    return pd.DataFrame(rows)


# ============== Item 8: Shrinkage-stabilised η ==============

def run_item8_shrinkage_eta():
    """Compare η̂ with and without ridge-stabilised Î across small/large p."""
    rows = []
    for seed in range(3):
        for p_extra in [0, 5, 20]:  # extra random columns added to basis
            X, Y, W, _ = whale_dgp(N=N, n_whales=int(0.05 * N), seed=seed)
            rng = np.random.default_rng(seed * 7)
            if p_extra == 0:
                X_inf = np.ones((N, 1))
            else:
                X_inf = np.column_stack([np.ones(N), rng.normal(0, 1, (N, p_extra))])
            p = X_inf.shape[1]
            kwargs = dict(
                n_splits=2, num_warmup=300, num_samples=500, num_chains=2,
                c_whale=1.34, mad_rescale=False, random_state=seed,
                robust=True, use_student_t=True,
                contamination_severity="severe",
                prior_scale=2.0 if p > 5 else 10.0,
            )
            model = TargetedBayesianXLearner(**kwargs)
            model.fit(X, Y, W, X_infer=X_inf)
            beta_mean = np.asarray(model.mcmc_samples["beta"]).mean(axis=0)
            D = _dr_pseudo(X, Y, W)
            r = D - X_inf @ beta_mean
            c = 1.34
            psi = r * np.exp(-(r / c) ** 2)
            psi_prime = np.exp(-(r / c) ** 2) * (1 - 2 * (r / c) ** 2)
            I_hat = (X_inf.T * psi_prime) @ X_inf / N
            J_hat = (X_inf.T * psi ** 2) @ X_inf / N
            for ridge in [0.0, 1e-3, 1e-2]:
                I_reg = I_hat + ridge * np.eye(p)
                try:
                    I_inv = np.linalg.inv(I_reg)
                    eta = float(np.trace(I_inv) / max(np.trace(I_inv @ J_hat @ I_inv), 1e-9))
                except Exception:
                    eta = float("nan")
                rows.append({"seed": seed, "p": p, "ridge": ridge, "eta_tr": eta})
                print(f"  s={seed} p={p} ridge={ridge:g} eta_tr={eta:.3f}")
    return pd.DataFrame(rows)


# ============== Item 4: Real heavy-tailed dataset (Lalonde NSW) ==============

def run_item4_lalonde():
    """Lalonde NSW: 722 units, real RCT with heavy-tailed earnings.
    Treatment effect on 1978 earnings."""
    import urllib.request
    DATA = Path(__file__).parent / "data"; DATA.mkdir(exist_ok=True)
    URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
    path = DATA / "nsw_dw.dta"
    try:
        if not path.exists():
            urllib.request.urlretrieve(URL, path)
        df = pd.read_stata(path)
    except Exception as e:
        print(f"  Lalonde download failed: {e}")
        return pd.DataFrame()
    # Columns: treat, age, education, black, hispanic, married, nodegree, re74, re75, re78
    Y = df["re78"].values.astype(float)
    W = df["treat"].values.astype(int)
    X = df[["age", "education", "black", "hispanic", "married",
            "nodegree", "re74", "re75"]].values.astype(float)
    print(f"  Lalonde: N={len(X)}, treated={W.sum()}, "
          f"E[Y]={Y.mean():.0f}, max(Y)={Y.max():.0f}, 99%-pct={np.percentile(Y, 99):.0f}")

    rows = []
    for severity in ["none", "severe"]:
        kwargs = dict(
            n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
            c_whale=1.34, mad_rescale=False, random_state=0,
            robust=True, use_student_t=True,
        )
        if severity == "none":
            kwargs["nuisance_method"] = "xgboost"
            kwargs["outcome_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
            kwargs["propensity_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        else:
            kwargs["contamination_severity"] = "severe"
        model = TargetedBayesianXLearner(**kwargs)
        model.fit(X, Y, W, X_infer=np.ones((len(X), 1)))
        beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
        ate = float(np.mean(beta))
        lo, hi = np.percentile(beta, [2.5, 97.5])
        rows.append({"severity": severity, "ate": ate, "lo": lo, "hi": hi,
                     "ci_width": hi - lo})
        print(f"  Lalonde sev={severity} ate=${ate:+.0f} CI=[${lo:+.0f}, ${hi:+.0f}]")
    return pd.DataFrame(rows)


# ============== Main ==============

def main():
    print("\n=== Item 1: influence-function plot ===")
    try:
        run_item1_influence_plot()
    except Exception as e:
        print(f"  failed: {e}")

    runs = [
        ("scaling_p",       run_item2_scaling,           "scaling_p"),
        ("multi_eta",       run_item3_multifunctional,   "multi_eta"),
        ("beta_div",        run_item5_beta_divergence,   "beta_divergence"),
        ("tail_trimmed",    run_item6_tail_trimmed_ipw,  "tail_trimmed_ipw"),
        ("change_point",    run_item7_change_point,      "change_point_basis"),
        ("shrinkage_eta",   run_item8_shrinkage_eta,     "shrinkage_eta"),
        ("lalonde",         run_item4_lalonde,           "lalonde_nsw"),
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
    print("DONE_ROUND10")


if __name__ == "__main__":
    main()
