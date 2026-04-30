"""Round-5 reviewer-response experiments — bundled.

Items 1, 4, 5, 7 from the round-5 review map (see paper/notes if any):

  1. Arm-specific contamination (whales only on the treated arm)
     vs symmetric (default).
  4. Continuous τ(x) CATE coverage (smooth heterogeneity, not step).
  5. Contaminated-normal Phase-3 likelihood as alternative to Welsch
     and Student-t.
  7. Spline-basis + horseshoe-prior CATE for tails-as-signal.

Items 2 (Hillstrom holdout), 3 (policy-risk), 6 (joint η+c) are in
separate scripts to keep run times manageable.

Usage: python -u -m benchmarks.run_round5_experiments --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random

from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from sert_xlearner.models.nuisance import NuisanceEstimator
from sert_xlearner.core.orthogonalization import impute_and_debias


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TAU_BULK = 2.0
TAU_TAIL = 10.0
WHALE_CUT = 1.96


# ---------- Item 1: arm-specific whale DGP ----------

def whale_arm_specific(N, density, seed, treated_only=True, contaminated_arm_high_pi=False):
    """Whales only on the treated arm (or with covariate-dependent
    propensity) instead of symmetric contamination.
    """
    rng = np.random.default_rng(seed)
    P = 8
    X = rng.normal(0, 1, (N, P))
    if contaminated_arm_high_pi:
        pi = np.clip(1.0 / (1.0 + np.exp(-2.0 * X[:, 0])), 0.05, 0.95)
    else:
        pi = np.clip(1.0 / (1.0 + np.exp(-X[:, 0])), 0.15, 0.85)
    W = rng.binomial(1, pi)
    Y0 = 1.5 * X[:, 0] + rng.normal(0, 0.5, N)
    Y1 = Y0 + 2.0
    Y = np.where(W == 1, Y1, Y0)
    n_w = int(round(density * N))
    if n_w > 0:
        if treated_only:
            treated_idx = np.where(W == 1)[0]
            n_w = min(n_w, len(treated_idx))
            idx = rng.choice(treated_idx, size=n_w, replace=False)
        else:
            idx = rng.choice(N, size=n_w, replace=False)
        Y[idx] += 5000.0 * np.sign(rng.normal(size=n_w))
    return X, Y, W, 2.0


def fit_rx(X, Y, W, severity, seed, basis="intercept"):
    n = len(X)
    if basis == "intercept":
        X_inf = np.ones((n, 1))
    elif basis == "tail":
        w_col = (np.abs(X[:, 0]) > 1.96).astype(float).reshape(-1, 1)
        X_inf = np.hstack([np.ones((n, 1)), w_col])
    elif basis == "linear":
        X_inf = np.column_stack([np.ones(n), X[:, 0]])
    else:
        raise ValueError(basis)
    kwargs = dict(
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
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
    model.fit(X, Y, W, X_infer=X_inf)
    cate, lo, hi = model.predict(X_new_infer=X_inf)
    return (np.asarray(cate).flatten(),
            np.asarray(lo).flatten(),
            np.asarray(hi).flatten())


def run_arm_specific(seeds):
    rows = []
    for seed in seeds:
        for density in [0.05, 0.20]:
            for treated_only, label in [(False, "symmetric"), (True, "treated_only")]:
                X, Y, W, true_ate = whale_arm_specific(
                    1000, density, seed, treated_only=treated_only)
                cate, lo, hi = fit_rx(X, Y, W, "severe", seed, "intercept")
                ate = float(np.mean(cate))
                lo_m, hi_m = float(np.mean(lo)), float(np.mean(hi))
                cov = int(lo_m <= true_ate <= hi_m)
                rows.append({"seed": seed, "density": density,
                             "contamination": label, "ate": ate,
                             "lo": lo_m, "hi": hi_m, "cov": cov,
                             "ci_width": hi_m - lo_m})
                print(f"  s={seed} p={density:.2f} {label:14s} "
                      f"ate={ate:+.3f} cov={cov} w={hi_m-lo_m:.3f}")
    return pd.DataFrame(rows)


# ---------- Item 4: continuous τ(x) CATE coverage ----------

def continuous_tau_dgp(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, 5))
    # Smooth τ(x) = 2 + 3 * sigmoid(2*x_0) — non-step, monotone heterogeneity
    tau = 2.0 + 3.0 / (1.0 + np.exp(-2.0 * X[:, 0]))
    eps = rng.normal(0, 1, N)
    Y0 = X[:, 0] + 0.5 * X[:, 1] + eps
    Y1 = Y0 + tau
    pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
    W = rng.binomial(1, pi)
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, tau


def run_continuous_tau(seeds):
    rows = []
    for seed in seeds:
        X, Y, W, tau = continuous_tau_dgp(seed)
        # Use linear basis [1, x_0] which is the natural smooth approx
        cate, lo, hi = fit_rx(X, Y, W, "severe", seed, "linear")
        in_ci = (lo <= tau) & (tau <= hi)
        rows.append({"seed": seed,
                     "pehe": float(np.sqrt(np.mean((cate - tau)**2))),
                     "cov_pointwise": float(np.mean(in_ci)),
                     "tau_min_hat": float(np.min(cate)),
                     "tau_max_hat": float(np.max(cate)),
                     "tau_min_true": float(np.min(tau)),
                     "tau_max_true": float(np.max(tau)),
                     "ci_width_mean": float(np.mean(hi - lo))})
        print(f"  s={seed} PEHE={rows[-1]['pehe']:.3f} cov_pw={rows[-1]['cov_pointwise']:.2f} "
              f"τ̂∈[{rows[-1]['tau_min_hat']:.2f},{rows[-1]['tau_max_hat']:.2f}] "
              f"true∈[{rows[-1]['tau_min_true']:.2f},{rows[-1]['tau_max_true']:.2f}]")
    return pd.DataFrame(rows)


# ---------- Item 5: Contaminated-normal Phase-3 likelihood ----------

def contam_normal_model(X_infer, D, sigma_in=1.0, sigma_out=10.0, eps=0.05, prior_scale=10.0):
    """Phase-3 with a 2-component contaminated-normal mixture likelihood:
    (1-ε) N(τ, σ_in²) + ε N(τ, σ_out²).
    Replaces Welsch with an explicit mixture model.
    """
    n_features = X_infer.shape[1]
    beta = numpyro.sample("beta", dist.Normal(0, prior_scale).expand([n_features]))
    tau = jnp.dot(X_infer, beta)
    log_in = dist.Normal(tau, sigma_in).log_prob(D) + jnp.log(1 - eps)
    log_out = dist.Normal(tau, sigma_out).log_prob(D) + jnp.log(eps)
    log_lik = jnp.logaddexp(log_in, log_out)
    numpyro.factor("contam_normal_ll", jnp.sum(log_lik))


def fit_contam_normal(X, Y, W, severity, seed):
    if severity == "severe":
        out_p = {"depth": 4, "iterations": 150, "loss_function": "Huber:delta=0.5"}
        prop_p = {"depth": 4, "iterations": 150}
        method = "catboost"
    else:
        out_p = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        prop_p = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        method = "xgboost"
    nuis = NuisanceEstimator(out_p, prop_p, n_splits=2,
                             random_state=seed, method=method)
    mu0, mu1, pi = nuis.fit_predict(X, Y, W)
    treated_mask, control_mask, D1, D0, *_ = impute_and_debias(
        Y, W, mu0, mu1, pi, robust=False, use_overlap=False)
    D = np.concatenate([D1, D0])
    X_inf = np.ones((len(D), 1))
    kernel = NUTS(contam_normal_model)
    mcmc = MCMC(kernel, num_warmup=400, num_samples=800, num_chains=2, progress_bar=False)
    mcmc.run(random.PRNGKey(seed), jnp.array(X_inf), jnp.array(D))
    beta = np.asarray(mcmc.get_samples()["beta"]).squeeze()
    return float(np.mean(beta)), float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def run_contam_normal(seeds):
    rows = []
    for seed in seeds:
        for density in [0.00, 0.05, 0.20]:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            for severity in ["none", "severe"]:
                ate, lo, hi = fit_contam_normal(X, Y, W, severity, seed)
                cov = int(lo <= 2.0 <= hi)
                rows.append({"seed": seed, "density": density, "severity": severity,
                             "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                             "ci_width": hi - lo})
                print(f"  s={seed} p={density:.2f} sev={severity:7s} "
                      f"ate={ate:+.3f} cov={cov} w={hi-lo:.3f}")
    return pd.DataFrame(rows)


# ---------- Item 7: Spline basis + horseshoe ----------

def spline_basis(x, knots, degree=3):
    """B-spline-like basis (cubic-truncated). Simple implementation."""
    cols = [np.ones(len(x)), x, x*x, x*x*x]
    for k in knots:
        cols.append(np.maximum(0, x - k) ** degree)
    return np.column_stack(cols)


def run_spline_basis(seeds):
    """On the tail-heterogeneous DGP, fit a spline basis with horseshoe prior on
    coefficients. Uses our existing RX-Welsch with the spline basis as X_infer."""
    rows = []
    knots = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (N, 5))
        whale = (np.abs(X[:, 0]) > WHALE_CUT).astype(float)
        tau = TAU_BULK * (1 - whale) + TAU_TAIL * whale
        Y0 = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, N)
        Y1 = Y0 + tau
        pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
        W = rng.binomial(1, pi)
        Y = np.where(W == 1, Y1, Y0)
        X_inf = spline_basis(X[:, 0], knots)
        # Use a small prior_scale to mimic horseshoe-like shrinkage
        model = TargetedBayesianXLearner(
            outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            nuisance_method="xgboost",
            n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
            robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
            prior_scale=2.0, random_state=seed,
        )
        model.fit(X, Y, W, X_infer=X_inf)
        cate, lo, hi = model.predict(X_new_infer=X_inf)
        cate = np.asarray(cate).flatten()
        whale_mask = whale.astype(bool)
        rows.append({"seed": seed,
                     "pehe": float(np.sqrt(np.mean((cate - tau)**2))),
                     "tau_hat_whale": float(np.mean(cate[whale_mask])),
                     "tau_hat_bulk": float(np.mean(cate[~whale_mask]))})
        print(f"  s={seed} spline PEHE={rows[-1]['pehe']:.3f} "
              f"τ̂_whale={rows[-1]['tau_hat_whale']:.2f} "
              f"τ̂_bulk={rows[-1]['tau_hat_bulk']:.2f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    if not (RESULTS_DIR / "arm_specific_contamination_raw.csv").exists():
        print("\n=== Item 1: arm-specific contamination ===")
        df1 = run_arm_specific(seeds)
        df1.to_csv(RESULTS_DIR / "arm_specific_contamination_raw.csv", index=False)
    else:
        df1 = pd.read_csv(RESULTS_DIR / "arm_specific_contamination_raw.csv")
        print("Loaded arm_specific from CSV (skipping rerun)")

    if not (RESULTS_DIR / "continuous_tau_coverage_raw.csv").exists():
        print("\n=== Item 4: continuous τ(x) CATE coverage ===")
        df4 = run_continuous_tau(seeds)
        df4.to_csv(RESULTS_DIR / "continuous_tau_coverage_raw.csv", index=False)
    else:
        df4 = pd.read_csv(RESULTS_DIR / "continuous_tau_coverage_raw.csv")
        print("Loaded continuous_tau from CSV (skipping rerun)")

    print("\n=== Item 5: contaminated-normal Phase-3 likelihood ===")
    df5 = run_contam_normal(seeds)
    df5.to_csv(RESULTS_DIR / "contam_normal_likelihood_raw.csv", index=False)

    print("\n=== Item 7: spline basis ===")
    df7 = run_spline_basis(seeds)
    df7.to_csv(RESULTS_DIR / "spline_basis_raw.csv", index=False)

    # Aggregate markdown writeups
    def _mean(df, col): return float(df[col].mean())
    md = ["# Round-5 reviewer-response bundled experiments", "",
          "## 1. Arm-specific contamination (whales on treated arm only)", ""]
    md.append("| seed | density | contamination | ATE | coverage | CI width |")
    md.append("|---:|---:|---|---:|---:|---:|")
    for _, r in df1.iterrows():
        md.append(f"| {int(r['seed'])} | {r['density']:.2f} | {r['contamination']} | "
                  f"{r['ate']:+.3f} | {int(r['cov'])} | {r['ci_width']:.3f} |")
    md += ["", "## 4. Continuous τ(x) CATE coverage (smooth heterogeneity)", "",
           "| seed | √PEHE | cov pointwise | τ̂ range | true τ range |",
           "|---:|---:|---:|---|---|"]
    for _, r in df4.iterrows():
        md.append(f"| {int(r['seed'])} | {r['pehe']:.3f} | {r['cov_pointwise']:.2f} | "
                  f"[{r['tau_min_hat']:.2f}, {r['tau_max_hat']:.2f}] | "
                  f"[{r['tau_min_true']:.2f}, {r['tau_max_true']:.2f}] |")
    md += ["", "## 5. Contaminated-normal Phase-3 likelihood", "",
           "| seed | density | severity | ATE | coverage | CI width |",
           "|---:|---:|---|---:|---:|---:|"]
    for _, r in df5.iterrows():
        md.append(f"| {int(r['seed'])} | {r['density']:.2f} | {r['severity']} | "
                  f"{r['ate']:+.3f} | {int(r['cov'])} | {r['ci_width']:.3f} |")
    md += ["", "## 7. Spline-basis CATE", "",
           "| seed | PEHE | τ̂_whale | τ̂_bulk |", "|---:|---:|---:|---:|"]
    for _, r in df7.iterrows():
        md.append(f"| {int(r['seed'])} | {r['pehe']:.3f} | "
                  f"{r['tau_hat_whale']:.2f} | {r['tau_hat_bulk']:.2f} |")
    (RESULTS_DIR / "round5_bundled.md").write_text("\n".join(md))
    print("wrote round5_bundled.md")


if __name__ == "__main__":
    main()
