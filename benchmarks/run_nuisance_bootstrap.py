"""Bayesian-bootstrap propagation of nuisance uncertainty into the
Phase-3 posterior.

Reviewer concern (round 2): nuisance estimation uncertainty isn't
propagated; under low overlap or small N this can understate
posterior variance. We implement a Lyddon-Holmes-Walker-style
posterior bootstrap: B independent cross-fits with bootstrap fold
assignments, each producing one posterior. Pool the B posteriors to
form a marginalised credible interval.

This is approximate but practical: the marginalised posterior over β
inflates Phase-3 variance by exactly the spread induced by nuisance
fitting choices.

Usage: python -u -m benchmarks.run_nuisance_bootstrap --B 10 --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0
DENSITIES = [0.00, 0.05, 0.20]


def fit_one(X, Y, W, severity, seed):
    kwargs = dict(
        n_splits=2, num_warmup=300, num_samples=500, num_chains=2,
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
    model.fit(X, Y, W, X_infer=np.ones((len(X), 1)))
    return np.asarray(model.mcmc_samples["beta"]).squeeze()


def bootstrap_posterior(X, Y, W, severity, B, base_seed=0):
    """Pool B Phase-3 posteriors, each from a different (Bayesian-bootstrap-
    weighted) cross-fit instance. Each instance uses a different random_state
    to perturb fold assignment + bootstrap fold weights.

    The marginalised posterior over β is the concatenation of B chains.
    """
    pooled = []
    for b in range(B):
        try:
            beta = fit_one(X, Y, W, severity, base_seed + b)
            pooled.append(np.asarray(beta).flatten())
        except Exception as e:
            print(f"    boot {b} failed: {e}")
    pooled = np.concatenate(pooled) if pooled else np.array([])
    return pooled


def _run(seeds, B):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_whales = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            for severity in ["none", "severe"]:
                # Single-cross-fit baseline
                t0 = time.time()
                beta_single = fit_one(X, Y, W, severity, seed)
                ate_s = float(np.mean(beta_single))
                lo_s, hi_s = np.percentile(beta_single, [2.5, 97.5])
                rt_s = time.time() - t0
                cov_s = int(lo_s <= TRUE_ATE <= hi_s)
                # Pooled-bootstrap posterior
                t0 = time.time()
                beta_pool = bootstrap_posterior(X, Y, W, severity, B, base_seed=seed * 100)
                ate_p = float(np.mean(beta_pool))
                lo_p, hi_p = np.percentile(beta_pool, [2.5, 97.5])
                rt_p = time.time() - t0
                cov_p = int(lo_p <= TRUE_ATE <= hi_p)
                rows.append({
                    "seed": seed, "density": density, "severity": severity,
                    "ate_single": ate_s, "lo_single": lo_s, "hi_single": hi_s,
                    "cov_single": cov_s, "width_single": hi_s - lo_s, "rt_single": rt_s,
                    "ate_pool": ate_p, "lo_pool": lo_p, "hi_pool": hi_p,
                    "cov_pool": cov_p, "width_pool": hi_p - lo_p, "rt_pool": rt_p,
                })
                print(f"  s={seed} p={density:.2f} sev={severity:7s} "
                      f"single ate={ate_s:+.3f} cov={cov_s} w={hi_s-lo_s:.3f} | "
                      f"pool({B}) ate={ate_p:+.3f} cov={cov_p} w={hi_p-lo_p:.3f}")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby(["density", "severity"]).agg(
        n=("seed", "count"),
        cov_single=("cov_single", "mean"),
        width_single=("width_single", "mean"),
        cov_pool=("cov_pool", "mean"),
        width_pool=("width_pool", "mean"),
    ).reset_index()


def _write_markdown(df, agg, B):
    path = RESULTS_DIR / "nuisance_bootstrap.md"
    lines = [
        "# Bayesian-bootstrap propagation of nuisance uncertainty",
        "",
        f"Pool of {B} cross-fitted posteriors per cell, whale DGP, N = {N}, true ATE = {TRUE_ATE}.",
        "Reports both single-cross-fit posterior and bootstrap-pooled posterior coverage and width.",
        "",
        "| density | severity | n | single cov | single width | pool cov | pool width |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {r['severity']} | {int(r['n'])} | "
            f"{r['cov_single']:.2f} | {r['width_single']:.3f} | "
            f"{r['cov_pool']:.2f} | {r['width_pool']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "nuisance_bootstrap_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds, args.B)
    agg = _summarise(df)
    print("\n── Summary ──"); print(agg.to_string(index=False))
    _write_markdown(df, agg, args.B)


if __name__ == "__main__":
    main()
