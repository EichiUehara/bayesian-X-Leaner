"""Pareto-contamination DGP — third tail family (vs whale + t_2).

Reviewer concern: contamination shape robustness. We have point-shift
(whale), Student-t(2) noise, and now Pareto (genuinely heavy-tailed
power-law contamination of Y).

DGP: standard linear potential outcomes with Pareto(α=1.5)
contamination on Y_0 with density p ∈ {0%, 5%, 20%}. Pareto with
α = 1.5 has finite mean but infinite variance — the canonical
"genuinely heavy-tailed" case.

Usage: python -u -m benchmarks.run_pareto_contamination --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0
DENSITIES = [0.00, 0.05, 0.20]
PARETO_ALPHA = 1.5


def pareto_contamination_dgp(seed, contamination=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, 5))
    eps = rng.normal(0.0, 1.0, size=N)
    Y0 = X[:, 0] + 0.5 * X[:, 1] + eps
    if contamination > 0:
        n_c = int(round(contamination * N))
        idx = rng.choice(N, size=n_c, replace=False)
        # Pareto-shaped additive contamination, random sign
        Y0[idx] += (rng.pareto(PARETO_ALPHA, size=n_c) * 5
                    * np.sign(rng.normal(size=n_c)))
    Y1 = Y0 + TRUE_ATE
    pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
    W = rng.binomial(1, pi).astype(int)
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W


def fit_rx(X, Y, W, severity, seed):
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
    model.fit(X, Y, W, X_infer=np.ones((N, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    return float(np.mean(beta)), float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def _run(seeds):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            X, Y, W = pareto_contamination_dgp(seed, contamination=density)
            for severity in ["none", "severe"]:
                t0 = time.time()
                ate, lo, hi = fit_rx(X, Y, W, severity, seed); rt = time.time() - t0
                cov = int(lo <= TRUE_ATE <= hi)
                rows.append({
                    "seed": seed, "density": density, "severity": severity,
                    "ate": ate, "lo": lo, "hi": hi, "cov": cov,
                    "ci_width": hi - lo, "runtime": rt,
                })
                print(f"  seed={seed} p={density:.2f} sev={severity:7s} "
                      f"ate={ate:+.3f} cov={cov} width={hi-lo:.3f} ({rt:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby(["density", "severity"]).agg(
        n=("seed", "count"),
        ate_mean=("ate", "mean"),
        bias=("ate", lambda s: float(np.mean(s - TRUE_ATE))),
        coverage=("cov", "mean"),
        ci_width=("ci_width", "mean"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "pareto_contamination.md"
    lines = [
        "# Pareto-contamination DGP — third tail family",
        "",
        f"Pareto(α={PARETO_ALPHA}) additive contamination on Y₀.",
        f"N = {N}, true ATE = {TRUE_ATE}, 3 seeds.",
        "Pareto with α=1.5 has finite mean but infinite variance.",
        "",
        "| density | severity | n | bias | 95% coverage | CI width |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {r['severity']} | {int(r['n'])} | "
            f"{r['bias']:+.3f} | {r['coverage']:.2f} | {r['ci_width']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "pareto_contamination_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──"); print(agg.to_string(index=False))
    _write_markdown(df, agg)


if __name__ == "__main__":
    main()
