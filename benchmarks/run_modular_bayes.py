"""Modular-Bayes nuisance uncertainty propagation.

Implements the cut/modular-Bayes construction of
\\citet{plummer2015cuts, jacob2017better} adapted to the DR + Welsch
pipeline (paper §4.4):

  1. Draw M Bayesian-bootstrap weights w^(m) ~ Dirichlet(1,...,1) on the
     N training units.
  2. Refit cross-fitted nuisances on each weighted resample to get
     (μ̂₀^(m), μ̂₁^(m), π̂^(m)).
  3. Build pseudo-outcomes D^(m) for each and run Phase 3 NUTS to obtain
     β-posterior draws {β^(m,s)}.
  4. Pool by:
     (a) modular-cut concatenation: pooled posterior = ⋃_m {β^(m,s)};
     (b) Rubin's rules: μ̄ = mean over m, V_total = mean(within) +
        (1 + 1/M) × var(between).

Reports point estimate, 95% credible interval, and coverage on the
whale DGP at densities {0%, 5%, 20%}.

Usage: python -u -m benchmarks.run_modular_bayes --M 10 --seeds 3
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


def fit_one_with_weights(X, Y, W, weights, severity, seed):
    """Fit RX-Welsch with Bayesian-bootstrap weights on the training data.

    The simplest implementation: resample (X, Y, W) with weights as
    multinomial probabilities. Welsch likelihood uses W_D1, W_D0
    multipliers internally; we pass weights as part of the DR
    pseudo-outcome scaling.

    For practical purposes here we use a hard-bootstrap surrogate for
    the Bayesian-bootstrap weights: resample N indices with replacement
    using the weights as probabilities. This converges to the same
    distribution at scale.
    """
    rng = np.random.default_rng(seed)
    N_ = len(X)
    # Hard bootstrap from Dirichlet weights
    idx = rng.choice(N_, size=N_, replace=True, p=weights)
    Xb, Yb, Wb = X[idx], Y[idx], W[idx]
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
    model.fit(Xb, Yb, Wb, X_infer=np.ones((N_, 1)))
    return np.asarray(model.mcmc_samples["beta"]).squeeze().flatten()


def modular_bayes(X, Y, W, severity, M, base_seed):
    """M Bayesian-bootstrap nuisance draws, each producing a Stage-3
    posterior. Returns list of β-arrays plus pooled diagnostics."""
    rng = np.random.default_rng(base_seed)
    N_ = len(X)
    chains = []
    for m in range(M):
        weights = rng.dirichlet(np.ones(N_))
        try:
            beta = fit_one_with_weights(X, Y, W, weights, severity, base_seed * 1000 + m)
            chains.append(beta)
        except Exception as e:
            print(f"    m={m} failed: {e}")
    if not chains:
        return None

    # Modular-cut posterior: concatenate all chains
    concat = np.concatenate(chains)
    # Rubin's rules
    means = np.array([np.mean(c) for c in chains])
    within_vars = np.array([np.var(c, ddof=1) for c in chains])
    rubin_mean = float(np.mean(means))
    within_v = float(np.mean(within_vars))
    between_v = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
    rubin_var = within_v + (1 + 1.0 / len(chains)) * between_v

    return {
        "M": len(chains),
        "concat_mean": float(np.mean(concat)),
        "concat_lo": float(np.percentile(concat, 2.5)),
        "concat_hi": float(np.percentile(concat, 97.5)),
        "rubin_mean": rubin_mean,
        "rubin_lo": rubin_mean - 1.96 * np.sqrt(rubin_var),
        "rubin_hi": rubin_mean + 1.96 * np.sqrt(rubin_var),
        "within_v": within_v,
        "between_v": between_v,
    }


def _run(seeds, M):
    rows = []
    for seed in seeds:
        for density in DENSITIES:
            n_w = int(round(density * N))
            X, Y, W, _ = whale_dgp(N=N, n_whales=n_w, seed=seed)
            for severity in ["none", "severe"]:
                t0 = time.time()
                r = modular_bayes(X, Y, W, severity, M, seed)
                rt = time.time() - t0
                if r is None:
                    continue
                cov_concat = int(r["concat_lo"] <= TRUE_ATE <= r["concat_hi"])
                cov_rubin = int(r["rubin_lo"] <= TRUE_ATE <= r["rubin_hi"])
                rows.append({
                    "seed": seed, "density": density, "severity": severity, "M": r["M"],
                    "concat_mean": r["concat_mean"],
                    "concat_lo": r["concat_lo"], "concat_hi": r["concat_hi"],
                    "concat_width": r["concat_hi"] - r["concat_lo"],
                    "concat_cov": cov_concat,
                    "rubin_mean": r["rubin_mean"],
                    "rubin_lo": r["rubin_lo"], "rubin_hi": r["rubin_hi"],
                    "rubin_width": r["rubin_hi"] - r["rubin_lo"],
                    "rubin_cov": cov_rubin,
                    "between_v": r["between_v"], "within_v": r["within_v"],
                    "runtime": rt,
                })
                print(f"  s={seed} p={density:.2f} sev={severity:7s} "
                      f"concat: ate={r['concat_mean']:+.3f} cov={cov_concat} w={r['concat_hi']-r['concat_lo']:.2f} | "
                      f"rubin: ate={r['rubin_mean']:+.3f} cov={cov_rubin} w={r['rubin_hi']-r['rubin_lo']:.2f} ({rt:.0f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby(["density", "severity"]).agg(
        n=("seed", "count"),
        concat_cov=("concat_cov", "mean"),
        concat_width=("concat_width", "mean"),
        rubin_cov=("rubin_cov", "mean"),
        rubin_width=("rubin_width", "mean"),
    ).reset_index()


def _write_markdown(df, agg, M):
    path = RESULTS_DIR / "modular_bayes.md"
    lines = [
        "# Modular-Bayes nuisance uncertainty propagation",
        "",
        f"Bayesian-bootstrap M = {M} nuisance draws on the whale DGP, N = {N}, true ATE = {TRUE_ATE}.",
        "",
        "Two pooling rules: (a) modular-cut concatenation; (b) Rubin's rules with",
        "(1 + 1/M) between-imputation correction.",
        "",
        "| density | severity | n | concat coverage | concat width | Rubin coverage | Rubin width |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['density']:.2f} | {r['severity']} | {int(r['n'])} | "
            f"{r['concat_cov']:.2f} | {r['concat_width']:.3f} | "
            f"{r['rubin_cov']:.2f} | {r['rubin_width']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "modular_bayes_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds, args.M)
    agg = _summarise(df)
    print("\n── Summary ──"); print(agg.to_string(index=False))
    _write_markdown(df, agg, args.M)


if __name__ == "__main__":
    main()
