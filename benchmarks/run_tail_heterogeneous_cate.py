"""Tail-heterogeneous CATE — tails-as-signal vs tails-as-contamination.

The whale / §17 robustness story treats extreme pseudo-outcomes as
contamination: Welsch downweights them, CB-Huber loss suppresses their
influence on the nuisance fit, and `normalize_extremes` scales them
down at the data layer. All three assume τ is *homogeneous* across
bulk and tail — the tail is just noise.

But a practitioner studying, e.g., rare catastrophic adverse events or
long-tail revenue whales may have the opposite belief: **τ is different
on the tail**, and the tail is the phenomenon of interest, not noise.
This script asks two questions:

  Q1. With an intercept-only CATE basis (no tail-aware covariate), how
      do the three likelihood configurations (Gaussian / Welsch /
      Welsch+EVT-scaling) estimate the mixed ATE?
  Q2. With a tail-aware basis `[1, 1(|X₀|>t)]`, can the library
      recover τ_bulk and τ_tail separately?

DGP
---
  X ~ N(0, I₅),  p = 5,  N = 1000
  whale = 1(|X₀| > 1.96)       (≈ 5 % tail mass under N(0,1))
  τ(X)  = τ_bulk·(1 − whale) + τ_tail·whale,   τ_bulk = 2,  τ_tail = 10
  Y₀    = X₀ + 0.5·X₁ + ε,     ε ~ N(0, 1)
  π(X)  = σ(0.3·X₀) clipped to [0.1, 0.9]
  W     ~ Bern(π),  Y = W·(Y₀+τ) + (1−W)·Y₀

True mixed ATE ≈ P(whale)·10 + P(bulk)·2 ≈ 2.4.
True whale ATE = 10 exactly; true bulk ATE = 2 exactly.

Metrics
-------
  - ε_ATE (mixed)       = |τ̂_mean − 2.4|
  - ε_ATE (whale)       = |τ̂_whale − 10|   (only with tail-aware basis)
  - ε_ATE (bulk)        = |τ̂_bulk − 2|     (only with tail-aware basis)
  - cov_whale           = 1 iff 10 ∈ CI_whale
  - cov_mixed           = 1 iff 2.4 ∈ CI_mean

Usage:
    python -m benchmarks.run_tail_heterogeneous_cate --seeds 5
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
from sert_xlearner.core.evt import estimate_tail_parameters


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TAU_BULK = 2.0
TAU_TAIL = 10.0
WHALE_CUT = 1.96


def tail_heterogeneous_dgp(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, 5))
    whale = (np.abs(X[:, 0]) > WHALE_CUT).astype(float)
    tau = TAU_BULK * (1 - whale) + TAU_TAIL * whale
    eps = rng.normal(0.0, 1.0, size=N)
    Y0 = X[:, 0] + 0.5 * X[:, 1] + eps
    Y1 = Y0 + tau
    logits = 0.3 * X[:, 0]
    pi = 1.0 / (1.0 + np.exp(-logits))
    pi = np.clip(pi, 0.1, 0.9)
    W = rng.binomial(1, pi).astype(int)
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, tau, whale


def _fit_one(X, Y, W, whale, basis, robust, use_evt, seed):
    """Returns dict with τ̂_mean, τ̂_whale, τ̂_bulk, CIs, runtime.

    basis ∈ {"intercept", "tail_aware"}
    robust True = Welsch likelihood; False = Gaussian
    use_evt True = estimate tail_threshold/tail_alpha via Hill on
                   pseudo-outcomes proxy (|Y - Y.mean()|).
    """
    if basis == "intercept":
        X_infer = np.ones((N, 1))
    elif basis == "tail_aware":
        w_col = (np.abs(X[:, 0]) > WHALE_CUT).astype(float).reshape(-1, 1)
        X_infer = np.hstack([np.ones((N, 1)), w_col])
    else:
        raise ValueError(basis)

    tail_threshold = tail_alpha = None
    if use_evt:
        residuals = Y - np.mean(Y)
        tail_threshold, tail_alpha = estimate_tail_parameters(
            residuals, top_percentile=90
        )

    # XGB-MSE nuisance — clean Y₀, no outcome contamination — keeps
    # attention on the likelihood-level effect.
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        nuisance_method="xgboost",
        n_splits=2,
        num_warmup=400, num_samples=800, num_chains=2,
        robust=robust,
        c_whale=1.34,
        use_student_t=robust,
        tail_threshold=tail_threshold,
        tail_alpha=tail_alpha,
        random_state=seed,
        mad_rescale=False,  # isolate the tail machinery we're testing
    )

    t0 = time.time()
    model.fit(X, Y, W, X_infer=X_infer)
    tau_hat, ci_lo, ci_hi = model.predict(X_new_infer=X_infer)
    runtime = time.time() - t0

    whale_mask = whale.astype(bool)
    tau_hat = np.asarray(tau_hat).flatten()
    ci_lo = np.asarray(ci_lo).flatten()
    ci_hi = np.asarray(ci_hi).flatten()

    out = {
        "tau_mean":    float(np.mean(tau_hat)),
        "ci_mean_lo":  float(np.mean(ci_lo)),
        "ci_mean_hi":  float(np.mean(ci_hi)),
        "tau_whale":   float(np.mean(tau_hat[whale_mask])) if whale_mask.any() else float("nan"),
        "ci_whale_lo": float(np.mean(ci_lo[whale_mask])) if whale_mask.any() else float("nan"),
        "ci_whale_hi": float(np.mean(ci_hi[whale_mask])) if whale_mask.any() else float("nan"),
        "tau_bulk":    float(np.mean(tau_hat[~whale_mask])),
        "runtime":     runtime,
        "tail_threshold": float(tail_threshold) if tail_threshold is not None else None,
        "tail_alpha":  float(tail_alpha) if tail_alpha is not None else None,
        "n_whales":    int(whale_mask.sum()),
    }
    return out


CONFIGS = [
    # (label, robust, use_evt)
    ("Gaussian",        False, False),
    ("Welsch",          True,  False),
    ("Welsch+EVT",      True,  True),
]
BASES = ["intercept", "tail_aware"]


def _run(seeds):
    rows = []
    for seed in seeds:
        X, Y, W, tau, whale = tail_heterogeneous_dgp(seed)
        n_w = int(whale.sum())
        true_mixed = float(np.mean(tau))
        print(f"\n── seed {seed}  n_whales={n_w}  "
              f"true_mixed_ATE={true_mixed:.3f}  "
              f"(bulk={TAU_BULK}, tail={TAU_TAIL}) ──")
        for basis in BASES:
            for label, robust, use_evt in CONFIGS:
                try:
                    r = _fit_one(X, Y, W, whale, basis, robust, use_evt, seed)
                except Exception as e:
                    print(f"  [{basis:10s}] {label:14s} ERR {e}")
                    continue
                eps_mixed = abs(r["tau_mean"] - true_mixed)
                eps_whale = abs(r["tau_whale"] - TAU_TAIL)
                eps_bulk  = abs(r["tau_bulk"]  - TAU_BULK)
                cov_mixed = int(r["ci_mean_lo"] <= true_mixed <= r["ci_mean_hi"])
                cov_whale = int(r["ci_whale_lo"] <= TAU_TAIL <= r["ci_whale_hi"])
                rows.append({
                    "seed": seed, "basis": basis, "config": label,
                    "n_whales": n_w, "true_mixed": true_mixed,
                    "tau_mean": r["tau_mean"],
                    "tau_whale": r["tau_whale"],
                    "tau_bulk": r["tau_bulk"],
                    "eps_mixed": eps_mixed,
                    "eps_whale": eps_whale,
                    "eps_bulk":  eps_bulk,
                    "cov_mixed": cov_mixed,
                    "cov_whale": cov_whale,
                    "ci_whale_lo": r["ci_whale_lo"],
                    "ci_whale_hi": r["ci_whale_hi"],
                    "runtime": r["runtime"],
                    "tail_threshold": r["tail_threshold"],
                    "tail_alpha": r["tail_alpha"],
                })
                print(f"  [{basis:10s}] {label:14s} "
                      f"τ̂_mean={r['tau_mean']:6.3f} (ε={eps_mixed:.3f}) "
                      f"τ̂_whale={r['tau_whale']:6.3f} (ε={eps_whale:.3f}) "
                      f"τ̂_bulk={r['tau_bulk']:6.3f} (ε={eps_bulk:.3f}) "
                      f"cov_whale={cov_whale} ({r['runtime']:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    grp = df.groupby(["basis", "config"])
    return grp.agg(
        n=("seed", "count"),
        eps_mixed=("eps_mixed", "mean"),
        eps_whale=("eps_whale", "mean"),
        eps_bulk=("eps_bulk", "mean"),
        tau_whale=("tau_whale", "mean"),
        tau_bulk=("tau_bulk", "mean"),
        cov_mixed=("cov_mixed", "mean"),
        cov_whale=("cov_whale", "mean"),
        runtime=("runtime", "mean"),
    ).reset_index()


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "tail_heterogeneous_cate.md"
    lines = [
        "# Tail-heterogeneous CATE — tails-as-signal probe",
        "",
        "DGP:  X ~ N(0, I₅),  p=5,  N=1000.  whale = 1(|X₀|>1.96) (~5 % tail mass).",
        "  τ(X) = 2·(1−whale) + 10·whale,    Y₀ = X₀ + 0.5 X₁ + N(0,1),",
        "  π(X) = σ(0.3 X₀) clipped to [0.1, 0.9],   W ~ Bern(π).",
        f"Seeds: {list(seeds)}.  True mixed ATE ≈ 2.4;  true τ_whale = 10;  true τ_bulk = 2.",
        "",
        "Nuisance: XGB-MSE (no outcome contamination in this DGP; we are",
        "isolating the *likelihood-level* tail handling, not the nuisance-level).",
        "mad_rescale disabled to avoid the §14 MAD-contamination pathway.",
        "",
        "Configurations:",
        "  - **Gaussian**      : `robust=False`  (standard Bayesian likelihood)",
        "  - **Welsch**        : `robust=True`, no EVT",
        "  - **Welsch+EVT**    : `robust=True` + Hill-estimated `tail_threshold`,",
        "                         `tail_alpha` applied via `normalize_extremes`",
        "",
        "Bases:",
        "  - **intercept**     : X_infer = [1]        (scalar ATE)",
        "  - **tail_aware**    : X_infer = [1, 1(|X₀|>1.96)]  (bulk + tail dummy)",
        "",
        "## Summary",
        "",
        "| basis | config | n | ε_ATE (mixed) | cov mixed | τ̂_whale | ε_ATE (whale) | cov whale | τ̂_bulk | ε_ATE (bulk) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['basis']} | {r['config']} | {int(r['n'])} | "
            f"{r['eps_mixed']:.3f} | {r['cov_mixed']:.2f} | "
            f"{r['tau_whale']:.3f} | {r['eps_whale']:.3f} | {r['cov_whale']:.2f} | "
            f"{r['tau_bulk']:.3f} | {r['eps_bulk']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "tail_heterogeneous_cate_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'tail_heterogeneous_cate_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string(index=False))
    _write_markdown(df, agg, seeds)


if __name__ == "__main__":
    main()
