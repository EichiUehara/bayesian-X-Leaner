"""CATE-level coverage under contamination variants.

Reviewer (round 2) concern: §5.7 measures only ATE coverage; the τ(x)
posterior calibration is what we promise. This script measures
pointwise CATE coverage on three contamination variants:

  (1) Tail-heterogeneous DGP (τ_bulk=2, τ_tail=10) with whale-style
      point-shift contamination of Y₀ at densities {0%, 5%, 20%}.
      Pointwise coverage of τ(x) credible intervals.
  (2) Low-overlap variant: same DGP, but propensity logit coefficient
      pushed to 2.0 (vs 0.3 baseline) so π saturates near 0/1; test
      with use_overlap=True.
  (3) Heavy-tail noise variant: replace whale point-shift with
      Student-t(ν=2) noise on Y₀ (genuinely heavy-tailed not just
      contaminated). Density doesn't apply here; only "noise type".

For each setup we measure:
  - Pointwise coverage of true τ(x_i) by the 95% credible interval at x_i
  - Subgroup coverage on the whale subgroup (|X_0|>1.96) and bulk
  - PEHE
  - 5 seeds

Usage: python -u -m benchmarks.run_cate_coverage --seeds 5
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
TAU_BULK = 2.0
TAU_TAIL = 10.0
WHALE_CUT = 1.96


def heterogeneous_dgp(seed, contamination=0.0, low_overlap=False, t_noise=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, 5))
    whale = (np.abs(X[:, 0]) > WHALE_CUT).astype(float)
    tau = TAU_BULK * (1 - whale) + TAU_TAIL * whale

    if t_noise:
        # Student-t(ν=2) heavy-tailed noise (no point contamination)
        eps = rng.standard_t(df=2, size=N)
    else:
        eps = rng.normal(0.0, 1.0, size=N)
    Y0 = X[:, 0] + 0.5 * X[:, 1] + eps
    Y1 = Y0 + tau

    # Point-contamination whales (only when contamination > 0 and not t_noise)
    if contamination > 0 and not t_noise:
        n_whales = int(round(contamination * N))
        idx = rng.choice(N, size=n_whales, replace=False)
        Y0[idx] += 10 * np.sign(rng.normal(size=n_whales))
        Y1[idx] += 10 * np.sign(rng.normal(size=n_whales))

    logit_c = 2.0 if low_overlap else 0.3
    pi_min, pi_max = (0.02, 0.98) if low_overlap else (0.1, 0.9)
    pi = np.clip(1.0 / (1.0 + np.exp(-logit_c * X[:, 0])), pi_min, pi_max)
    W = rng.binomial(1, pi).astype(int)
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, tau, whale


def fit_and_score(X, Y, W, tau, whale, seed, severity="severe", use_overlap=False):
    """Fit RX-Learner with tail-aware basis, return per-unit
    CI bounds and aggregate coverage metrics."""
    w_col = (np.abs(X[:, 0]) > WHALE_CUT).astype(float).reshape(-1, 1)
    X_infer = np.hstack([np.ones((N, 1)), w_col])
    kwargs = dict(
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34, use_student_t=True,
        mad_rescale=False, random_state=seed,
        contamination_severity=severity,
        use_overlap=use_overlap,
    )
    model = TargetedBayesianXLearner(**kwargs)
    model.fit(X, Y, W, X_infer=X_infer)
    cate, lo, hi = model.predict(X_new_infer=X_infer)
    cate = np.asarray(cate).flatten()
    lo = np.asarray(lo).flatten()
    hi = np.asarray(hi).flatten()
    in_ci = (lo <= tau) & (tau <= hi)

    whale_mask = whale.astype(bool)
    return {
        "pehe": float(np.sqrt(np.mean((cate - tau) ** 2))),
        "coverage_pointwise": float(np.mean(in_ci)),
        "coverage_whale": float(np.mean(in_ci[whale_mask])) if whale_mask.any() else float("nan"),
        "coverage_bulk":  float(np.mean(in_ci[~whale_mask])),
        "tau_hat_whale": float(np.mean(cate[whale_mask])) if whale_mask.any() else float("nan"),
        "tau_hat_bulk":  float(np.mean(cate[~whale_mask])),
        "ci_width_mean": float(np.mean(hi - lo)),
    }


SETUPS = [
    # name, (contamination, low_overlap, t_noise), severity, use_overlap
    ("clean",           (0.00, False, False), "severe", False),
    ("contam_5%",       (0.05, False, False), "severe", False),
    ("contam_20%",      (0.20, False, False), "severe", False),
    ("low_overlap",     (0.05, True,  False), "severe", True),  # overlap weights enabled
    ("t_nu2_noise",     (0.00, False, True),  "severe", False),
]


def _run(seeds):
    rows = []
    for seed in seeds:
        for name, dgp_kwargs, severity, use_overlap in SETUPS:
            contam, low_overlap, t_noise = dgp_kwargs
            X, Y, W, tau, whale = heterogeneous_dgp(
                seed, contamination=contam, low_overlap=low_overlap, t_noise=t_noise)
            t0 = time.time()
            try:
                m = fit_and_score(X, Y, W, tau, whale, seed, severity, use_overlap)
                rt = time.time() - t0
                rows.append({"seed": seed, "setup": name, **m, "runtime": rt})
                print(f"  seed={seed} {name:14s} PEHE={m['pehe']:.3f} "
                      f"cov_pw={m['coverage_pointwise']:.2f} "
                      f"cov_whale={m['coverage_whale']:.2f} "
                      f"cov_bulk={m['coverage_bulk']:.2f} "
                      f"width={m['ci_width_mean']:.3f} ({rt:.1f}s)")
            except Exception as e:
                print(f"  seed={seed} {name:14s} ERR {e}")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby("setup").agg(
        n=("seed", "count"),
        pehe=("pehe", "mean"),
        cov_pointwise=("coverage_pointwise", "mean"),
        cov_whale=("coverage_whale", "mean"),
        cov_bulk=("coverage_bulk", "mean"),
        ci_width=("ci_width_mean", "mean"),
    ).reindex([s[0] for s in SETUPS])


def _write_markdown(df, agg):
    path = RESULTS_DIR / "cate_coverage.md"
    lines = [
        "# CATE-level coverage under contamination variants",
        "",
        f"Tail-heterogeneous DGP (τ_bulk={TAU_BULK}, τ_tail={TAU_TAIL}, "
        f"whale=|X₀|>{WHALE_CUT}), N = {N}, 5 seeds.",
        "Tail-aware basis $[1, \\mathbf{1}(|X_0|>1.96)]$, "
        "`severity=severe`, MCMC posterior credible intervals.",
        "",
        "Pointwise coverage: fraction of units i where τ(x_i) ∈ CI_i.",
        "",
        "| setup | n | √PEHE | cov pointwise | cov whale | cov bulk | mean CI width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in agg.iterrows():
        if pd.isna(r['n']):
            continue
        lines.append(
            f"| {name} | {int(r['n'])} | {r['pehe']:.3f} | "
            f"{r['cov_pointwise']:.2f} | {r['cov_whale']:.2f} | "
            f"{r['cov_bulk']:.2f} | {r['ci_width']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "cate_coverage_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──"); print(agg.to_string())
    _write_markdown(df, agg)


if __name__ == "__main__":
    main()
