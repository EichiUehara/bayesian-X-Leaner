"""Hillstrom subgroup posteriors — go beyond marginal ATE.

Reviewer concern: §5.9 reports Hillstrom marginal ATE only; the
paper's promise is τ(x), not τ. This script fits RX-Welsch with
a tail-aware basis on Hillstrom and reports posterior summaries
for predefined subgroups (recency bins, history bins).

Usage: python -u -m benchmarks.run_hillstrom_subgroups
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import time
from pathlib import Path
import numpy as np
import pandas as pd

from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from benchmarks.run_hillstrom import load_hillstrom


RESULTS_DIR = Path(__file__).parent / "results"


def main():
    print("Loading Hillstrom...")
    X, Y, W = load_hillstrom()
    n = len(X)
    print(f"  N={n}, treated={int(W.sum())}")

    # Build a basis with two subgroup indicators on continuous covariates:
    #   - high_recency: months since last purchase > median
    #   - high_history: lifetime spend (history) > 90th percentile (heavy-tail
    #     subgroup most likely to drive marginal ATE)
    recency = X[:, 0]  # column order in load_hillstrom
    history = X[:, 1]
    high_recency = (recency > np.median(recency)).astype(float)
    high_history = (history > np.percentile(history, 90)).astype(float)
    X_infer = np.column_stack([
        np.ones(n), high_recency, high_history,
    ])

    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        nuisance_method="xgboost",
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
        random_state=0,
    )
    t0 = time.time()
    model.fit(X, Y, W, X_infer=X_infer)
    cate, lo, hi = model.predict(X_new_infer=X_infer)
    cate = np.asarray(cate).flatten()
    lo = np.asarray(lo).flatten()
    hi = np.asarray(hi).flatten()
    rt = time.time() - t0
    print(f"Fit + predict in {rt:.1f}s")

    # Subgroup summaries
    rows = []
    masks = {
        "all":               np.ones(n, dtype=bool),
        "low recency":       ~high_recency.astype(bool),
        "high recency":      high_recency.astype(bool),
        "high history (top 10%)": high_history.astype(bool),
        "low history":       ~high_history.astype(bool),
    }
    for label, mask in masks.items():
        if mask.sum() == 0:
            continue
        rows.append({
            "subgroup": label,
            "n": int(mask.sum()),
            "tau_hat_mean": float(np.mean(cate[mask])),
            "ci_lo_mean":   float(np.mean(lo[mask])),
            "ci_hi_mean":   float(np.mean(hi[mask])),
            "ci_width":     float(np.mean(hi[mask] - lo[mask])),
        })
        print(f"  {label:24s} n={mask.sum():6d} τ̂={np.mean(cate[mask]):+.4f} "
              f"CI=[{np.mean(lo[mask]):+.4f},{np.mean(hi[mask]):+.4f}]")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hillstrom_subgroups_raw.csv", index=False)

    md = ["# Hillstrom subgroup posteriors",
          "",
          f"N = {n}, basis = $[1, \\mathbf{{1}}(\\text{{recency above median}}), "
          f"\\mathbf{{1}}(\\text{{history in top 10\\%}})]$.",
          "RX-Welsch (severity=none, default config). Posterior credible interval per unit.",
          "",
          "| Subgroup | n | mean τ̂ | mean 95% CI | mean width |",
          "|---|---:|---:|---|---:|"]
    for _, r in df.iterrows():
        md.append(
            f"| {r['subgroup']} | {r['n']} | {r['tau_hat_mean']:+.4f} | "
            f"[{r['ci_lo_mean']:+.4f}, {r['ci_hi_mean']:+.4f}] | "
            f"{r['ci_width']:.4f} |"
        )
    (RESULTS_DIR / "hillstrom_subgroups.md").write_text("\n".join(md))
    print(f"\nwrote {RESULTS_DIR / 'hillstrom_subgroups.md'}")


if __name__ == "__main__":
    main()
