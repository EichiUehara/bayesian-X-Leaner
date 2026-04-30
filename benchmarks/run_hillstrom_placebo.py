"""Hillstrom placebo: triangulate the near-zero ATE finding.

Reviewer concern: §5.9 reports a near-zero robust ATE on Hillstrom
where naive difference-of-means is +$0.77. Two sanity checks:

  (i) Random-label placebo: shuffle W within the dataset; the ATE
      should be statistically zero. Tests for any systematic bias in
      the pipeline.
  (ii) Treatment-symmetry: swap W → 1 − W. The ATE should flip sign
       (treated vs control symmetric).

Usage: python -u -m benchmarks.run_hillstrom_placebo
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


def fit_rx(X, Y, W, severity="severe", seed=0):
    n = len(X)
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
    model.fit(X, Y, W, X_infer=np.ones((n, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    return float(np.mean(beta)), float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def main():
    print("Loading Hillstrom..."); X, Y, W = load_hillstrom()
    n = len(X); print(f"  N={n}")
    rng = np.random.default_rng(0)

    rows = []
    # Original
    for sev in ["none", "severe"]:
        t0 = time.time()
        a, lo, hi = fit_rx(X, Y, W, severity=sev, seed=0); rt = time.time() - t0
        rows.append({"setup": "Original W", "severity": sev,
                     "ate": a, "lo": lo, "hi": hi, "rt": rt})
        print(f"  Original  sev={sev:7s} ate={a:+.4f} CI=[{lo:+.4f},{hi:+.4f}] ({rt:.0f}s)")

    # Placebo: shuffle W within the population
    for s in range(3):
        W_perm = rng.permutation(W)
        for sev in ["none", "severe"]:
            t0 = time.time()
            a, lo, hi = fit_rx(X, Y, W_perm, severity=sev, seed=s)
            rt = time.time() - t0
            rows.append({"setup": f"Placebo seed={s}", "severity": sev,
                         "ate": a, "lo": lo, "hi": hi, "rt": rt})
            print(f"  Placebo   s={s} sev={sev:7s} ate={a:+.4f} "
                  f"CI=[{lo:+.4f},{hi:+.4f}] ({rt:.0f}s)")

    # Treatment-symmetry: 1 - W; the ATE for the flipped DGP is -original_ATE
    for sev in ["none", "severe"]:
        t0 = time.time()
        a, lo, hi = fit_rx(X, Y, 1 - W, severity=sev, seed=10); rt = time.time() - t0
        rows.append({"setup": "Flipped W", "severity": sev,
                     "ate": a, "lo": lo, "hi": hi, "rt": rt})
        print(f"  Flipped   sev={sev:7s} ate={a:+.4f} CI=[{lo:+.4f},{hi:+.4f}] ({rt:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hillstrom_placebo_raw.csv", index=False)

    md = ["# Hillstrom placebo & symmetry sanity checks",
          "",
          f"Hillstrom RCT, N = {n}. RX-Welsch, intercept-only basis.",
          "Placebo: W permuted at random; expected ATE ≈ 0.",
          "Flipped: W → 1 − W; expected ATE = −original ATE.",
          "",
          "| Setup | Severity | ATE | 95% CI |",
          "|---|---|---:|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['setup']} | {r['severity']} | {r['ate']:+.4f} | "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] |")
    (RESULTS_DIR / "hillstrom_placebo.md").write_text("\n".join(md))
    print(f"\nwrote {RESULTS_DIR / 'hillstrom_placebo.md'}")


if __name__ == "__main__":
    main()
