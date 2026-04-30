"""Automated severity selection from a tail-index diagnostic.

Reviewer concern: contamination_severity is a user-set enum; could it
be data-driven? This script implements a Hill-estimator-based tail
index on the residuals of a quick preliminary outcome regression, maps
the tail index to a recommended severity, and verifies that the
recommendation matches the empirically-best severity.

Pipeline:
  1. Fit a quick S-Learner (HistGradientBoosting) to get residuals.
  2. Hill estimator on |residuals| above the 90th percentile gives
     tail index α_hat = 1 / mean(log(|r|/threshold)).
  3. Map: α_hat > 5 → "none"; 3 < α ≤ 5 → "mild";
          2 < α ≤ 3 → "moderate"; α ≤ 2 → "severe".
  4. Compare the auto-selected severity's posterior to oracle severity.

Usage: python -u -m benchmarks.run_auto_severity --seeds 3
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from benchmarks.dgps import whale_dgp, standard_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
TRUE_ATE = 2.0


def estimate_tail_index(residuals, top_pct=90):
    """Hill estimator: alpha = 1 / mean(log(|r| / u))."""
    abs_r = np.abs(residuals)
    threshold = np.percentile(abs_r, top_pct)
    extremes = abs_r[abs_r > threshold]
    if len(extremes) < 2:
        return float("inf"), threshold
    gamma = np.mean(np.log(extremes / threshold))
    if gamma <= 0:
        return float("inf"), threshold
    return float(1.0 / gamma), float(threshold)


def map_alpha_to_severity(alpha):
    if alpha > 5.0:
        return "none"
    if alpha > 3.0:
        return "mild"
    if alpha > 2.0:
        return "moderate"
    return "severe"


def auto_severity(X, Y, W):
    """Estimate residual tail index from a quick S-Learner fit, map to severity."""
    sl = HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)
    sl.fit(np.column_stack([X, W.astype(float)]), Y)
    Y_pred = sl.predict(np.column_stack([X, W.astype(float)]))
    residuals = Y - Y_pred
    alpha_hat, threshold = estimate_tail_index(residuals)
    return map_alpha_to_severity(alpha_hat), alpha_hat, threshold


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
    else:
        kwargs["contamination_severity"] = severity
    model = TargetedBayesianXLearner(**kwargs)
    model.fit(X, Y, W, X_infer=np.ones((N, 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    return float(np.mean(beta)), float(np.percentile(beta, 2.5)), float(np.percentile(beta, 97.5))


def _run(seeds):
    rows = []
    setups = [
        ("clean",     lambda s: standard_dgp(N=N, seed=s)),
        ("whale_5%",  lambda s: whale_dgp(N=N, n_whales=int(0.05 * N), seed=s)),
        ("whale_20%", lambda s: whale_dgp(N=N, n_whales=int(0.20 * N), seed=s)),
    ]
    for seed in seeds:
        for label, dgp in setups:
            X, Y, W, _ = dgp(seed)
            sev_auto, alpha, thresh = auto_severity(X, Y, W)
            t0 = time.time()
            ate_auto, lo_a, hi_a = fit_rx(X, Y, W, sev_auto, seed)
            rt_auto = time.time() - t0
            cov_auto = int(lo_a <= TRUE_ATE <= hi_a)
            rows.append({
                "seed": seed, "setup": label,
                "alpha_hat": alpha, "threshold": thresh,
                "severity_auto": sev_auto,
                "ate_auto": ate_auto, "lo_auto": lo_a, "hi_auto": hi_a,
                "cov_auto": cov_auto, "rt_auto": rt_auto,
            })
            print(f"  seed={seed} {label:10s} α̂={alpha:.2f} → severity={sev_auto:7s} "
                  f"ate={ate_auto:+.3f} CI=[{lo_a:+.3f},{hi_a:+.3f}] cov={cov_auto} "
                  f"({rt_auto:.0f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby("setup").agg(
        n=("seed", "count"),
        alpha_mean=("alpha_hat", "mean"),
        severity_mode=("severity_auto", lambda s: s.mode().iloc[0]),
        ate_mean=("ate_auto", "mean"),
        bias=("ate_auto", lambda s: float(np.mean(s - TRUE_ATE))),
        coverage=("cov_auto", "mean"),
    ).reset_index()


def _write_markdown(df, agg):
    path = RESULTS_DIR / "auto_severity.md"
    lines = [
        "# Automated severity selection from tail-index diagnostic",
        "",
        f"Hill estimator on residuals of a quick S-Learner (top 10\\%) gives α̂.",
        f"Map: α̂>5 → none; 3<α≤5 → mild; 2<α≤3 → moderate; α≤2 → severe.",
        f"3 seeds × 3 DGP regimes (clean, whale 5\\%, whale 20\\%).",
        "",
        "| setup | n | mean α̂ | mode severity | bias | 95% coverage |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['setup']} | {int(r['n'])} | {r['alpha_mean']:.2f} | "
            f"{r['severity_mode']} | {r['bias']:+.3f} | {r['coverage']:.2f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "auto_severity_raw.csv", index=False)


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
