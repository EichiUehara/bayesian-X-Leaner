"""Proper Hillstrom train-only fit + holdout evaluation.

The original §5.9 paragraph used a fit on the *full* dataset and
then evaluated the posterior on each half. Because the basis is
parametric, that gives identical train and holdout subgroup ATEs by
construction (consistency, not validation).

This script does a true holdout: fit only on training half, then
predict at holdout covariates. The cut for the high-history basis is
defined from the training half only.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from benchmarks.run_hillstrom import load_hillstrom

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    print("Loading Hillstrom..."); X, Y, W = load_hillstrom()
    n = len(X)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    train_idx, holdout_idx = idx[:n // 2], idx[n // 2:]

    # Pre-specified basis: top 10% by training-half history
    history_train = X[train_idx, 1]
    cut = float(np.percentile(history_train, 90))
    high_hist_full = (X[:, 1] > cut).astype(float).reshape(-1, 1)
    X_inf_full = np.column_stack([np.ones(n), high_hist_full])

    print(f"Fit on TRAIN ONLY (n={len(train_idx)}), evaluate on HOLDOUT (n={len(holdout_idx)})")
    print(f"High-history cut from train: history > {cut:.4f}")

    # Match the original §5.9 in-sample fit exactly: xgboost with no severity preset.
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        nuisance_method="xgboost", n_splits=2,
        num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34, use_student_t=True, mad_rescale=False,
        random_state=0,
    )
    # Fit only on train half
    model.fit(X[train_idx], Y[train_idx], W[train_idx],
              X_infer=X_inf_full[train_idx])

    # Predict at holdout covariates (and train, for comparison)
    cate_h, lo_h, hi_h = model.predict(X_new_infer=X_inf_full[holdout_idx])
    cate_t, lo_t, hi_t = model.predict(X_new_infer=X_inf_full[train_idx])

    rows = []
    for split_label, idxs, cate, lo, hi in [
        ("train",   train_idx,   cate_t, lo_t, hi_t),
        ("holdout", holdout_idx, cate_h, lo_h, hi_h),
    ]:
        cate = np.asarray(cate).flatten()
        lo = np.asarray(lo).flatten()
        hi = np.asarray(hi).flatten()
        hh_mask = high_hist_full[idxs, 0].astype(bool)
        for subgrp_label, mask in [
            ("all",       np.ones(len(idxs), dtype=bool)),
            ("high_hist", hh_mask),
            ("low_hist",  ~hh_mask),
        ]:
            if mask.sum() == 0:
                continue
            rows.append({
                "split": split_label, "subgroup": subgrp_label,
                "n": int(mask.sum()),
                "ate": float(np.mean(cate[mask])),
                "lo": float(np.mean(lo[mask])),
                "hi": float(np.mean(hi[mask])),
                "ci_width": float(np.mean(hi[mask] - lo[mask])),
            })
            print(f"  {split_label:7s} {subgrp_label:10s} n={mask.sum():6d} "
                  f"tau_hat={rows[-1]['ate']:+.4f} "
                  f"CI=[{rows[-1]['lo']:+.4f}, {rows[-1]['hi']:+.4f}]")

    out = RESULTS_DIR / "hillstrom_true_holdout_raw.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
