"""Basis-ablation on the tail-heterogeneous DGP.

Reviewer concern: §5.5 shows intercept vs perfect-tail-indicator. That
contrast is binary; it does not show what happens under *graded*
basis misspecification. This script sweeps six bases on the same DGP
(\tau_bulk = 2, \tau_tail = 10, whale = |X_0| > 1.96) with Welsch
likelihood held fixed, 5 seeds.

Bases:
  1. intercept            — only mean CATE; severe underfit
  2. linear_X0            — [1, X_0]; smooth approx, wrong shape
  3. poly2                — [1, X_0, X_0^2]; smoother approx
  4. tail_t15             — [1, 1(|X_0|>1.5)]; too-liberal threshold
  5. tail_t196            — [1, 1(|X_0|>1.96)]; correct threshold
  6. tail_t25             — [1, 1(|X_0|>2.5)]; too-strict threshold

Metrics per seed:
  - PEHE on full population
  - eps_ATE(mixed), eps_ATE(whale), eps_ATE(bulk)
  - coverage of true mixed ATE and of true whale ATE

Usage:  python -m benchmarks.run_basis_ablation --seeds 5
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


def dgp(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, 5))
    whale = (np.abs(X[:, 0]) > WHALE_CUT).astype(float)
    tau = TAU_BULK * (1 - whale) + TAU_TAIL * whale
    eps = rng.normal(0.0, 1.0, size=N)
    Y0 = X[:, 0] + 0.5 * X[:, 1] + eps
    Y1 = Y0 + tau
    pi = np.clip(1.0 / (1.0 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
    W = rng.binomial(1, pi).astype(int)
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, tau, whale


def make_basis(X, kind):
    x0 = X[:, 0]
    if kind == "intercept":
        return np.ones((len(X), 1))
    if kind == "linear_X0":
        return np.column_stack([np.ones(len(X)), x0])
    if kind == "poly2":
        return np.column_stack([np.ones(len(X)), x0, x0 * x0])
    if kind == "tail_t15":
        return np.column_stack([np.ones(len(X)), (np.abs(x0) > 1.5).astype(float)])
    if kind == "tail_t196":
        return np.column_stack([np.ones(len(X)), (np.abs(x0) > 1.96).astype(float)])
    if kind == "tail_t25":
        return np.column_stack([np.ones(len(X)), (np.abs(x0) > 2.5).astype(float)])
    raise ValueError(kind)


BASES = ["intercept", "linear_X0", "poly2", "tail_t15", "tail_t196", "tail_t25"]


def fit_once(X, Y, W, basis_kind, seed):
    X_infer = make_basis(X, basis_kind)
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        nuisance_method="xgboost",
        n_splits=2,
        num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34, use_student_t=True,
        mad_rescale=False, random_state=seed,
    )
    model.fit(X, Y, W, X_infer=X_infer)
    cate, lo, hi = model.predict(X_new_infer=X_infer)
    return np.asarray(cate).flatten(), np.asarray(lo).flatten(), np.asarray(hi).flatten()


def _run(seeds):
    rows = []
    for seed in seeds:
        X, Y, W, tau, whale = dgp(seed)
        whale_mask = whale.astype(bool)
        true_mixed = float(np.mean(tau))
        for b in BASES:
            t0 = time.time()
            try:
                tau_hat, lo, hi = fit_once(X, Y, W, b, seed)
            except Exception as e:
                print(f"  [{b:12s}] seed={seed} ERR {e}")
                continue
            rt = time.time() - t0
            pehe = float(np.sqrt(np.mean((tau_hat - tau) ** 2)))
            eps_mixed = abs(float(np.mean(tau_hat)) - true_mixed)
            eps_whale = abs(float(np.mean(tau_hat[whale_mask])) - TAU_TAIL) if whale_mask.any() else float("nan")
            eps_bulk = abs(float(np.mean(tau_hat[~whale_mask])) - TAU_BULK)
            cov_mixed = int(
                float(np.mean(lo)) <= true_mixed <= float(np.mean(hi)))
            cov_whale = int(
                float(np.mean(lo[whale_mask])) <= TAU_TAIL <= float(np.mean(hi[whale_mask]))
            ) if whale_mask.any() else 0
            rows.append({
                "seed": seed, "basis": b,
                "pehe": pehe,
                "eps_mixed": eps_mixed,
                "eps_whale": eps_whale,
                "eps_bulk": eps_bulk,
                "cov_mixed": cov_mixed,
                "cov_whale": cov_whale,
                "runtime": rt,
            })
            print(f"  [{b:12s}] seed={seed} "
                  f"PEHE={pehe:.3f} eps_mix={eps_mixed:.3f} "
                  f"eps_whale={eps_whale:.3f} cov_whale={cov_whale} "
                  f"({rt:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.groupby("basis").agg(
        n=("seed", "count"),
        pehe=("pehe", "mean"),
        pehe_std=("pehe", "std"),
        eps_mixed=("eps_mixed", "mean"),
        eps_whale=("eps_whale", "mean"),
        eps_bulk=("eps_bulk", "mean"),
        cov_mixed=("cov_mixed", "mean"),
        cov_whale=("cov_whale", "mean"),
    ).reindex(BASES)


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "basis_ablation.md"
    lines = [
        "# Basis-sensitivity ablation on the tail-heterogeneous DGP",
        "",
        f"DGP: tau_bulk = 2, tau_tail = 10, whale = 1(|X_0|>{WHALE_CUT}), "
        f"N = {N}, seeds = {list(seeds)}.",
        "Welsch likelihood held fixed; only the CATE basis varies.",
        "",
        "| basis | n | PEHE | std | ε(mixed) | ε(whale) | ε(bulk) | cov(mix) | cov(whale) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in agg.iterrows():
        if pd.isna(r['n']):
            continue
        lines.append(
            f"| {name} | {int(r['n'])} | {r['pehe']:.3f} | {r['pehe_std']:.3f} | "
            f"{r['eps_mixed']:.3f} | {r['eps_whale']:.3f} | {r['eps_bulk']:.3f} | "
            f"{r['cov_mixed']:.2f} | {r['cov_whale']:.2f} |"
        )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "basis_ablation_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'basis_ablation_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, seeds)


if __name__ == "__main__":
    main()
