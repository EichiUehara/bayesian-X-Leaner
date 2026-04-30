"""Extended IHDP: cheap baselines on 25 replications.

Reviewer concern: IHDP at 5 reps is underpowered. We can't extend
BCF (~10 min/rep × 25 = ~4 h) without major compute, but we can
extend the cheaper estimators (S/T/X-learner, EconML, RX-Learner,
Huber-DR) to 25 reps to improve significance claims for these
specific contrasts.

Usage: python -u -m benchmarks.run_ihdp_extended --replications 25
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.dgps import load_ihdp
from benchmarks.run_ihdp_benchmark import (
    cate_s_learner, cate_t_learner, cate_x_learner, cate_econml_forest,
    cate_huber_dr, cate_rx_learner,
)


RESULTS_DIR = Path(__file__).parent / "results"
ESTIMATORS = {
    "S-Learner":             cate_s_learner,
    "T-Learner":             cate_t_learner,
    "X-Learner (std)":       cate_x_learner,
    "EconML Forest":         cate_econml_forest,
    "Huber-DR (point)":      cate_huber_dr,
    "RX-Learner (robust)":   lambda X, Y, W: cate_rx_learner(X, Y, W, robust=True),
}


def _pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def _ate_err(tau_hat, tau):
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def _run(reps):
    rows = []
    for rep in reps:
        X, Y, W, tau = load_ihdp(rep)
        for name, fn in ESTIMATORS.items():
            t0 = time.time()
            try:
                tau_hat = fn(X, Y, W)
                pehe = _pehe(tau_hat, tau); ate_err = _ate_err(tau_hat, tau); err = None
            except Exception as e:
                pehe = ate_err = float("nan"); err = str(e)
            rt = time.time() - t0
            rows.append({
                "estimator": name, "replication": rep,
                "pehe": pehe, "ate_err": ate_err, "runtime": rt,
            })
            print(f"  rep {rep} {name:25s} √PEHE={pehe:.3f} ε_ATE={ate_err:.3f} ({rt:.1f}s)")
    return pd.DataFrame(rows)


def _summarise(df):
    return df.dropna(subset=["pehe"]).groupby("estimator").agg(
        n=("replication", "count"),
        pehe_mean=("pehe", "mean"),
        pehe_std=("pehe", "std"),
        ate_err_mean=("ate_err", "mean"),
    ).sort_values("pehe_mean")


def _significance(df, baseline="RX-Learner (robust)"):
    """Welch's t-test of each estimator's PEHE against the baseline."""
    rows = []
    base = df[df["estimator"] == baseline]["pehe"].dropna().values
    for est in df["estimator"].unique():
        if est == baseline:
            continue
        other = df[df["estimator"] == est]["pehe"].dropna().values
        if len(other) < 2 or len(base) < 2:
            continue
        t, p = stats.ttest_ind(other, base, equal_var=False)
        rows.append({"vs_baseline": baseline, "estimator": est,
                     "mean_diff": float(np.mean(other) - np.mean(base)),
                     "t_stat": float(t), "p_value": float(p)})
    return pd.DataFrame(rows)


def _write_markdown(df, agg, sig):
    path = RESULTS_DIR / "ihdp_extended.md"
    lines = [
        "# Extended IHDP — cheap baselines on 25 replications",
        "",
        f"5 reps was the previous benchmark cadence (CEVAE preprocessing convention).",
        f"BART/BCF rows omitted (each rep takes ~10 min); other baselines run on {df['replication'].max()} reps.",
        "",
        "| Estimator | n | √PEHE | std(√PEHE) | ε_ATE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in agg.iterrows():
        lines.append(
            f"| {name} | {int(r['n'])} | "
            f"{r['pehe_mean']:.3f} | {r['pehe_std']:.3f} | "
            f"{r['ate_err_mean']:.3f} |"
        )
    if len(sig):
        lines += ["", "## Welch's t-test vs RX-Learner (robust)",
                  "", "| Estimator | mean diff | t | p-value |", "|---|---:|---:|---:|"]
        for _, r in sig.iterrows():
            lines.append(
                f"| {r['estimator']} | {r['mean_diff']:+.3f} | "
                f"{r['t_stat']:+.2f} | {r['p_value']:.3f} |"
            )
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "ihdp_extended_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replications", type=int, default=25)
    args = ap.parse_args()
    reps = list(range(1, args.replications + 1))
    df = _run(reps)
    agg = _summarise(df)
    sig = _significance(df)
    print("\n── Summary ──"); print(agg.to_string())
    print("\n── Significance vs RX-Learner (robust) ──"); print(sig.to_string(index=False))
    _write_markdown(df, agg, sig)


if __name__ == "__main__":
    main()
