"""
Targeted experiment: does `use_overlap=True` fix the 0.67 coverage gap
that RX-Learner (robust) shows on the imbalance DGP?

Runs three variants on imbalance_dgp only, multi-seed:

  - RX-Learner (robust)          — DR pseudo-outcomes, Welsch pseudo-likelihood
  - RX-Learner (robust+overlap)  — bounded overlap weights, Welsch pseudo-likelihood
  - RX-Learner (std)             — Gaussian likelihood, no overlap (baseline)

Writes benchmarks/results/overlap_experiment.md

Usage:
    python -m benchmarks.run_overlap_experiment --seeds 8
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import imbalance_dgp
from benchmarks.estimators import fit_rx_learner


VARIANTS = {
    "RX-Learner (robust)":         lambda X, Y, W: fit_rx_learner(
        X, Y, W, robust=True, use_student_t=True, use_overlap=False),
    "RX-Learner (robust+overlap)": lambda X, Y, W: fit_rx_learner(
        X, Y, W, robust=True, use_student_t=True, use_overlap=True),
    "RX-Learner (std)":            lambda X, Y, W: fit_rx_learner(
        X, Y, W, robust=False, use_student_t=False, use_overlap=False),
}


RESULTS_DIR = Path(__file__).parent / "results"


def _run(seeds):
    rows = []
    for seed in seeds:
        X, Y, W, tau = imbalance_dgp(seed=seed)
        for name, fn in VARIANTS.items():
            r = fn(X, Y, W)
            covered = (r["ci_lo"] is not None and r["ci_hi"] is not None
                       and r["ci_lo"] <= tau <= r["ci_hi"])
            rows.append({
                "variant": name, "seed": seed, "tau_true": tau,
                "ate": r["ate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                "covered": bool(covered), "runtime": r["runtime"],
                "error": r["error"],
            })
            ate_str = f"{r['ate']:+.3f}" if r["ate"] is not None else "None"
            lo = r["ci_lo"] if r["ci_lo"] is not None else float("nan")
            hi = r["ci_hi"] if r["ci_hi"] is not None else float("nan")
            print(f"  seed={seed}  {name:<32}  ate={ate_str}  "
                  f"CI=[{lo:+.2f},{hi:+.2f}]  cov={'Y' if covered else 'N'}  "
                  f"({r['runtime']:.1f}s)"
                  + (f"  ERR={r['error']}" if r["error"] else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    rows = []
    for name, g in df.groupby("variant"):
        ok = g.dropna(subset=["ate"])
        ates = ok["ate"].astype(float).values
        tau = ok["tau_true"].astype(float).values
        rows.append({
            "variant":       name,
            "n":             len(g),
            "mean_ate":      float(np.mean(ates)) if len(ates) else float("nan"),
            "bias":          float(np.mean(ates - tau)) if len(ates) else float("nan"),
            "rmse":          float(np.sqrt(np.mean((ates - tau) ** 2))) if len(ates) else float("nan"),
            "coverage":      float(g["covered"].mean()),
            "mean_ci_width": float((g["ci_hi"] - g["ci_lo"]).mean()),
            "mean_runtime":  float(g["runtime"].mean()),
        })
    return pd.DataFrame(rows).sort_values("rmse")


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "overlap_experiment.md"
    lines = [
        "# Overlap-weights experiment on imbalance DGP",
        "",
        f"DGP: `imbalance_dgp` (treatment_prob = 0.95, ~50 controls of 1000). "
        f"True ATE = 2.0.",
        f"Seeds: {list(seeds)}",
        "",
        "Tests whether `use_overlap=True` (bounded overlap weights, Li 2018) "
        "closes the coverage gap of the DR-AIPW variant.",
        "",
        "| Variant | n | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | "
            f"{r['mean_ate']:+.3f} | {r['bias']:+.3f} | {r['rmse']:.3f} | "
            f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} | "
            f"{r['mean_runtime']:.2f} |"
        )

    # Verdict
    lines += ["", "## Verdict", ""]
    rob = agg[agg["variant"] == "RX-Learner (robust)"].iloc[0]
    ov = agg[agg["variant"] == "RX-Learner (robust+overlap)"].iloc[0]
    delta = ov["coverage"] - rob["coverage"]
    if delta >= 0.1:
        verdict = (f"**Overlap weights improve coverage** by {delta:+.2f} "
                   f"({rob['coverage']:.2f} → {ov['coverage']:.2f}). "
                   "The limitation flagged in STABILITY_SUMMARY.md is resolvable.")
    elif delta >= 0:
        verdict = (f"Overlap weights give a **marginal** coverage change "
                   f"({delta:+.2f}). The limitation persists.")
    else:
        verdict = (f"Overlap weights **hurt** coverage by {-delta:.2f}. "
                   "The DR-AIPW default is preferable even on imbalance.")
    lines += [verdict, ""]

    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    csv = RESULTS_DIR / "overlap_experiment_raw.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string(index=False))
    _write_markdown(df, agg, seeds)


if __name__ == "__main__":
    main()
