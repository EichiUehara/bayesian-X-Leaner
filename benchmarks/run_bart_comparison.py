"""
Focused Bayesian-baseline comparison: RX-Learner vs Causal BART (T-Learner).

Runs on standard + whale DGPs (Bayesian methods on the two most interpretable
DGPs: clean data + outlier contamination).  BART is the canonical Bayesian
causal estimator; this test answers "is the added Welsch + DR targeting
machinery earning its keep vs. plain Bayesian T-learning?"

Usage:
    python -m benchmarks.run_bart_comparison --seeds 5
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import standard_dgp, whale_dgp
from benchmarks.estimators import fit_causal_bart, fit_rx_learner


VARIANTS = {
    "Causal BART (T-Learner)": lambda X, Y, W: fit_causal_bart(
        X, Y, W, draws=300, tune=300, trees=30, chains=2),
    "RX-Learner (robust)": lambda X, Y, W: fit_rx_learner(
        X, Y, W, robust=True, use_student_t=True),
    "RX-Learner (std)": lambda X, Y, W: fit_rx_learner(X, Y, W, robust=False),
}

DGPS = {
    "standard": standard_dgp,
    "whale":    whale_dgp,
}


RESULTS_DIR = Path(__file__).parent / "results"


def _run(seeds):
    rows = []
    for dgp_name, dgp_fn in DGPS.items():
        for seed in seeds:
            X, Y, W, tau = dgp_fn(seed=seed)
            for name, fn in VARIANTS.items():
                r = fn(X, Y, W)
                covered = (r["ci_lo"] is not None and r["ci_hi"] is not None
                           and r["ci_lo"] <= tau <= r["ci_hi"])
                rows.append({
                    "dgp": dgp_name, "variant": name, "seed": seed,
                    "tau_true": tau, "ate": r["ate"],
                    "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                    "covered": bool(covered), "runtime": r["runtime"],
                    "error": r["error"],
                })
                ate_str = f"{r['ate']:+.3f}" if r["ate"] is not None else "None"
                print(f"  {dgp_name:<9} seed={seed} {name:<26} ate={ate_str} "
                      f"cov={'Y' if covered else 'N'} ({r['runtime']:.1f}s)"
                      + (f" ERR={r['error']}" if r["error"] else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    out = []
    for (dgp, variant), g in df.groupby(["dgp", "variant"]):
        ok = g.dropna(subset=["ate"])
        ates = ok["ate"].astype(float).values
        tau = ok["tau_true"].astype(float).values
        out.append({
            "dgp": dgp, "variant": variant, "n": len(g),
            "mean_ate": float(np.mean(ates)) if len(ates) else float("nan"),
            "bias":    float(np.mean(ates - tau)) if len(ates) else float("nan"),
            "rmse":    float(np.sqrt(np.mean((ates - tau) ** 2))) if len(ates) else float("nan"),
            "coverage": float(g["covered"].mean()),
            "mean_ci_width": float((g["ci_hi"] - g["ci_lo"]).mean()),
            "mean_runtime":  float(g["runtime"].mean()),
        })
    return pd.DataFrame(out).sort_values(["dgp", "rmse"])


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "bart_comparison.md"
    lines = [
        "# Bayesian baseline: RX-Learner vs Causal BART",
        "",
        f"Seeds: {list(seeds)}.  Causal BART = BART T-Learner "
        "(separate BART regressions for treated and control, ATE is "
        "posterior mean of μ̂₁ − μ̂₀).",
        "",
    ]
    for dgp in sorted(df["dgp"].unique()):
        tau = df[df["dgp"] == dgp]["tau_true"].iloc[0]
        lines += [
            f"## DGP: `{dgp}`   (true ATE = {tau})",
            "",
            "| Variant | n | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        sub = agg[agg["dgp"] == dgp]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['variant']} | {int(r['n'])} | "
                f"{r['mean_ate']:+.3f} | {r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} | "
                f"{r['mean_runtime']:.2f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    csv = RESULTS_DIR / "bart_comparison_raw.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")


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
