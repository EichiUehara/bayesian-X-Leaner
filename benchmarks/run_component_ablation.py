"""
Component ablation — decomposes the "robust" machinery into its pieces.

Our consolidated-picture claim (EXTENSIONS.md):
  "RX-Learner's robustness is a specific combination of
   (a) DR pseudo-outcome targeting + (b) Welsch loss + (c) Student-T likelihood,
   and each element is necessary."

DR targeting (a) is baked into every variant — it's the whole architecture.
So this ablation decomposes (b) and (c).

CODE-LEVEL FINDING (documented on first reading of bayesian.py):
  When `robust=True`, the model uses `numpyro.factor(welsch_loss(...))`
  directly and never reads `use_student_t`. So the "Welsch + Student-T"
  combined variant advertised in docs is not actually what runs — the
  architecture has three discrete modes, not four.

Variants tested (what the code actually produces):
  | Name             | robust | use_student_t | What actually runs              |
  |------------------|--------|---------------|---------------------------------|
  | Gaussian (std)   | False  | False         | Normal likelihood, L²           |
  | Student-T only   | False  | True          | Student-T likelihood, L²        |
  | Welsch only      | True   | (ignored)     | Welsch factor, no likelihood    |

Predicted outcomes:
  - Gaussian fails on whale (baseline, RMSE ~20).
  - Welsch handles whale via bounded-influence loss (RMSE ~0.08).
  - Student-T handles whale via thick posterior tails (intermediate).
  - If Student-T ≈ Welsch → either is sufficient; docs can drop one claim.
  - If Welsch ≫ Student-T → Welsch is the load-bearing piece.
  - If Student-T ≫ Welsch → Student-T is.

Usage:
    python -m benchmarks.run_component_ablation --seeds 15
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import standard_dgp, whale_dgp, sharp_null_dgp


RESULTS_DIR = Path(__file__).parent / "results"


DGPS = {
    "standard":   standard_dgp,
    "whale":      whale_dgp,
    "sharp_null": sharp_null_dgp,
}

VARIANTS = {
    "Gaussian (std)":   dict(robust=False, use_student_t=False),
    "Student-T only":   dict(robust=False, use_student_t=True),
    "Welsch only":      dict(robust=True,  use_student_t=False),
}


def fit_rx(X, Y, W, variant_params, nuisance="xgb_mse"):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    if nuisance == "catboost_huber":
        kwargs = dict(
            outcome_model_params={"depth": 4, "iterations": 150,
                                   "loss_function": "Huber:delta=0.5"},
            propensity_model_params={"depth": 4, "iterations": 150},
            nuisance_method="catboost",
        )
    else:
        kwargs = dict(
            outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
            nuisance_method="xgboost",
        )
    model = TargetedBayesianXLearner(
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, use_overlap=False, random_state=42,
        **kwargs, **variant_params,
    )
    model.fit(X, Y, W)
    mean, lo, hi = model.predict()
    return float(np.asarray(mean).mean()), float(np.asarray(lo).mean()), float(np.asarray(hi).mean())


def _run(seeds, nuisance="xgb_mse"):
    rows = []
    for dgp_name, dgp_fn in DGPS.items():
        for seed in seeds:
            X, Y, W, tau = dgp_fn(seed=seed)
            tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
            for name, params in VARIANTS.items():
                t0 = time.time()
                try:
                    ate, lo, hi = fit_rx(X, Y, W, params, nuisance=nuisance)
                    cov = int(lo <= tau_true <= hi)
                    err = None
                except Exception as e:
                    ate = lo = hi = float("nan"); cov = 0; err = str(e)
                rt = time.time() - t0
                rows.append({
                    "dgp": dgp_name, "variant": name, "seed": seed,
                    "tau_true": tau_true, "ate": ate, "ci_lo": lo, "ci_hi": hi,
                    "covered": cov, "runtime": rt, "error": err,
                })
                print(f"  {dgp_name:<10}  seed={seed:<2}  {name:<16}  "
                      f"ate={ate:+.3f}  CI=[{lo:+.2f},{hi:+.2f}]  "
                      f"cov={'Y' if cov else 'N'}  ({rt:.1f}s)"
                      + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["dgp", "variant"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "mean_ate": g["ate"].mean(),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
                  "mean_runtime": g["runtime"].mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds, nuisance="xgb_mse"):
    suffix = "_catboost_huber" if nuisance == "catboost_huber" else ""
    path = RESULTS_DIR / f"component_ablation{suffix}.md"
    lines = [
        "# Component ablation — is each robust piece necessary?",
        "",
        "Decomposes the \"RX-Learner (robust)\" claim that Welsch loss and "
        "Student-T likelihood are each load-bearing. Inspection of "
        "`sert_xlearner/inference/bayesian.py` reveals **the architecture has "
        "three modes, not four** — when `robust=True`, the model uses "
        "`numpyro.factor(welsch_loss)` directly and ignores `use_student_t`. "
        "So the previously-documented \"Welsch + Student-T\" variant was never "
        "actually tested; this ablation tests the three modes the code does "
        "produce.",
        "",
        f"Seeds: {list(seeds)}. DR pseudo-outcome targeting is on in every "
        "variant (it's architectural, not optional).",
        "",
    ]
    for dgp in agg["dgp"].unique():
        sub = agg[agg["dgp"] == dgp].sort_values("rmse")
        tau_true = df[df["dgp"] == dgp]["tau_true"].iloc[0]
        lines += [
            f"## DGP: `{dgp}` (true ATE = {tau_true:.2f})",
            "",
            "| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['variant']} | {int(r['n'])} | "
                f"{r['mean_ate']:+.3f} | {r['bias']:+.3f} | {r['rmse']:.3f} | "
                f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} | "
                f"{r['mean_runtime']:.2f} |"
            )
        lines.append("")
    lines += [
        "## Interpretation guide",
        "",
        "- **If Welsch and Student-T both handle whale ≈ equally well** → "
        "either is sufficient and our docs over-claim by calling both "
        "necessary.",
        "- **If only Welsch handles whale** → Welsch is the load-bearing "
        "robustness piece; Student-T's contribution is narrower CIs via thick-"
        "tail posterior, not point-estimate robustness.",
        "- **If neither alone handles whale well but the implicit \"combined\" "
        "variant we can't directly test would** → motivates a code change to "
        "enable true Welsch+StudentT combined mode.",
    ]
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    csv_path = RESULTS_DIR / f"component_ablation_raw{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--nuisance", choices=["xgb_mse", "catboost_huber"],
                    default="xgb_mse",
                    help="Nuisance learner: xgb_mse (legacy) or catboost_huber (§16 default)")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds, nuisance=args.nuisance)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, seeds, nuisance=args.nuisance)


if __name__ == "__main__":
    main()
