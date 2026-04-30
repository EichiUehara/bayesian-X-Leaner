"""
Sample-size scaling for RX-Learner (std and robust).

Verifies the implicit consistency claim: RMSE should decrease at the √N
rate as N grows, for both the clean (standard) DGP and the contaminated
(whale) DGP where only the robust variant is expected to converge.

Predictions:
  - RX-Learner (robust) on standard: RMSE ∝ 1/√N (classical rate).
  - RX-Learner (robust) on whale:    RMSE ∝ 1/√N — robustness does not
                                      break the rate; it just bounds the
                                      constant against outlier leakage.
  - RX-Learner (std) on standard:    RMSE ∝ 1/√N — but constant larger
                                      (heavy-tailed DR residuals).
  - RX-Learner (std) on whale:       No convergence — RMSE stays O(1)
                                      or worse as N grows (whales scale
                                      with N at 1% density).

Slope of log(RMSE) vs log(N) should be near −0.5 in the convergent cases.

Usage:
    python -m benchmarks.run_sample_size_scaling --seeds 8
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import standard_dgp, whale_dgp


RESULTS_DIR = Path(__file__).parent / "results"

DGPS = {"standard": standard_dgp, "whale": whale_dgp}
N_VALUES = [200, 500, 1000, 2000, 5000]
VARIANTS = {
    "RX-Learner (std)":    dict(robust=False, use_student_t=False),
    "RX-Learner (robust)": dict(robust=True,  use_student_t=False),
}


def fit_rx(X, Y, W, variant_params, nuisance="xgb_mse"):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    if nuisance == "catboost_huber":
        # §16 default: CatBoost + Huber(delta=0.5), depth 4, 150 iterations.
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
    mean, _, _ = model.predict()
    return float(np.asarray(mean).mean())


def _run(seeds, nuisance="xgb_mse"):
    rows = []
    for dgp_name, dgp_fn in DGPS.items():
        for N in N_VALUES:
            for seed in seeds:
                X, Y, W, tau = dgp_fn(N=N, seed=seed)
                tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
                for name, params in VARIANTS.items():
                    t0 = time.time()
                    try:
                        ate = fit_rx(X, Y, W, params, nuisance=nuisance)
                        err = None
                    except Exception as e:
                        ate = float("nan"); err = str(e)
                    rt = time.time() - t0
                    rows.append({
                        "dgp": dgp_name, "variant": name, "N": N, "seed": seed,
                        "tau_true": tau_true, "ate": ate,
                        "runtime": rt, "error": err,
                    })
                    print(f"  {dgp_name:<10}  N={N:<5} seed={seed:<2} "
                          f"{name:<22} ate={ate:+.3f} ({rt:.1f}s)"
                          + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["dgp", "variant", "N"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "mean_runtime": g["runtime"].mean(),
              })).reset_index())


def _fit_slope(Ns, rmses):
    """Slope of log(RMSE) vs log(N). Classical rate is −0.5."""
    Ns = np.asarray(Ns, dtype=float)
    rmses = np.asarray(rmses, dtype=float)
    mask = np.isfinite(rmses) & (rmses > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(Ns[mask]), np.log(rmses[mask]), 1)[0])


def _write_markdown(df, agg, seeds, nuisance="xgb_mse"):
    suffix = "_catboost_huber" if nuisance == "catboost_huber" else ""
    path = RESULTS_DIR / f"sample_size_scaling{suffix}.md"
    lines = [
        "# Sample-size scaling",
        "",
        f"Seeds: {list(seeds)}. N ∈ {N_VALUES}.",
        "",
        "Classical Bayesian consistency predicts RMSE ∝ 1/√N, i.e. a slope of "
        "**−0.5** in log-log. A slope near 0 means non-convergence.",
        "",
    ]
    for dgp in agg["dgp"].unique():
        lines += [
            f"## DGP: `{dgp}`",
            "",
            "| Variant | " + " | ".join(f"N={n}" for n in N_VALUES) + " | log-log slope |",
            "|---|" + "---:|" * (len(N_VALUES) + 1),
        ]
        for variant in agg["variant"].unique():
            sub = agg[(agg["dgp"] == dgp) & (agg["variant"] == variant)]\
                  .set_index("N").reindex(N_VALUES)
            rmses = sub["rmse"].to_list()
            slope = _fit_slope(N_VALUES, rmses)
            rmse_cells = " | ".join(
                (f"{v:.3f}" if np.isfinite(v) else "—") for v in rmses
            )
            lines.append(f"| {variant} | {rmse_cells} | {slope:+.2f} |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- **Slope near −0.5** → √N-consistent (canonical Bayesian rate).",
        "- **Slope near 0 or positive** → non-convergent; extra data does not "
        "help (expected for RX-Learner std on whale: 1 % whale density is "
        "preserved as N grows).",
    ]
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    csv_path = RESULTS_DIR / f"sample_size_scaling_raw{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
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
