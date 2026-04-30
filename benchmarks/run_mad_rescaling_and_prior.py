"""MAD rescaling × prior-scale sweep — revisits §14.

The initial §14 conclusion ("prior not load-bearing at high whale density")
was produced with the production default `c_whale` rescaling: inside
`TargetedBayesianXLearner.fit`, the configured `c_whale=1.34` is
multiplied by `mad_scaled = MAD(all pseudo-outcomes) / 0.6745`. When
whales inflate pseudo-outcome spread, MAD is itself contaminated — at
density 10 % it rises to ~1100 — so the effective Welsch constant
becomes ~1500. With c ~1500, Welsch no longer clips whales and the
pseudo-likelihood *is* peaked at the biased value, hiding the
contribution of the prior.

This sweep factors the two knobs:

  - `mad_rescale` ∈ {on (default), off (c_whale used raw)}
  - `prior_scale` ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 10.0}
  - density ∈ {5 %, 10 %, 20 %}, N=1000, robust only, 8 seeds

When `mad_rescale=off` we bypass the wrapper's rescaling by running the
three phases directly.

Usage:
    python -m benchmarks.run_mad_rescaling_and_prior --seeds 8
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dgps import whale_dgp
from sert_xlearner.models.nuisance import NuisanceEstimator
from sert_xlearner.core.orthogonalization import impute_and_debias
from sert_xlearner.inference.bayesian import BayesianMCMC


RESULTS_DIR = Path(__file__).parent / "results"
N = 1000
DENSITIES = [0.05, 0.10, 0.20]
PRIOR_SCALES = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
MAD_MODES = ["on", "off"]


def fit_direct(X, Y, W, prior_scale, mad_mode, c_whale_base=1.34):
    """Run all three phases directly to expose the MAD rescaling knob."""
    nuis = NuisanceEstimator(
        {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        {"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, random_state=42, method="xgboost",
    )
    mu0, mu1, pi = nuis.fit_predict(X, Y, W)
    treated_mask, control_mask, D1, D0, W_D1, W_D0 = impute_and_debias(
        Y, W, mu0, mu1, pi, robust=True,
        tail_threshold=None, tail_alpha=None, use_overlap=False,
    )

    all_res = np.concatenate([D1, D0])
    mad = np.median(np.abs(all_res - np.median(all_res)))
    mad_scaled = mad / 0.6745 if mad > 1e-6 else 1.0

    if mad_mode == "on":
        c_eff = c_whale_base * mad_scaled
    elif mad_mode == "off":
        c_eff = c_whale_base
    else:
        raise ValueError(mad_mode)

    X_infer = np.ones((X.shape[0], 1))
    X_D1 = X_infer[treated_mask]
    X_D0 = X_infer[control_mask]

    mcmc = BayesianMCMC(
        num_warmup=400, num_samples=800, num_chains=2, random_seed=42,
        prior_scale=prior_scale, robust=True, c_whale=c_eff,
    )
    mcmc.sample_posterior(X_D1, X_D0, D1, D0, W_D1, W_D0)
    beta = np.asarray(mcmc.mcmc_samples["beta"]).squeeze()
    ate = float(np.mean(beta))
    lo, hi = np.percentile(beta, [2.5, 97.5])
    return ate, float(lo), float(hi), float(c_eff), float(mad_scaled)


def _run(seeds):
    rows = []
    for density in DENSITIES:
        n_whales = max(1, int(round(density * N)))
        for seed in seeds:
            X, Y, W, tau = whale_dgp(N=N, n_whales=n_whales, seed=seed)
            tau_true = float(np.mean(tau) if hasattr(tau, "__len__") else tau)
            for mad_mode in MAD_MODES:
                for prior_scale in PRIOR_SCALES:
                    t0 = time.time()
                    try:
                        ate, lo, hi, c_eff, mad_scaled = fit_direct(
                            X, Y, W, prior_scale, mad_mode
                        )
                        cov = int(lo <= tau_true <= hi)
                        err = None
                    except Exception as e:
                        ate = lo = hi = c_eff = mad_scaled = float("nan")
                        cov = 0; err = str(e)
                    rt = time.time() - t0
                    rows.append({
                        "density": density, "n_whales": n_whales, "seed": seed,
                        "mad_mode": mad_mode, "prior_scale": prior_scale,
                        "c_eff": c_eff, "mad_scaled": mad_scaled,
                        "tau_true": tau_true, "ate": ate, "ci_lo": lo, "ci_hi": hi,
                        "covered": cov, "runtime": rt, "error": err,
                    })
                    print(
                        f"  dens={density:<5} seed={seed:<2} mad={mad_mode:<3} "
                        f"σ={prior_scale:<5} c_eff={c_eff:>8.2f} "
                        f"ATE={ate:+.3f} cov={'Y' if cov else 'N'} ({rt:.1f}s)"
                        + (f" ERR={err}" if err else "")
                    )
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["ate"]).groupby(["density", "mad_mode", "prior_scale"])
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "c_eff_median": g["c_eff"].median(),
                  "bias": g["ate"].mean() - g["tau_true"].mean(),
                  "rmse": float(np.sqrt(((g["ate"] - g["tau_true"]) ** 2).mean())),
                  "median_ate": g["ate"].median(),
                  "coverage": g["covered"].mean(),
                  "mean_ci_width": (g["ci_hi"] - g["ci_lo"]).mean(),
              })).reset_index())


def _write_markdown(df, agg, seeds):
    path = RESULTS_DIR / "mad_rescaling_and_prior.md"
    lines = [
        "# MAD rescaling × prior-scale — revisiting §14",
        "",
        f"N fixed at {N}. Whale density ∈ {[f'{d*100:g}%' for d in DENSITIES]}, "
        f"`mad_rescale` ∈ {MAD_MODES}, `prior_scale` ∈ {PRIOR_SCALES}. "
        f"Robust (Welsch) variant only, base `c_whale=1.34`. Seeds: {list(seeds)}.",
        "",
        "Motivation. The initial §14 conclusion (*prior is not load-bearing*) "
        "was produced with the production-default MAD rescaling of `c_whale` "
        "(lines 85-92 of [targeted_bayesian_xlearner.py](../../sert_xlearner/targeted_bayesian_xlearner.py)). "
        "When whales inflate pseudo-outcome spread, MAD rises to ~1100, so "
        "the effective Welsch constant becomes ~1500 — Welsch no longer "
        "clips whales and the pseudo-likelihood is itself peaked at the "
        "biased value, hiding any effect of the prior. Turning MAD "
        "rescaling off restores Welsch's clipping behaviour and lets the "
        "prior act.",
        "",
    ]
    for density in sorted(agg["density"].unique()):
        for mad_mode in MAD_MODES:
            sub = (agg[(agg["density"] == density) & (agg["mad_mode"] == mad_mode)]
                    .sort_values("prior_scale"))
            if sub.empty:
                continue
            c_med = sub["c_eff_median"].iloc[0]
            lines += [
                f"## density={density*100:g}%, MAD rescale **{mad_mode}** "
                f"(effective c ≈ {c_med:.2f})",
                "",
                "| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['prior_scale']} | {int(r['n'])} | "
                    f"{r['bias']:+.3f} | {r['rmse']:.3f} | "
                    f"{r['median_ate']:+.3f} | "
                    f"{r['coverage']:.2f} | {r['mean_ci_width']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "mad_rescaling_and_prior_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'mad_rescaling_and_prior_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    df = _run(seeds)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, seeds)


if __name__ == "__main__":
    main()
