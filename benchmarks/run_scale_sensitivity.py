"""Scale-sensitivity probe: Huber-δ on raw Y vs MAD-rescaled Y.

Demonstrates the §6.5 caveat that minimax-δ values assume standardised
residuals; on dollar-scale outcomes the absolute Huber threshold (here
δ=0.5 from severity=severe) over-suppresses legitimate variation.

Usage: python -u -m benchmarks.run_scale_sensitivity
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import pandas as pd
from pathlib import Path

from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


RESULTS = Path(__file__).parent / "results"


def generate_scaled_data(N=1000, scale=500, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, 5))
    pi = np.clip(1 / (1 + np.exp(-0.3 * X[:, 0])), 0.1, 0.9)
    W = rng.binomial(1, pi)
    tau = 2.0
    Y0 = X[:, 0] + rng.normal(0, 1, N)
    Y = Y0 + W * tau
    # 10% whales: Y shifted by +50 in unscaled units
    whales = rng.choice(N, size=int(N * 0.1), replace=False)
    Y[whales] += 50.0
    # Scale Y to dollar-like magnitude
    Y = Y * scale
    # True ATE on the scaled outcome is tau * scale
    return X, Y, W, tau * scale


def fit_one(X, Y, W, normalize=False, seed=0):
    if normalize:
        # MAD scale of Y
        y_scale = float(np.median(np.abs(Y - np.median(Y))) / 0.6745)
        Y_fit = Y / y_scale
    else:
        y_scale = 1.0
        Y_fit = Y
    model = TargetedBayesianXLearner(
        contamination_severity="severe",
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        random_state=seed,
    )
    model.fit(X, Y_fit, W)
    cate, lo_norm, hi_norm = model.predict()
    ate_norm = float(np.mean(cate))
    lo_norm = float(np.mean(lo_norm)); hi_norm = float(np.mean(hi_norm))
    return ate_norm * y_scale, lo_norm * y_scale, hi_norm * y_scale, y_scale


def main():
    X, Y, W, true_ate = generate_scaled_data()
    print(f"True ATE = {true_ate}, Y range = [{Y.min():.0f}, {Y.max():.0f}]")
    rows = []
    for normalize in [False, True]:
        ate, lo, hi, y_scale = fit_one(X, Y, W, normalize=normalize)
        bias = ate - true_ate
        rows.append({
            "normalize": normalize, "y_scale": y_scale,
            "ate": ate, "lo": lo, "hi": hi, "bias": bias,
        })
        print(f"  normalize={normalize}  Y/MAD scale={y_scale:.3f}  "
              f"ATE={ate:+.2f} (true {true_ate})  bias={bias:+.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "scale_sensitivity_raw.csv", index=False)
    print(f"wrote {RESULTS / 'scale_sensitivity_raw.csv'}")


if __name__ == "__main__":
    main()
