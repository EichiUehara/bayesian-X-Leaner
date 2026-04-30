"""
Smoke test for the pipeline comparison framework.

This file validates that every estimator runs without crashing on a small
single-seed DGP.  Single-seed results are too noisy for claims of the form
"architecture A beats architecture B" — for that, use the Monte-Carlo runner:

    python -m benchmarks.run_pipeline_comparison --seeds 30

The multi-seed runner writes:
    benchmarks/results/results_raw.csv         (per-run raw data)
    benchmarks/results/results_summary.md      (aggregated bias/RMSE/coverage)

Tolerances here are loose by design — they only catch code-level regressions
(imports, API drift, NaNs).  Numerical claims live in the Monte Carlo report.
"""

# Parallel-chain MCMC — must precede any jax / numpyro import
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import pytest

from benchmarks.dgps import standard_dgp, whale_dgp
from benchmarks.estimators import ESTIMATORS


# ---------------------------------------------------------------------------
# Per-estimator smoke test (clean DGP) — parametrised for clear failure attribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ESTIMATORS.keys()))
def test_estimator_runs_on_clean_dgp(name):
    """Each estimator must return a finite ATE on a small clean DGP."""
    est = ESTIMATORS[name]
    X, Y, W, tau = standard_dgp(N=300, P=5, tau=2.0, seed=0)
    result = est(X, Y, W)

    if result["error"] is not None:
        pytest.skip(f"{name} not available: {result['error']}")

    assert result["ate"] is not None, f"{name}: returned None ATE"
    assert abs(result["ate"]) < 50.0, (
        f"{name}: ATE={result['ate']:.3f} diverged on clean DGP — something is broken."
    )


# ---------------------------------------------------------------------------
# Regression guard — RX-Learner (robust) must survive a mild whale injection
# ---------------------------------------------------------------------------

def test_rx_learner_robust_survives_whale():
    """
    Single-seed regression guard: RX-Learner (robust) should recover the true
    ATE to within 1.0 under a 6-whale contamination with tau_true=2.0.

    Statistical claims across estimators live in the Monte Carlo runner.
    """
    X, Y, W, tau = whale_dgp(N=500, P=6, tau=2.0, n_whales=6, seed=0)
    res = ESTIMATORS["RX-Learner (robust)"](X, Y, W)
    assert res["ate"] is not None
    assert abs(res["ate"] - tau) < 1.0, (
        f"RX-Learner (robust) ATE={res['ate']:.3f}, |bias|={abs(res['ate']-tau):.3f}. "
        "Welsch MCMC failed to recover true CATE under multi-whale contamination."
    )


# ---------------------------------------------------------------------------
# Heavy Monte-Carlo benchmark — skipped by default
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_monte_carlo_pipeline_comparison():
    """
    Full 10-seed × all-DGPs × all-estimators run. Takes several minutes.

    Skip by default; invoke explicitly via:
        pytest tests/test_pipeline_comparison.py -m benchmark -s

    Writes benchmarks/results/results_summary.md on completion.
    """
    from benchmarks.run_pipeline_comparison import main as run_main
    run_main(["--seeds", "10"])
