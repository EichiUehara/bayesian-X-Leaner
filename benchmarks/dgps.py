"""
Canonical data-generating processes for causal inference benchmarks.

Every DGP returns (X, Y, W, tau_true) and documents the failure mode it is
designed to stress.  Seeds are explicit to make Monte Carlo replication
deterministic.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Standard clean DGP — baseline for competitiveness checks
# ---------------------------------------------------------------------------

def standard_dgp(N: int = 600, P: int = 8, tau: float = 2.0, seed: int = 0):
    """
    Linear confounding, balanced treatment, Gaussian noise.

    Propensity:   pi(x) = σ(x₀),  clipped to [0.15, 0.85]
    Outcome:      Y₀ = 1.5·x₀ + N(0, 0.5),  Y₁ = Y₀ + tau

    Every well-implemented estimator should achieve low bias here.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (N, P))
    pi = np.clip(1 / (1 + np.exp(-X[:, 0])), 0.15, 0.85)
    W = rng.binomial(1, pi)
    Y0 = 1.5 * X[:, 0] + rng.normal(0, 0.5, N)
    Y = np.where(W == 1, Y0 + tau, Y0)
    return X, Y, W, tau


# ---------------------------------------------------------------------------
# Whale DGP — outlier contamination
# ---------------------------------------------------------------------------

def whale_dgp(
    N: int = 600,
    P: int = 8,
    tau: float = 2.0,
    n_whales: int | None = None,
    whale_size: float = 5000.0,
    seed: int = 0,
):
    """
    Standard DGP with ``n_whales`` control-group whales.

    Default ``n_whales = max(1, N // 100)`` (~1 % of N).

    Why multiple whales?
    --------------------
    A single whale can be isolated by XGBoost in a leaf of size 1 and does
    not propagate through the S-Learner (accidental robustness).  With k > 1
    whales, tree isolation fails because (a) splits must balance many outliers
    against many inliers, and (b) pseudo-outcome estimators that average over
    control units see the outlier influence directly.
    """
    X, Y, W, tau = standard_dgp(N, P, tau, seed)
    if n_whales is None:
        n_whales = max(1, N // 100)
    rng = np.random.RandomState(seed + 10_000)
    ctrl = np.where(W == 0)[0]
    if len(ctrl) < n_whales:
        n_whales = max(1, len(ctrl))
    whale_idx = rng.choice(ctrl, size=n_whales, replace=False)
    Y[whale_idx] += whale_size
    return X, Y, W, tau


# ---------------------------------------------------------------------------
# Imbalance DGP — 'Compassionate Use' propensity skew
# ---------------------------------------------------------------------------

def imbalance_dgp(
    N: int = 1000,
    P: int = 8,
    tau: float = 2.0,
    treatment_prob: float = 0.95,
    seed: int = 0,
):
    """
    Linear confounding, user-specified constant treatment probability.

    ``treatment_prob = 0.95`` yields ~50 control units from N=1000; the
    (1-W)/(1-pi) term in AIPW amplifies residuals by a factor of up to 20.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (N, P))
    pi = np.full(N, treatment_prob)
    W = rng.binomial(1, pi)
    Y0 = X[:, 0] + rng.normal(0, 0.5, N)
    Y = np.where(W == 1, Y0 + tau, Y0)
    return X, Y, W, tau


# ---------------------------------------------------------------------------
# Sharp-null DGP — confounding-leakage test
# ---------------------------------------------------------------------------

def heterogeneous_cate_dgp(N: int = 1000, P: int = 6, seed: int = 0):
    """
    DGP with a KNOWN heterogeneous treatment effect τ(x) = 2 + x₀.

    Returns (X, Y, W, tau_vec) where tau_vec is the per-unit true CATE.
    Used for PEHE evaluation rather than ATE-only benchmarks.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (N, P))
    pi = np.clip(1 / (1 + np.exp(-0.5 * X[:, 1])), 0.2, 0.8)
    W = rng.binomial(1, pi)
    tau_vec = 2.0 + X[:, 0]
    Y0 = 1.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, N)
    Y = Y0 + W * tau_vec
    return X, Y, W, tau_vec


def nonlinear_cate_dgp(N: int = 1000, P: int = 6, seed: int = 0):
    """
    DGP with a *nonlinear* heterogeneous treatment effect.

    τ(x) = 2 + sin(2·x₀)   — smooth, bounded, with curvature.

    Stresses the parametric CATE basis of RX-Learner: a [1, x₀] basis
    (what the previous CATE test used) is deliberately misspecified here.
    Methods that model CATE flexibly (EconML Forest, X-Learner with
    flexible base) should do better than linear-basis RX-Learner.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (N, P))
    pi = np.clip(1 / (1 + np.exp(-0.5 * X[:, 1])), 0.2, 0.8)
    W = rng.binomial(1, pi)
    tau_vec = 2.0 + np.sin(2 * X[:, 0])
    Y0 = 1.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, N)
    Y = Y0 + W * tau_vec
    return X, Y, W, tau_vec


def sharp_null_dgp(N: int = 1500, P: int = 10, seed: int = 0):
    """
    Non-linear confounding with zero true treatment effect (τ = 0).

    Estimators that leak confounding through imperfect nuisance estimation
    will hallucinate a non-zero effect.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (N, P))
    logit = 0.8 * X[:, 0] + 0.6 * X[:, 1]
    pi = np.clip(1 / (1 + np.exp(-logit)), 0.10, 0.90)
    W = rng.binomial(1, pi)
    X1c = np.clip(X[:, 1], -2.5, 2.5)
    Y0 = (
        np.sin(X[:, 0])
        + 0.5 * X1c ** 2
        - 0.4 * X[:, 0] * X[:, 1]
        + 0.3 * X[:, 2]
        + rng.normal(0, 0.4, N)
    )
    Y = Y0.copy()  # Y1 = Y0 under the sharp null
    return X, Y, W, 0.0


# ---------------------------------------------------------------------------
# Registry for iteration in benchmark runners
# ---------------------------------------------------------------------------

DGPS = {
    "standard":    standard_dgp,
    "whale":       whale_dgp,
    "imbalance":   imbalance_dgp,
    "sharp_null":  sharp_null_dgp,
    "heterogeneous_cate": heterogeneous_cate_dgp,
    "nonlinear_cate":     nonlinear_cate_dgp,
}


def load_ihdp(replication: int = 1, data_dir=None):
    """
    Load one replication of the IHDP semi-synthetic benchmark.

    Source: Hill (2011) / CEVAE preprocessed CSVs. Each CSV has columns:
        [treatment, y_factual, y_cfactual, mu0, mu1, x0, ..., x24]

    Returns: (X, Y, W, tau_vec) — tau_vec = mu1 − mu0 (per-unit ground truth).
    """
    from pathlib import Path
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    path = Path(data_dir) / f"ihdp_{replication}.csv"
    arr = np.loadtxt(path, delimiter=",")
    W = arr[:, 0].astype(int)
    Y = arr[:, 1].astype(float)
    # arr[:, 2] is y_cfactual — unobserved, not used for fitting
    mu0 = arr[:, 3].astype(float)
    mu1 = arr[:, 4].astype(float)
    X = arr[:, 5:].astype(float)
    tau_vec = mu1 - mu0
    return X, Y, W, tau_vec
