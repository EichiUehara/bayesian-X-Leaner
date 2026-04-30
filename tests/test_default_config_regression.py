"""
Regression test pinning the production default configuration.

The library's defaults (robust=True, CatBoost+Huber nuisance, Welsch
Bayesian likelihood) should produce reasonable CATE estimates on
clean synthetic data without requiring the user to pass any
arguments. This test guards against silent refactors that would
re-flip the defaults and silently hurt users.

The thresholds are deliberately loose — we're checking that the
default "just works", not that it's optimal. Tightening these
would make the test fragile to MCMC-seed variation.
"""

import numpy as np
import pytest
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner


def _make_simple_cate_dgp(n=300, p=5, true_ate=2.0, seed=0):
    """Linear outcome with constant treatment effect — the simplest
    DGP that still exercises nuisance cross-fitting + Bayesian MCMC."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, p))
    pi = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    W = rng.binomial(1, np.clip(pi, 0.1, 0.9))
    Y0 = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, size=n)
    Y1 = Y0 + true_ate
    Y = np.where(W == 1, Y1, Y0)
    tau_true = np.full(n, true_ate)
    return X, Y, W, tau_true


def test_default_config_works_on_clean_data():
    """With no arguments, the library should estimate ATE to within
    0.3 of truth on a simple linear DGP."""
    X, Y, W, tau_true = _make_simple_cate_dgp(n=300, seed=0)

    model = TargetedBayesianXLearner(num_warmup=200, num_samples=400)
    model.fit(X, Y, W)
    cate, ci_lo, ci_hi = model.predict(X)

    ate_hat = float(np.mean(cate))
    ate_err = abs(ate_hat - float(np.mean(tau_true)))
    assert ate_err < 0.3, (
        f"Default config ATE error {ate_err:.3f} exceeds 0.3 — "
        "check that robust=True and CatBoost-Huber defaults haven't "
        "regressed."
    )

    ci_width = float(np.mean(ci_hi - ci_lo))
    assert 0.05 < ci_width < 5.0, (
        f"Default CI width {ci_width:.3f} looks pathological "
        "(should be O(0.1-1) for N=300 at this DGP scale)."
    )


def test_default_config_produces_robust_flag():
    """Pin the specific defaults that the production narrative
    depends on. If these change, the Summary.md / README.md story
    needs updating too."""
    model = TargetedBayesianXLearner()
    assert model.robust is True, (
        "robust=True is the production default (Welsch likelihood "
        "handles real-world outcome tails; robust=False failed IHDP "
        "at sqrt(PEHE) 5.22 in the §17 benchmark)."
    )
    assert model.nuisance_method == "catboost", (
        "CatBoost is the production default nuisance learner "
        "(§16: Huber(delta=0.5) loss handles whale-grade "
        "contamination)."
    )
    assert model.use_student_t is True, (
        "Student-T prior is the production default when robust=True."
    )


def test_default_config_posterior_covers_true_ate():
    """Weak coverage check: a 95% credible interval on the average
    CATE should contain the true ATE on clean data. This is a
    calibration floor, not a tight test."""
    X, Y, W, tau_true = _make_simple_cate_dgp(n=300, seed=1)

    model = TargetedBayesianXLearner(num_warmup=200, num_samples=400)
    model.fit(X, Y, W)
    cate, ci_lo, ci_hi = model.predict(X)

    ate_lo = float(np.mean(ci_lo))
    ate_hi = float(np.mean(ci_hi))
    true_ate = float(np.mean(tau_true))

    assert ate_lo <= true_ate <= ate_hi, (
        f"Default posterior 95% CI on ATE [{ate_lo:.3f}, {ate_hi:.3f}] "
        f"does not contain true ATE {true_ate:.3f} — coverage "
        "calibration has regressed."
    )


# ---------------------------------------------------------------------------
# contamination_severity enum: Huber's (1964) minimax-delta prescription.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity,expected_nuisance,expected_delta", [
    ("none",     "xgboost",  None),
    ("mild",     "catboost", "1.345"),
    ("moderate", "catboost", "1.0"),
    ("severe",   "catboost", "0.5"),
])
def test_contamination_severity_maps_to_huber_delta(
    severity, expected_nuisance, expected_delta
):
    """Each severity level must map to the Huber-1964 minimax delta
    documented in EXTENSIONS.md §17.1. Guards against silent reshuffling
    of the (severity -> delta) table, which is the library's principled
    API for exposing Huber's efficiency-robustness tradeoff."""
    from sert_xlearner.models.nuisance import CATBOOST_AVAILABLE

    model = TargetedBayesianXLearner(contamination_severity=severity)

    if expected_nuisance == "catboost" and not CATBOOST_AVAILABLE:
        pytest.skip("CatBoost not importable; silent fallback is tested elsewhere")

    assert model.nuisance_method == expected_nuisance, (
        f"severity={severity!r} produced nuisance_method="
        f"{model.nuisance_method!r}, expected {expected_nuisance!r}"
    )
    if expected_delta is None:
        assert "loss_function" not in model.outcome_model_params, (
            f"severity='none' must not impose a Huber loss; got "
            f"{model.outcome_model_params!r}"
        )
    else:
        loss = model.outcome_model_params.get("loss_function", "")
        assert f"Huber:delta={expected_delta}" == loss, (
            f"severity={severity!r} must set loss_function="
            f"Huber:delta={expected_delta}, got {loss!r}"
        )


def test_contamination_severity_rejects_unknown_label():
    """Typos must raise — silent fallback would defeat the principled
    API and leave users with an unexpected delta."""
    with pytest.raises(ValueError, match="contamination_severity"):
        TargetedBayesianXLearner(contamination_severity="extreme")


def test_contamination_severity_defers_to_explicit_params():
    """When a caller passes outcome_model_params explicitly, severity
    must not override them. The enum is a convenience, not a lock."""
    custom = {"depth": 6, "iterations": 50, "loss_function": "RMSE"}
    model = TargetedBayesianXLearner(
        contamination_severity="severe",
        outcome_model_params=custom,
    )
    assert model.outcome_model_params == custom, (
        "Explicit outcome_model_params must win over severity preset."
    )
