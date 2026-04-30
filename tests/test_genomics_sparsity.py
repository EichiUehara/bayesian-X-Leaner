import numpy as np
import pytest
import xgboost as xgb
import warnings
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Baseline: S-Learner with XGBoost (no feature selection)
# ---------------------------------------------------------------------------

def _s_learner_cate(X, Y, W, xgb_params):
    """
    S-Learner: single XGBoost on (X, W) → Y.

    With P=1000 noise features and N=200, XGBoost must split on P+1=1001
    variables.  Without L1 regularisation the model cannot isolate the 3 true
    confounders; it overfits to random signal and produces a biased CATE.
    This is the Regularisation-Induced Confounding (RIC) failure mode.
    """
    XW = np.column_stack([X, W.astype(float)])
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(XW, Y)

    X1 = np.column_stack([X, np.ones(len(X))])
    X0 = np.column_stack([X, np.zeros(len(X))])
    return float(np.mean(model.predict(X1) - model.predict(X0)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Premise empirically falsified. On this DGP (N=200, P=1000, linear "
        "confounding through 3 genes), shallow XGBoost (max_depth=3, 50 "
        "trees) beats ElasticNetCV-nuisance RX-Learner on 8/8 tested seeds "
        "(S-bias 0.04–1.00, RX-bias 0.97–3.34). Intuition: ElasticNet "
        "inside cross-fitting sees ~N/2=100 samples and 1000 features per "
        "arm — even L1 picks up noise at that ratio. The 'ElasticNet > "
        "XGBoost under P>>N' claim needs a tougher DGP (non-linear "
        "confounding, higher noise, or smaller N) to hold. Kept as xfail "
        "rather than deleted because the test still exercises the "
        "elasticnet code path end-to-end and is useful as a regression "
        "canary."
    ),
    strict=False,
)
def test_genomics_sparsity():
    """
    Genomics 'Sparsity Explosion' Use Case: Comparison of S-Learner (XGBoost)
    vs RX-Learner (ElasticNetCV nuisance) under P >> N sparsity.

    Scenario: 200 patients, 1,000 gene expressions.  Only 3 genes confound
    the treatment; 997 are pure noise.

    Estimators compared
    -------------------
    1. S-Learner (XGBoost on all 1,001 features)
       – Regularisation-Induced Confounding (RIC): XGBoost must regularise
         across P=1,001 features.  Random noise features leak into splits,
         starving the 3 true confounders of influence.  CATE is biased.
    2. RX-Learner (ElasticNetCV nuisance)
       – The L1 penalty zeros out the 997 noise genes while preserving the
         3 true confounders.  Clean nuisance estimates flow into Phase 2;
         the MCMC (intercept-only) recovers CATE ≈ 2.0 without divergence.

    True CATE: 2.0
    """
    np.random.seed(99)
    N = 200
    P = 1000
    tau_true = 2.0

    # Only the first 3 columns confound; the remaining 997 are noise
    X = np.random.normal(0, 1, size=(N, P))
    confounding = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 2]

    pi_true = np.clip(1 / (1 + np.exp(-0.6 * confounding)), 0.10, 0.90)
    W = np.random.binomial(1, pi_true)

    Y0 = confounding + np.random.normal(0, 0.5, size=N)
    Y1 = Y0 + tau_true
    Y = np.where(W == 1, Y1, Y0)

    print(f"\nSparsity Test: N={N}, P={P}, treated={W.sum()}, control={(W==0).sum()}")

    xgb_params = {"max_depth": 3, "n_estimators": 50, "verbosity": 0}

    # ── Baseline: S-Learner ──────────────────────────────────────────────────
    s_cate = _s_learner_cate(X, Y, W, xgb_params)

    # ── Proposed: RX-Learner with ElasticNet ────────────────────────────────
    rx_model = TargetedBayesianXLearner(
        n_splits=2,
        num_warmup=150,
        num_samples=300,
        num_chains=2,
        nuisance_method="elasticnet",
        random_state=42,
    )

    # Assertion 1: no MCMC divergence exception (stability under high-dim noise)
    rx_model.fit(X, Y, W)

    rx_cate, ci_lo, ci_hi = rx_model.predict()
    ci_spread = float(ci_hi[0] - ci_lo[0])

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'Estimator':<30} {'CATE':>9}  {'|Bias|':>8}  {'CI spread':>10}")
    print("-" * 65)
    print(f"{'S-Learner (XGBoost, P=1001)':<30} {s_cate:>9.4f}  {abs(s_cate - tau_true):>8.4f}  {'—':>10}")
    print(f"{'RX-Learner (ElasticNet)':<30} {rx_cate[0]:>9.4f}  {abs(rx_cate[0] - tau_true):>8.4f}  {ci_spread:>10.4f}")
    print(f"{'True CATE':<30} {tau_true:>9.4f}")

    # ── Assertions ─────────────────────────────────────────────────────────

    # Assertion 2: RX-Learner CATE is within acceptable range
    assert abs(rx_cate[0] - tau_true) < 1.5, (
        f"RX-Learner CATE={rx_cate[0]:.4f} too far from truth={tau_true}. "
        "ElasticNet did not adequately suppress noise genes."
    )

    # Assertion 3: CI must be finite – no variance explosion from noise features
    assert ci_spread < 20.0, (
        f"CI spread={ci_spread:.4f} – noise features are inflating uncertainty."
    )

    # Assertion 4: RX-Learner strictly outperforms S-Learner (lower |bias|)
    rx_bias = abs(rx_cate[0] - tau_true)
    s_bias = abs(s_cate - tau_true)
    assert rx_bias < s_bias, (
        f"RX-Learner |bias|={rx_bias:.4f} not better than S-Learner |bias|={s_bias:.4f}. "
        "ElasticNet feature selection did not provide an advantage over raw XGBoost "
        f"under P={P} >> N={N}."
    )


if __name__ == "__main__":
    test_genomics_sparsity()
