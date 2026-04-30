import numpy as np
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Baseline estimators (inline – no external causal packages required)
# ---------------------------------------------------------------------------

def _dr_learner(X, Y, W, xgb_params):
    """
    DR-Learner (AIPW) with standard IPW and XGBoost cross-fitting.

    Returns (mean_cate, std_pseudo_outcomes).  Under Whale contamination the
    (1-W)/(1-pi) IPW term amplifies the whale residual by 1/(1-pi) ≈ 2-7x,
    so the pseudo-outcome variance will be significantly larger than the raw
    outcome variance, and the point estimate will be heavily biased.
    """
    N = len(X)
    pseudo = np.zeros(N)
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        Xtr, Ytr, Wtr = X[train_idx], Y[train_idx], W[train_idx]
        Xte, Wte, Yte = X[test_idx], W[test_idx], Y[test_idx]

        mu0 = xgb.XGBRegressor(**xgb_params)
        ctrl = Wtr == 0
        mu0.fit(Xtr[ctrl], Ytr[ctrl]) if ctrl.sum() > 1 else mu0.fit(Xtr, Ytr)

        mu1 = xgb.XGBRegressor(**xgb_params)
        mu1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])

        pi_mdl = xgb.XGBClassifier(**xgb_params)
        pi_mdl.fit(Xtr, Wtr)
        pi = np.clip(pi_mdl.predict_proba(Xte)[:, 1], 0.01, 0.99)

        mu0_te, mu1_te = mu0.predict(Xte), mu1.predict(Xte)

        # AIPW: the (1-Wte)/(1-pi) term amplifies any Whale in the control group
        pseudo[test_idx] = (
            mu1_te - mu0_te
            + Wte / pi * (Yte - mu1_te)
            - (1 - Wte) / (1 - pi) * (Yte - mu0_te)
        )

    return float(np.mean(pseudo)), float(np.std(pseudo))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_adtech_whale_smearing():
    """
    AdTech/E-Commerce Use Case: Three-way comparison of CATE estimators under
    a single Control Whale (Y += 5,000).

    Estimators compared
    -------------------
    1. DR-Learner (AIPW)          – fails hardest: IPW term multiplies the whale
                                    residual by 1/(1-pi), exploding pseudo-outcomes.
    2. Standard X-Learner (MSE)   – fails: Gaussian MCMC is dominated by the
                                    huge negative D0 outlier, pulling beta < 0.
    3. RX-Learner (Welsch MCMC)   – survives: redescending influence function
                                    assigns near-zero gradient to the outlier.

    True CATE: 2.0
    """
    np.random.seed(42)
    N = 600
    P = 8
    tau_true = 2.0

    X = np.random.normal(0, 1, size=(N, P))
    pi_true = np.clip(1 / (1 + np.exp(-X[:, 0])), 0.15, 0.85)
    W = np.random.binomial(1, pi_true)

    Y0 = 1.5 * X[:, 0] + np.random.normal(0, 0.3, size=N)
    Y1 = Y0 + tau_true
    Y = np.where(W == 1, Y1, Y0)

    # Inject one massive Whale into the control group
    control_indices = np.where(W == 0)[0]
    whale_idx = control_indices[0]
    Y[whale_idx] += 5000.0

    xgb_params = {"max_depth": 3, "n_estimators": 50}
    mcmc_kwargs = dict(
        outcome_model_params=xgb_params,
        propensity_model_params=xgb_params,
        n_splits=2,
        num_warmup=150,
        num_samples=300,
        num_chains=2,
        random_state=42,
    )

    # ── Baseline 1: DR-Learner (AIPW) ──────────────────────────────────────
    dr_cate, dr_std = _dr_learner(X, Y, W, xgb_params)

    # ── Baseline 2: Standard X-Learner (Gaussian MCMC) ─────────────────────
    standard_model = TargetedBayesianXLearner(**mcmc_kwargs, robust=False)
    standard_model.fit(X, Y, W)
    standard_cate, _, _ = standard_model.predict()

    # ── Proposed: RX-Learner (Welsch MCMC) ─────────────────────────────────
    robust_model = TargetedBayesianXLearner(**mcmc_kwargs, robust=True, c_whale=1.34)
    robust_model.fit(X, Y, W)
    robust_cate, _, _ = robust_model.predict()

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'Estimator':<28} {'CATE':>9}  {'|Bias|':>8}  {'Std(D)':>9}")
    print("-" * 60)
    print(f"{'DR-Learner (AIPW)':<28} {dr_cate:>9.4f}  {abs(dr_cate - tau_true):>8.4f}  {dr_std:>9.4f}")
    print(f"{'Standard X-Learner':<28} {standard_cate[0]:>9.4f}  {abs(standard_cate[0] - tau_true):>8.4f}  {'—':>9}")
    print(f"{'RX-Learner (Welsch)':<28} {robust_cate[0]:>9.4f}  {abs(robust_cate[0] - tau_true):>8.4f}  {'—':>9}")
    print(f"{'True CATE':<28} {tau_true:>9.4f}")

    # ── Assertions ─────────────────────────────────────────────────────────

    # DR-Learner: IPW amplification → pseudo-outcome std ≫ baseline noise σ=0.3
    assert dr_std > 50.0, (
        f"DR-Learner std={dr_std:.2f}. Expected IPW to amplify whale residuals "
        "into std > 50 under pi ∈ [0.15, 0.85]."
    )

    # Standard X-Learner: D0[whale] ≈ -(5000/(1-pi)) drags beta below zero
    assert standard_cate[0] < 0.0, (
        f"Expected standard model smeared below 0, got {standard_cate[0]:.4f}."
    )

    # RX-Learner: Welsch gradient → 0 at the outlier; remaining points recover truth
    assert abs(robust_cate[0] - tau_true) < 1.0, (
        f"RX-Learner failed to suppress Whale. CATE={robust_cate[0]:.4f}"
    )

    # RX-Learner must strictly outperform both baselines on absolute bias
    assert abs(robust_cate[0] - tau_true) < abs(standard_cate[0] - tau_true), (
        "RX-Learner did not reduce bias vs Standard X-Learner."
    )
    assert abs(robust_cate[0] - tau_true) < abs(dr_cate - tau_true), (
        "RX-Learner did not reduce bias vs DR-Learner."
    )


if __name__ == "__main__":
    test_adtech_whale_smearing()
