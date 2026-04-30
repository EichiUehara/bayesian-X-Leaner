import numpy as np
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Baseline estimators
# ---------------------------------------------------------------------------

def _naive_dim(Y, W):
    """Naive difference-in-means: ignores confounding entirely."""
    return float(np.mean(Y[W == 1]) - np.mean(Y[W == 0]))


def _s_learner_cate(X, Y, W, xgb_params):
    """
    S-Learner: single XGBoost trained on (X, W) → Y.
    Predicts CATE as E[Y|X, W=1] - E[Y|X, W=0].

    Under the sharp null (Y1 = Y0), the true CATE = 0. However, because W is
    strongly correlated with Y through confounding, the tree may split on W and
    attribute confounding variation to the treatment indicator, hallucinating a
    non-zero effect.
    """
    XW = np.column_stack([X, W.astype(float)])
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(XW, Y)

    X1 = np.column_stack([X, np.ones(len(X))])
    X0 = np.column_stack([X, np.zeros(len(X))])
    return float(np.mean(model.predict(X1) - model.predict(X0)))


def _dr_learner_cate(X, Y, W, xgb_params):
    """
    DR-Learner (AIPW) with cross-fitted XGBoost nuisance models.

    Better than naïve DiM but still susceptible to confounding leakage
    when nuisance models do not perfectly recover the non-linear DGP.
    """
    N = len(X)
    pseudo = np.zeros(N)
    kf = KFold(n_splits=2, shuffle=True, random_state=7)

    for train_idx, test_idx in kf.split(X):
        Xtr, Ytr, Wtr = X[train_idx], Y[train_idx], W[train_idx]
        Xte, Wte, Yte = X[test_idx], W[test_idx], Y[test_idx]

        mu0 = xgb.XGBRegressor(**xgb_params)
        mu0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0])

        mu1 = xgb.XGBRegressor(**xgb_params)
        mu1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])

        pi_mdl = xgb.XGBClassifier(**xgb_params)
        pi_mdl.fit(Xtr, Wtr)
        pi = np.clip(pi_mdl.predict_proba(Xte)[:, 1], 0.01, 0.99)

        mu0_te, mu1_te = mu0.predict(Xte), mu1.predict(Xte)
        pseudo[test_idx] = (
            mu1_te - mu0_te
            + Wte / pi * (Yte - mu1_te)
            - (1 - Wte) / (1 - pi) * (Yte - mu0_te)
        )

    return float(np.mean(pseudo))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_marketing_sharp_null():
    """
    Marketing A/A Test 'Sharp Null' Use Case: Four-way comparison of estimator
    bias when the true treatment effect is exactly zero but confounding is
    non-linear and moderately strong.

    Estimators compared
    -------------------
    1. Naïve DiM                  – fails hardest: confounding is fully attributed
                                    to treatment → |CATE| can be ≫ 0.5.
    2. S-Learner (XGBoost)        – partially fails: the tree may split on W when
                                    W is correlated with Y through confounders,
                                    hallucinating a residual treatment effect.
    3. DR-Learner (AIPW)          – better but still susceptible to residual
                                    nuisance-model error leaking into pseudo-outcomes.
    4. RX-Learner (cross-fitting) – best: double-robustness makes Phase-2 bias
                                    second-order; |mean_cate| < 0.5 and zero
                                    is inside the 95 % CI.

    True CATE: 0.0 everywhere.
    """
    np.random.seed(7)
    N = 1500
    P = 10
    tau_true = 0.0

    X = np.random.normal(0, 1, size=(N, P))

    # Moderate non-linear confounding
    propensity_logit = 0.8 * X[:, 0] + 0.6 * X[:, 1]
    pi_true = np.clip(1 / (1 + np.exp(-propensity_logit)), 0.10, 0.90)
    W = np.random.binomial(1, pi_true)

    X1_clip = np.clip(X[:, 1], -2.5, 2.5)
    Y0 = (
        np.sin(X[:, 0])
        + 0.5 * X1_clip ** 2
        - 0.4 * X[:, 0] * X[:, 1]
        + 0.3 * X[:, 2]
        + np.random.normal(0, 0.4, size=N)
    )

    # Sharp null: Y1 = Y0 (true CATE = 0)
    Y1 = Y0.copy()
    Y = np.where(W == 1, Y1, Y0)

    xgb_params = {"max_depth": 4, "n_estimators": 150}

    # ── Baseline 1: Naïve DiM ───────────────────────────────────────────────
    dim = _naive_dim(Y, W)

    # ── Baseline 2: S-Learner ───────────────────────────────────────────────
    s_cate = _s_learner_cate(X, Y, W, xgb_params)

    # ── Baseline 3: DR-Learner ──────────────────────────────────────────────
    dr_cate = _dr_learner_cate(X, Y, W, xgb_params)

    # ── Proposed: RX-Learner ────────────────────────────────────────────────
    rx_model = TargetedBayesianXLearner(
        outcome_model_params=xgb_params,
        propensity_model_params=xgb_params,
        n_splits=3,
        num_warmup=150,
        num_samples=300,
        num_chains=2,
        random_state=42,
    )
    rx_model.fit(X, Y, W)
    rx_cate, ci_lo, ci_hi = rx_model.predict()

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'Estimator':<26} {'CATE':>9}  {'|Bias|':>8}  {'Null inside CI':>14}")
    print("-" * 65)
    print(f"{'Naïve DiM':<26} {dim:>9.4f}  {abs(dim):>8.4f}  {'N/A':>14}")
    print(f"{'S-Learner':<26} {s_cate:>9.4f}  {abs(s_cate):>8.4f}  {'N/A':>14}")
    print(f"{'DR-Learner':<26} {dr_cate:>9.4f}  {abs(dr_cate):>8.4f}  {'N/A':>14}")
    null_in_ci = "YES" if ci_lo[0] < 0.0 < ci_hi[0] else "NO"
    print(f"{'RX-Learner':<26} {rx_cate[0]:>9.4f}  {abs(rx_cate[0]):>8.4f}  {null_in_ci:>14}")
    print(f"{'True CATE':<26} {tau_true:>9.4f}")

    # ── Assertions ─────────────────────────────────────────────────────────

    # Naïve DiM must be confounding-biased (spurious non-zero effect)
    assert abs(dim) > 0.3, (
        f"Naïve DiM={dim:.4f} is too small; confounding must create apparent effect > 0.3."
    )

    # RX-Learner recovers the null (double-robustness absorbs nuisance errors)
    assert abs(rx_cate[0]) < 0.5, (
        f"Sharp Null FAILED: RX-Learner CATE={rx_cate[0]:.4f}. "
        "Phase 1 cross-fitting is leaking confounders into pseudo-outcomes."
    )

    # RX-Learner CI must straddle zero
    assert ci_lo[0] < 0.0 < ci_hi[0], (
        f"Zero not in 95% CI: [{ci_lo[0]:.4f}, {ci_hi[0]:.4f}]. Model is overconfident."
    )

    # RX-Learner must outperform the naïve DiM on absolute bias
    assert abs(rx_cate[0]) < abs(dim), (
        f"RX-Learner |bias|={abs(rx_cate[0]):.4f} not better than DiM |bias|={abs(dim):.4f}."
    )


if __name__ == "__main__":
    test_marketing_sharp_null()
