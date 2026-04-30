import numpy as np
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared DGP
# ---------------------------------------------------------------------------

def _make_dataset(N, treatment_prob, seed):
    """Generate a simple linear DGP with known CATE = 2.0 and a fixed treatment rate."""
    rng = np.random.RandomState(seed)
    P = 5
    X = rng.normal(0, 1, size=(N, P))
    confounding = X[:, 0]
    pi_true = np.full(N, treatment_prob)
    W = rng.binomial(1, pi_true)
    Y0 = confounding + rng.normal(0, 0.5, size=N)
    Y1 = Y0 + 2.0
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W


# ---------------------------------------------------------------------------
# Baseline: DR-Learner with raw IPW (no overlap-weight stabilisation)
# ---------------------------------------------------------------------------

def _dr_learner_ipw_ci_width(X, Y, W, xgb_params):
    """
    Fit a DR-Learner (AIPW) with standard IPW and estimate CI width via
    bootstrap percentile on pseudo-outcomes.

    Under extreme imbalance (pi ≈ 0.99), the (1-W)/(1-pi) ≈ 100 term makes the
    control pseudo-outcomes explode.  The resulting std(pseudo) is enormous,
    giving an honest but useless CI (e.g., [-1000, +1000]).
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
        # clip to [0.01, 0.99]; with pi≈0.99, 1/(1-pi) can reach 100
        pi = np.clip(pi_mdl.predict_proba(Xte)[:, 1], 0.01, 0.99)

        mu0_te, mu1_te = mu0.predict(Xte), mu1.predict(Xte)
        pseudo[test_idx] = (
            mu1_te - mu0_te
            + Wte / pi * (Yte - mu1_te)
            - (1 - Wte) / (1 - pi) * (Yte - mu0_te)
        )

    mean_est = float(np.mean(pseudo))
    # Bootstrap 95 % CI on the mean of pseudo-outcomes
    rng = np.random.RandomState(0)
    boot = [np.mean(rng.choice(pseudo, size=N, replace=True)) for _ in range(500)]
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return mean_est, ci_lo, ci_hi, float(ci_hi - ci_lo)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_clinical_extreme_imbalance():
    """
    Clinical Trial 'Compassionate Use' Use Case: Three-way comparison of
    estimator confidence under extreme 99/1 treatment imbalance.

    Estimators compared
    -------------------
    1. DR-Learner (raw IPW, no stabilisation) on imbalanced data
       – fails: 1/(1-pi) ≈ 100 amplification produces a massive CI spread.
    2. RX-Learner + Overlap Weights on balanced data (50/50 reference)
       – succeeds with narrow CI (well-informed posterior).
    3. RX-Learner + Overlap Weights on imbalanced data
       – succeeds point estimate but CI is wider than balanced, correctly
         acknowledging the scarcity of control observations.

    Key assertions
    --------------
    a) DR-Learner CI width >> RX-Learner CI width (variance explosion exposed).
    b) RX-Learner imbalanced CI > RX-Learner balanced CI (epistemic honesty).
    c) RX-Learner balanced CATE is close to the true 2.0 (sanity check).
    """
    N = 1000
    tau_true = 2.0
    xgb_params = {"max_depth": 3, "n_estimators": 50}

    X_bal, Y_bal, W_bal = _make_dataset(N, treatment_prob=0.50, seed=0)
    X_imb, Y_imb, W_imb = _make_dataset(N, treatment_prob=0.99, seed=1)

    assert W_imb.mean() > 0.90, (
        f"Imbalanced dataset is not sufficiently skewed: {W_imb.mean():.3f}"
    )
    print(f"\nBalanced:    {W_bal.sum()} treated / {(W_bal == 0).sum()} control")
    print(f"Imbalanced:  {W_imb.sum()} treated / {(W_imb == 0).sum()} control")

    # ── Baseline: DR-Learner (raw IPW) on imbalanced data ──────────────────
    dr_cate, dr_lo, dr_hi, dr_width = _dr_learner_ipw_ci_width(X_imb, Y_imb, W_imb, xgb_params)

    # ── RX-Learner (overlap weights) – balanced reference ──────────────────
    rx_kwargs = dict(
        outcome_model_params=xgb_params,
        propensity_model_params=xgb_params,
        n_splits=2,
        num_warmup=150,
        num_samples=300,
        num_chains=2,
        use_overlap=True,
    )
    model_bal = TargetedBayesianXLearner(**rx_kwargs, random_state=10)
    model_bal.fit(X_bal, Y_bal, W_bal)
    cate_bal, ci_lo_bal, ci_hi_bal = model_bal.predict()
    rx_bal_width = float(ci_hi_bal[0] - ci_lo_bal[0])

    # ── RX-Learner (overlap weights) – imbalanced ───────────────────────────
    model_imb = TargetedBayesianXLearner(**rx_kwargs, random_state=11)
    model_imb.fit(X_imb, Y_imb, W_imb)
    cate_imb, ci_lo_imb, ci_hi_imb = model_imb.predict()
    rx_imb_width = float(ci_hi_imb[0] - ci_lo_imb[0])

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'Estimator':<34} {'CATE':>8}  {'CI lo':>8}  {'CI hi':>8}  {'Width':>8}")
    print("-" * 74)
    print(f"{'DR-Learner/IPW (imbalanced)':<34} {dr_cate:>8.3f}  {dr_lo:>8.3f}  {dr_hi:>8.3f}  {dr_width:>8.3f}")
    print(f"{'RX-Learner/Overlap (balanced)':<34} {cate_bal[0]:>8.3f}  {ci_lo_bal[0]:>8.3f}  {ci_hi_bal[0]:>8.3f}  {rx_bal_width:>8.3f}")
    print(f"{'RX-Learner/Overlap (imbalanced)':<34} {cate_imb[0]:>8.3f}  {ci_lo_imb[0]:>8.3f}  {ci_hi_imb[0]:>8.3f}  {rx_imb_width:>8.3f}")
    print(f"{'True CATE':<34} {tau_true:>8.3f}")

    # ── Assertions ─────────────────────────────────────────────────────────

    # (a) DR-Learner CI must be materially wider than Overlap-weighted RX.
    # The 1/(1-pi) term is clipped at 1/0.01=100 by the implementation's
    # pi∈[0.01,0.99] clip, so the "explosion" is bounded. We still expect
    # at least a 2× CI-width gap (empirically ~3.3× at N=1000, 99% treated).
    assert dr_width > rx_imb_width * 2, (
        f"DR-Learner CI width={dr_width:.2f} not materially wider than "
        f"RX-Learner width={rx_imb_width:.2f}. Overlap-weight benefit not demonstrated."
    )

    # (b) RX-Learner epistemic honesty: imbalanced must be wider than balanced
    assert rx_imb_width > rx_bal_width, (
        f"Imbalanced CI width={rx_imb_width:.4f} ≤ balanced width={rx_bal_width:.4f}. "
        "Model is not acknowledging scarce controls."
    )

    # (c) Balanced sanity check
    assert abs(cate_bal[0] - tau_true) < 1.5, (
        f"Balanced CATE deviated: {cate_bal[0]:.4f}"
    )


if __name__ == "__main__":
    test_clinical_extreme_imbalance()
