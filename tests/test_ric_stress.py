import numpy as np
import pytest
from sert_xlearner.models.nuisance import NuisanceEstimator

def test_ric_stress_high_dimensional_confounding():
    """
    Validation Test for Phase 1: Nuisance Quarantine
    Execution: Generate synthetic data with 100+ covariates (only 3 true confounders, rest noise).
    Failure Condition: Algorithm over-regularizes and drops true confounders, causing RIC.
    """
    np.random.seed(42)
    N = 800
    P = 150 # High-dimensional
    
    X = np.random.normal(0, 1, size=(N, P))
    
    # Only X[:, 0], X[:, 1], X[:, 2] are true confounders. The other 147 are pure noise.
    confounding_effect = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 2]
    
    pi_true = 1 / (1 + np.exp(-confounding_effect))
    pi_true = np.clip(pi_true, 0.05, 0.95)
    W = np.random.binomial(1, pi_true)
    
    mu_true = 3.0 * X[:, 0] + 1.0 * X[:, 1] - 2.0 * X[:, 2]
    Y = mu_true + W * 2.0 + np.random.normal(0, 0.5, size=N)
    
    # We will use the NuisanceEstimator and see if it drops the true confounders
    # by measuring the First-Stage MSE against the true outcome surface.
    
    # Test baseline XGBoost (often over-regularizes in P >> N if not tuned carefully)
    outcome_params = {'max_depth': 2, 'n_estimators': 20, 'reg_alpha': 1.0, 'reg_lambda': 1.0}
    propensity_params = {'max_depth': 2, 'n_estimators': 20, 'reg_alpha': 1.0, 'reg_lambda': 1.0}
    
    estimator = NuisanceEstimator(outcome_params, propensity_params, n_splits=2, random_state=42)
    out_pred_mu0, out_pred_mu1, out_pred_pi = estimator.fit_predict(X, Y, W)
    
    # We calculate the True Mu on out-of-fold predictions.
    # True mu0 is mu_true. True mu1 is mu_true + 2.0.
    mse_mu0 = np.mean((out_pred_mu0[W==0] - mu_true[W==0])**2)
    mse_mu1 = np.mean((out_pred_mu1[W==1] - (mu_true[W==1] + 2.0))**2)
    
    print(f"\nBaseline XGBoost Phase 1 MSE (Control): {mse_mu0:.4f}")
    print(f"Baseline XGBoost Phase 1 MSE (Treated): {mse_mu1:.4f}")
    
    # In a fully functioning Phase 1 without Regularization Induced Confounding (RIC),
    # the algorithm MUST have Low MSE on the true confounder manifold.
    # A failure here implies the baseline learner was "dogmatic" and generalized poorly.
    # NOTE: Since this is a test script, we verify the baseline XGBoost FAILS as an assertion
    # that RIC actually exists in this dataset configuration.
    assert mse_mu0 > 1.0, f"Expected XGBoost to suffer RIC, but MSE was unusually good: {mse_mu0}"
    
    print("\nTesting Regularized Linear Models (ElasticNet/LogRegCV) Phase 1 Replacement...")
    en_estimator = NuisanceEstimator({}, {}, n_splits=2, random_state=42)
    en_estimator.use_en = True
    out_pred_mu0_en, out_pred_mu1_en, out_pred_pi_en = en_estimator.fit_predict(X, Y, W)
    
    mse_mu0_en = np.mean((out_pred_mu0_en[W==0] - mu_true[W==0])**2)
    mse_mu1_en = np.mean((out_pred_mu1_en[W==1] - (mu_true[W==1] + 2.0))**2)
    
    print(f"ElasticNet Phase 1 MSE (Control): {mse_mu0_en:.4f}")
    print(f"ElasticNet Phase 1 MSE (Treated): {mse_mu1_en:.4f}")
    
    # ElasticNet must succeed mathematically where XGBoost failed
    assert mse_mu0_en < 1.0, f"ElasticNet also dropped confounders! MSE: {mse_mu0_en}"
    assert mse_mu1_en < 1.0, f"ElasticNet also dropped confounders! MSE: {mse_mu1_en}"

if __name__ == '__main__':
    test_ric_stress_high_dimensional_confounding()
