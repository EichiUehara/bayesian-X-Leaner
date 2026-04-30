import numpy as np
import pytest
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

@pytest.mark.xfail(reason="Limit Theorem: Unidentifiable when all 3 structural layers are blinded.")
def test_regularization_leakage():
    # Extreme confounding, force underfitting
    np.random.seed(0)
    N = 1000
    P = 10 
    X = np.random.normal(0, 1, size=(N, P))
    
    # 3 massive confounders
    confounding_effect = 2.0 * X[:, 0] + 2.0 * X[:, 1] + 2.0 * X[:, 2]
    pi_true = 1 / (1 + np.exp(-confounding_effect))
    pi_true = np.clip(pi_true, 0.1, 0.9)
    W = np.random.binomial(1, pi_true)
    
    # CATE is exactly 2.0
    tau = 2.0
    Y0 = confounding_effect + np.random.normal(0, 0.1, size=N)
    Y1 = Y0 + tau
    Y = np.where(W==1, Y1, Y0)
    
    # Underfit XGBoost parameters heavily to simulate heavy regularization/leakage in the outcome
    outcome_params = {'max_depth': 1, 'n_estimators': 3, 'reg_alpha': 10, 'reg_lambda': 10}
    propensity_params = {'max_depth': 1, 'n_estimators': 3}
    
    model = TargetedBayesianXLearner(
        outcome_model_params=outcome_params,
        propensity_model_params=propensity_params,
        n_splits=2,
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        random_state=42
    )
    
    # High-dimensional Nuisance, but low-dimensional MCMC (intercept only)
    model.fit(X, Y, W)
    
    # Predict CATE for a new sample (intercept only automatically uses ones)
    mean_cate, ci_lower, ci_upper = model.predict()
    
    # Check if the predicted CATE is close to 2.0 despite bad nuisance models
    assert np.abs(mean_cate[0] - 2.0) < 1.0, f"Regularization leakage detected! CATE = {mean_cate[0]} != 2.0"
