import numpy as np
import pytest
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
import warnings
warnings.filterwarnings("ignore")

def test_catboost_vs_xgboost_ric():
    """
    Stress test verifying that CatBoost (Symmetric Trees) survives 
    the RIC trap where standard XGBoost fails.
    """
    np.random.seed(42)
    N, P = 1000, 150
    
    # Generate high-dimensional noise
    X = np.random.normal(0, 1, size=(N, P))
    
    # Only the first 3 variables are true confounders
    confounding = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 2]
    
    # Treatment Assignment (strongly dependent on X0, X1, X2)
    pi = 1 / (1 + np.exp(-confounding))
    W = np.random.binomial(1, pi)
    
    # Outcome Model
    mu_true = 3.0 * X[:, 0] + 1.0 * X[:, 1] - 2.0 * X[:, 2]
    # Structural CATE is constant 5.0
    true_cate = 5.0
    Y = mu_true + W * true_cate + np.random.normal(0, 0.5, size=N)
    
    print("\n" + "="*50)
    print("RIC STRESS TEST: CATBOOST VS XGBOOST")
    print("="*50)
    
    # 1. Baseline XGBoost (Likely to fall into RIC trap)
    xgb_learner = TargetedBayesianXLearner(
        outcome_model_params={'max_depth': 3, 'n_estimators': 100},
        nuisance_method='xgboost',
        random_state=42
    )
    xgb_learner.fit(X, Y, W)
    mean_cate_xgb, _, _ = xgb_learner.predict()
    mse_xgb = np.mean((mean_cate_xgb - true_cate)**2)
    print(f"XGBoost CATE Estimation MSE: {mse_xgb:.4f}")
    
    # 2. CatBoost (Symmetric Trees - Structural Regularization)
    cb_learner = TargetedBayesianXLearner(
        outcome_model_params={'depth': 3, 'iterations': 100},
        nuisance_method='catboost',
        random_state=42
    )
    cb_learner.fit(X, Y, W)
    mean_cate_cb, _, _ = cb_learner.predict()
    mse_cb = np.mean((mean_cate_cb - true_cate)**2)
    print(f"CatBoost CATE Estimation MSE: {mse_cb:.4f}")
    
    # Assertion: CatBoost should significantly outperform XGBoost on this trap
    # Note: On some seeds XGBoost might recover, but generally it fails.
    print(f"Relative Improvement: {((mse_xgb - mse_cb) / mse_xgb) * 100:.2f}%")
    assert mse_cb < mse_xgb, "CatBoost failed to outperform XGBoost in the RIC trap!"

if __name__ == "__main__":
    test_catboost_vs_xgboost_ric()
