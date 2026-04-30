import numpy as np
import pytest
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

def test_whale_injection_outlier_smearing():
    """
    Validation Test for Phase 3: The Bayesian Update
    Execution: Multiply the outcome of 3 control units by 1,000 ("Whales"). 
    Failure Condition: MCMC posterior mean shifts significantly toward the outliers.
    """
    np.random.seed(42)
    N = 1000
    P = 10 
    X = np.random.normal(0, 1, size=(N, P))
    
    pi_true = np.clip(1 / (1 + np.exp(-X[:,0])), 0.1, 0.9)
    W = np.random.binomial(1, pi_true)
    
    Y0 = X[:,0] + np.random.normal(0, 0.5, size=N)
    Y1 = Y0 + 2.0 # True CATE = 2.0
    Y = np.where(W==1, Y1, Y0)
    
    # Inject "Whales" in exactly 3 control units
    control_indices = np.where(W == 0)[0]
    whale_idx = np.random.choice(control_indices, size=3, replace=False)
    Y[whale_idx] *= 10000.0 # Extreme outliers
    
    # 1. Baseline Test (Gaussian Likelihood should completely shatter and shift the CATE)
    # We use fast configurations for the test to avoid huge MCMC wait times
    baseline_model = TargetedBayesianXLearner(
        outcome_model_params={'max_depth': 2, 'n_estimators': 10},
        propensity_model_params={'max_depth': 2, 'n_estimators': 10},
        nuisance_method='xgboost',
        n_splits=2,
        num_warmup=100,
        num_samples=200,
        robust=False,
        use_student_t=False,
        random_state=42
    )
    
    baseline_model.fit(X, Y, W)
    mean_cate_base, _, _ = baseline_model.predict()
    
    print(f"\nBaseline Gaussian Posterior Mean (With Whales): {mean_cate_base[0]:.2f}")
    
    # True CATE is 2.0. If the outlier smears, the shift will be huge (e.g. > +10 or < -10).
    assert np.abs(mean_cate_base[0] - 2.0) > 3.0, f"Expected massive outlier smearing in baseline, found {mean_cate_base[0]}"

    # 2. Validation of Replacement: Student-T Likelihood Robustness
    robust_model = TargetedBayesianXLearner(
        outcome_model_params={'max_depth': 2, 'n_estimators': 10},
        propensity_model_params={'max_depth': 2, 'n_estimators': 10},
        nuisance_method='xgboost',
        n_splits=2,
        num_warmup=100,
        num_samples=200,
        robust=False,
        use_student_t=True, # The Replacement!
        random_state=42
    )
    
    robust_model.fit(X, Y, W)
    mean_cate_robust, _, _ = robust_model.predict()
    
    print(f"Student-T Robust Posterior Mean (With Whales): {mean_cate_robust[0]:.2f}")
    
    # Student-T should prevent smearing and preserve the 2.0 centering (within a reasonable variance bound ~ <1.5 dev)
    assert np.abs(mean_cate_robust[0] - 2.0) < 1.5, f"Student-T failed to reject outlier influence! CATE: {mean_cate_robust[0]}"

if __name__ == '__main__':
    test_whale_injection_outlier_smearing()
