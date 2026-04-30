import numpy as np
import pytest
from sert_xlearner.core.evt import estimate_tail_parameters
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
import warnings
warnings.filterwarnings("ignore")

def test_deep_tail_extrapolation_vs_winsorization():
    """
    Validation Test for Phase 4: Extreme Value Theory (EVT) Extrapolation
    Executes a benchmark measuring MSE reconstruction on the >99th percentile
    comparing EVT parameters vs naive threshold clipping.
    """
    np.random.seed(42)
    N = 5000
    
    # 1. Generate Heavy-Tailed Data (Pareto distribution for extreme right-skew)
    # We create a covariate X that perfectly maps to a heavy-tailed outcome Y.
    # True CATE will scale non-linearly into the extreme tail.
    X = np.random.pareto(a=2.0, size=(N, 1)) * 10.0
    
    W = np.random.binomial(1, 0.5, size=N)
    
    # Baseline CATE = 2.0. But for heavy-tailed X, CATE = X
    # It blows up into the >99.9th percentile (> 1000.0)
    tau = X[:, 0] * 1.5 
    
    Y0 = X[:, 0] + np.random.normal(0, 1.0, size=N)
    Y1 = Y0 + tau
    Y = np.where(W == 1, Y1, Y0)
    
    # 90th percentile threshold for moderate-frequency training
    t_90 = np.percentile(Y, 90)
    
    # Identify the >99th percentile test set (The Extreme Unobserved Domain)
    t_99 = np.percentile(Y, 99)
    extreme_mask = Y > t_99
    
    X_extreme = X[extreme_mask]
    Y_extreme = Y[extreme_mask]
    W_extreme = W[extreme_mask]
    true_tau_extreme = tau[extreme_mask]
    
    # 2. Execute Baseline Failure Mode (Winsorization/Clipping)
    # Standard practice is to clip data above the 95th percentile
    clip_thresh = np.percentile(Y, 95)
    Y_clipped = np.clip(Y, a_min=None, a_max=clip_thresh)
    
    winsor_model = TargetedBayesianXLearner(
        outcome_model_params={'max_depth': 2, 'n_estimators': 10},
        propensity_model_params={'max_depth': 2, 'n_estimators': 10},
        robust=False,
        random_state=42
    )
    # Train heavily under-fitted baseline on clipped data
    winsor_model.fit(X, Y_clipped, W)
    pred_tau_winsor, _, _ = winsor_model.predict(X_extreme)
    mse_winsor = np.mean((pred_tau_winsor - true_tau_extreme)**2)
    
    print(f"\nBaseline (Winsorization) Extrapolation MSE: {mse_winsor:.2f}")
    
    # 3. Validate Replacement: Hill Estimator (EVT Tail Refinement)
    # Estimate physics from actual data structure 
    tail_t, tail_alpha = estimate_tail_parameters(Y, top_percentile=95)
    
    evt_model = TargetedBayesianXLearner(
        outcome_model_params={'max_depth': 2, 'n_estimators': 10},
        propensity_model_params={'max_depth': 2, 'n_estimators': 10},
        robust=True,
        tail_threshold=tail_t,
        tail_alpha=tail_alpha,
        random_state=42
    )
    evt_model.fit(X, Y, W)
    pred_tau_evt, _, _ = evt_model.predict(X_extreme)
    mse_evt = np.mean((pred_tau_evt - true_tau_extreme)**2)
    
    print(f"EVT (Hill Estimator) Extrapolation MSE: {mse_evt:.2f}")
    
    # EVT should dramatically outperform naive clipping in the 99th percentile
    assert mse_evt < mse_winsor * 0.90, f"EVT failed to outperform Winsorization! EVT: {mse_evt}, Winsor: {mse_winsor}"

if __name__ == '__main__':
    test_deep_tail_extrapolation_vs_winsorization()
