import numpy as np
from sert_xlearner.core.orthogonalization import impute_and_debias

def test_algebraic_orthogonalization():
    """
    If the density ratio weighting is perfectly specified (and mu models match Y),
    the bias should algebraicially cancel out.
    """
    N = 1000
    np.random.seed(42)
    # Synthetic scenario
    pi_true = np.random.uniform(0.1, 0.9, size=N)
    W = np.random.binomial(1, pi_true)
    
    # Perfect nuisance estimators prediction
    # If pred_mu0 exactly matches Control Y and pred_mu1 matches Treated Y
    # Then imputation should calculate perfect 0 empirical mean.
    
    Y = np.random.normal(0, 1, size=N)
    
    # Suppose models perfectly fit Y
    pred_mu0 = Y.copy()
    pred_mu1 = Y.copy()
    pred_pi = pi_true.copy()
    
    treated_mask, control_mask, D1, D0, W_D1, W_D0 = impute_and_debias(
        Y, W, pred_mu0, pred_mu1, pred_pi
    )
    
    # Pseudo-outcomes should evaluate to exactly 0 when model predictions match Y
    assert np.allclose(D1, 0.0), "D1 pseudo-outcomes not exactly zero under perfect specification"
    assert np.allclose(D0, 0.0), "D0 pseudo-outcomes not exactly zero under perfect specification"
    
    # Weighted mean of D1 and D0
    weighted_mean_D1 = np.mean(D1 * W_D1)
    weighted_mean_D0 = np.mean(D0 * W_D0)
    
    assert np.isclose(weighted_mean_D1, 0.0), "Weighted mean of D1 not zero"
    assert np.isclose(weighted_mean_D0, 0.0), "Weighted mean of D0 not zero"
