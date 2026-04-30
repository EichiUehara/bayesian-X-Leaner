import numpy as np
import pytest
from sert_xlearner.core.orthogonalization import impute_and_debias

def test_few_placebo_imbalance():
    """
    Validation Test for Phase 2: Imputation & Debiasing
    Execution: Skew dataset heavily, e.g., 99% treatment, 1% control.
    Failure Condition: Inverse propensity weighting inside pseudo-outcomes triggers variance explosion.
    """
    np.random.seed(42)
    N = 2000
    
    # Simulate an environment where treatment is nearly universal
    pi_true = np.random.uniform(0.95, 0.999, size=N) 
    W = np.random.binomial(1, pi_true)
    
    Y = np.random.normal(0, 1, size=N)
    
    # Simulated Phase 1 predictions
    pred_mu0 = np.zeros(N)
    pred_mu1 = np.ones(N)
    # Give it the actual extreme propensities to demonstrate the mathematical flaw in IPW
    pred_pi = pi_true 
    
    # 1. Baseline Test (Traditional Doubly Robust / IPW)
    treated_mask, control_mask, D1, D0, W_D1, W_D0 = impute_and_debias(
        Y, W, pred_mu0, pred_mu1, pred_pi, robust=False
    )
    
    var_D1 = np.var(D1)
    var_D0 = np.var(D0)
    print(f"\nBaseline IPW Phase 2 Variance (Treated - D1): {var_D1:.2f}")
    print(f"Baseline IPW Phase 2 Variance (Control - D0): {var_D0:.2f}")
    
    # The variance should be massive. The control group specifically, where pi approaches 0.999, 
    # will have 1/(1-pi) approaching 1000. D0 will have variance ~ 1,000,000.
    
    # Assert the baseline FAILS (variance explosion) as expected by the Phase 2 boundary
    assert var_D0 > 100.0, f"Expected massive Variance Explosion on D0, but got {var_D0}!"

    # 2. Validate Replacement: Overlap Weights
    treated_mask_ow, control_mask_ow, D1_ow, D0_ow, W_D1_ow, W_D0_ow = impute_and_debias(
        Y, W, pred_mu0, pred_mu1, pred_pi, robust=False, use_overlap=True
    )
    
    var_D1_ow = np.var(D1_ow)
    var_D0_ow = np.var(D0_ow)
    print(f"\nOverlap Phase 2 Variance (Treated - D1): {var_D1_ow:.2f}")
    print(f"Overlap Phase 2 Variance (Control - D0): {var_D0_ow:.2f}")
    
    assert var_D0_ow < 5.0, f"Overlap Weights failed to stabilize D0! Variance: {var_D0_ow}"
    assert var_D1_ow < 5.0, f"Overlap Weights failed to stabilize D1! Variance: {var_D1_ow}"
    
if __name__ == '__main__':
    test_few_placebo_imbalance()
