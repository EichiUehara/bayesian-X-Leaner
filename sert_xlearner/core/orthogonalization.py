import numpy as np
import warnings


def normalize_extremes(D, threshold, alpha):
    """Hill-estimator-based tail rescaling (architectural residue).

    EMPIRICALLY DISCOURAGED. The tail-heterogeneous probe of the
    accompanying paper (§5.5) shows this operator is contamination-
    directed, not signal-preserving: it divides extreme pseudo-outcomes
    by threshold^alpha, which removes signal when the tail carries
    heterogeneous treatment effect. Documented in Appendix C of the
    paper as "architectural residue"; retained for reproducibility of
    earlier experiments. Disabled by default — only triggered when both
    `tail_threshold` and `tail_alpha` are passed explicitly.

    Use the tail-aware CATE basis (X_infer) for tails-as-signal, and
    Welsch + Huber-nuisance (severity="severe") for tails-as-contamination,
    instead of activating this path.
    """
    warnings.warn(
        "normalize_extremes is empirically discouraged; see paper §5.5 "
        "and Appendix C. For tails-as-signal, use a tail-aware X_infer "
        "basis. For tails-as-contamination, use contamination_severity "
        "and the Welsch likelihood (the library defaults).",
        UserWarning, stacklevel=2,
    )
    extremes_mask = np.abs(D) > threshold

    # Scale down the whales by t^alpha so they map to a stable causal parameter
    D_normalized = np.copy(D)
    D_normalized[extremes_mask] = D[extremes_mask] / (threshold ** alpha)

    return D_normalized

def impute_and_debias(Y, W, pred_mu0, pred_mu1, pred_pi, robust=False, tail_threshold=None, tail_alpha=None, use_overlap=False):
    """
    Phase 2: X-Learner Imputation & Targeted Debiasing
    
    Y: Output (N,)
    W: Treatment (N,)
    pred_mu0: Predicted control outcome (N,)
    pred_mu1: Predicted treated outcome (N,)
    pred_pi: Predicted propensity score (N,)
    robust: Whether to apply tail normalization
    tail_threshold: Threshold for extreme values
    tail_alpha: Tail growth index
    use_overlap: Whether to use bounded Overlap Weights (Li et al.) instead of IPW Doubly Robust
    
    Returns:
    tuple: Filtered arrays for MCMC
        - treated_mask, control_mask
        - D1, D0 (pseudo-outcomes)
        - W_D1, W_D0 (density ratio weights)
    """
    treated_mask = (W == 1)
    control_mask = (W == 0)

    pi_t = pred_pi[treated_mask]
    mu1_t = pred_mu1[treated_mask]
    mu0_t = pred_mu0[treated_mask]
    
    pi_c = pred_pi[control_mask]
    mu1_c = pred_mu1[control_mask]
    mu0_c = pred_mu0[control_mask]
    
    if use_overlap:
        # Overlap Weights (Li et al.) - Structurally bounds variance in "Few Placebo" scenarios
        # Targets are simply unscaled residuals, bounded mathematically by their likelihood weights.
        D1 = Y[treated_mask] - mu0_t
        D0 = mu1_c - Y[control_mask]
        
        W_D1 = 1.0 - pi_t
        W_D0 = pi_c
    else:
        # 1. Doubly Robust Target Outcomes (Kennedy 2020)
        D1 = mu1_t - mu0_t + (Y[treated_mask] - mu1_t) / pi_t
        D0 = mu1_c - mu0_c - (Y[control_mask] - mu0_c) / (1.0 - pi_c)
        
        # We don't multiply by IPW a second time in the Gaussian scale, to prevent variance explosion.
        W_D1 = np.ones_like(D1)
        W_D0 = np.ones_like(D0)

    # --- THE TAIL REFINEMENT STEP ---
    if robust and tail_threshold is not None and tail_alpha is not None:
        D1 = normalize_extremes(D1, tail_threshold, tail_alpha)
        D0 = normalize_extremes(D0, tail_threshold, tail_alpha)

    return treated_mask, control_mask, D1, D0, W_D1, W_D0

