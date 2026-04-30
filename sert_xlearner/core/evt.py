import numpy as np

def estimate_tail_parameters(Y, top_percentile=95):
    """
    Estimates the tail threshold (t) and tail growth index (alpha) 
    using the Hill Estimator from Extreme Value Theory (EVT).
    
    Parameters:
    Y (array): The observed outcomes (or residuals).
    top_percentile (float): The percentile defining the start of the heavy tail.
    
    Returns:
    float, float: tail_threshold (t), tail_alpha (alpha)
    """
    # 1. We care about magnitude for heavy tails, so take absolute values
    Y_abs = np.abs(Y)
    
    # 2. Define the Threshold (t) based on the chosen percentile
    t = np.percentile(Y_abs, top_percentile)
    
    # 3. Isolate the Extremes (the "Whales" over the threshold)
    extremes = Y_abs[Y_abs > t]
    
    # Fallback if the data isn't actually heavy-tailed at this percentile
    if len(extremes) < 2:
        return t, 1.0 
        
    # 4. Calculate the Hill Estimator (gamma)
    # Formula: gamma = (1/k) * sum(ln(Y_i / t))
    gamma = np.mean(np.log(extremes / t))
    
    # 5. Convert to Tail Index (alpha)
    # The tail index alpha is the inverse of the Hill estimator gamma
    # We add a small epsilon to prevent division by zero in perfect edge cases
    alpha = 1.0 / (gamma + 1e-8)
    
    return t, alpha
