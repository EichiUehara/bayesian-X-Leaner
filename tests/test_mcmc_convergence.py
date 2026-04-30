import numpy as np
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
import numpyro.diagnostics as diag

def test_mcmc_convergence():
    np.random.seed(42)
    N = 200
    P = 2
    X = np.random.normal(0, 1, size=(N, P))
    W = np.random.binomial(1, 0.5, size=N)
    Y = W * 2.0 + np.random.normal(0, 0.1, size=N)
    
    model = TargetedBayesianXLearner(
        outcome_model_params={'n_estimators': 10},
        propensity_model_params={'n_estimators': 10},
        num_warmup=200,
        num_samples=500,
        num_chains=2
    )
    
    model.fit(X, Y, W)
    
    # Get un-collapsed mcmc samples from the MCMC runner directly
    mcmc_runner = model.bayesian_mcmc.mcmc
    samples = mcmc_runner.get_samples(group_by_chain=True)
    
    beta_samples = samples['beta']
    
    # Calculate R_hat
    r_hat = diag.split_gelman_rubin(beta_samples)
    
    # Calculate ESS
    ess = diag.effective_sample_size(beta_samples)
    
    # Assert R_hat < 1.05 and ESS > 10% of total samples (2 chains * 500 = 1000)
    assert np.all(r_hat < 1.05), "Gelman-Rubin statistic is > 1.05! Chains didn't converge."
    assert np.all(ess > 100), "Effective Sample Size is too low!"
