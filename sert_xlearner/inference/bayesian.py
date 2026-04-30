import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import multiprocessing

class BayesianMCMC:
    """Three discrete inference modes — they are alternatives, not composable.

    | `robust` | `use_student_t` | What actually runs                            |
    |----------|-----------------|-----------------------------------------------|
    | False    | False           | Gaussian likelihood (L² loss)                 |
    | False    | True            | Student-T likelihood (heavy-tailed, unbounded)|
    | True     | (ignored)       | Welsch redescending loss as `numpyro.factor`  |

    When `robust=True` the model uses `numpyro.factor(-welsch_loss(...))` and
    never reads `use_student_t` — there is no separate likelihood distribution
    to combine with. The previously-documented "Welsch + Student-T combined"
    mode was a doc artefact; see benchmarks/results/component_ablation.md.
    """

    def __init__(self, num_warmup=1000, num_samples=2000, num_chains=2, random_seed=0, prior_scale=10.0, robust=False, c_whale=1.34, use_student_t=False):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.random_seed = random_seed
        self.prior_scale = prior_scale
        self.robust = robust
        self.c_whale = c_whale
        self.use_student_t = use_student_t
        self.mcmc_samples = None
        self.mcmc = None

    def welsch_loss(self, residual, c):
        """
        Welsch Loss: Structurally equivalent to gamma-divergence under Gaussian assumptions.
        Bounds the influence of extreme residuals (whales) to prevent Outlier Smearing.
        """
        return (c**2 / 2.0) * (1.0 - jnp.exp(-(residual / c)**2))

    def targeted_model(self, X_D1, X_D0, D1_obs, D0_obs, W_D1, W_D0):
        # Define priors
        n_features = X_D1.shape[1]
        
        beta = numpyro.sample("beta", dist.Normal(0, self.prior_scale).expand([n_features]))
        
        # Expected CATE
        tau_1 = jnp.dot(X_D1, beta)
        tau_0 = jnp.dot(X_D0, beta)

        if self.robust:
            # --- THE ROBUST PSEUDO-LIKELIHOOD ---
            res_1 = D1_obs - tau_1
            res_0 = D0_obs - tau_0
            
            log_prob_1 = -W_D1 * self.welsch_loss(res_1, self.c_whale)
            numpyro.factor("robust_ll_D1", jnp.sum(log_prob_1))
            
            log_prob_0 = -W_D0 * self.welsch_loss(res_0, self.c_whale)
            numpyro.factor("robust_ll_D0", jnp.sum(log_prob_0))
        else:
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            if self.use_student_t:
                # Student-T Likelihood (Degrees of Freedom = 3 for thick tails)
                nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
                with numpyro.handlers.scale(scale=W_D1):
                    numpyro.sample("obs_D1", dist.StudentT(nu, tau_1, sigma), obs=D1_obs)
                with numpyro.handlers.scale(scale=W_D0):
                    numpyro.sample("obs_D0", dist.StudentT(nu, tau_0, sigma), obs=D0_obs)
            else:
                # The Weighted Likelihood (Targeted Update)
                with numpyro.handlers.scale(scale=W_D1):
                    numpyro.sample("obs_D1", dist.Normal(tau_1, sigma), obs=D1_obs)
                
                with numpyro.handlers.scale(scale=W_D0):
                    numpyro.sample("obs_D0", dist.Normal(tau_0, sigma), obs=D0_obs)

    def sample_posterior(self, X_D1, X_D0, D1_obs, D0_obs, W_D1, W_D0):
        kernel = NUTS(self.targeted_model)
        self.mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains, progress_bar=False)
        rng_key = random.PRNGKey(self.random_seed)
        
        # On linux, jax requires passing array
        self.mcmc.run(rng_key, jnp.array(X_D1), jnp.array(X_D0), jnp.array(D1_obs), jnp.array(D0_obs), jnp.array(W_D1), jnp.array(W_D0))
        self.mcmc_samples = self.mcmc.get_samples(group_by_chain=False)
        return self.mcmc_samples

