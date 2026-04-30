import numpy as np
import pandas as pd
from pathlib import Path
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from benchmarks.dgps import whale_dgp
import arviz as az
import time

RESULTS_DIR = Path(__file__).parent / "results"

def run_diagnostics():
    rows = []
    densities = [0.0, 0.01, 0.05, 0.10, 0.20]
    n_seeds = 5
    N = 1000
    
    print("Running MCMC Diagnostics...")
    for density in densities:
        for seed in range(n_seeds):
            n_whales = int(N * density)
            X, Y, W, tau = whale_dgp(N=N, P=6, tau=2.0, n_whales=n_whales, seed=seed)
            
            model = TargetedBayesianXLearner(
                contamination_severity="severe",
                n_splits=2,
                num_warmup=400,
                num_samples=800,
                num_chains=2,
                random_state=seed,
            )
            
            t0 = time.time()
            model.fit(X, Y, W)
            rt = time.time() - t0
            
            mcmc = model.bayesian_mcmc.mcmc
            idata = az.from_numpyro(mcmc)
            summary = az.summary(idata)
            
            rhat_max = summary['r_hat'].max()
            ess_min = summary['ess_bulk'].min()
            
            try:
                bfmi_vals = az.bfmi(idata)
                bfmi_min = float(bfmi_vals.min())
            except Exception as e:
                bfmi_min = np.nan
                
            iac_max = float(1600 / ess_min) if ess_min > 0 else np.nan
            
            try:
                divergences = int(idata.sample_stats.diverging.sum().item())
            except:
                divergences = 0
            
            row = {
                "density": density,
                "seed": seed,
                "rhat_max": rhat_max,
                "ess_min": ess_min,
                "bfmi_min": bfmi_min,
                "iac_max": iac_max,
                "divergences": divergences,
                "runtime": rt
            }
            print(f"Density: {density:.2f}, Seed: {seed}, Rhat_max: {rhat_max:.3f}, BFMI_min: {bfmi_min:.3f}, Divs: {divergences}")
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "mcmc_diagnostics_raw.csv", index=False)
    
    agg = df.groupby("density").agg({
        "rhat_max": "max",
        "ess_min": "min",
        "bfmi_min": "min",
        "iac_max": "max",
        "divergences": "sum"
    }).reset_index()
    print("\n--- Aggregated MCMC Diagnostics ---")
    print(agg)
    agg.to_csv(RESULTS_DIR / "mcmc_diagnostics_agg.csv", index=False)

if __name__ == "__main__":
    run_diagnostics()
