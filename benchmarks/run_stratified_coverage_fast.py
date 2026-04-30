import numpy as np
import pandas as pd
from pathlib import Path
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from benchmarks.dgps import nonlinear_cate_dgp

RESULTS_DIR = Path(__file__).parent / "results"

def run_stratified():
    rows = []
    n_seeds = 30
    N = 1000
    
    print("Running Stratified Coverage...")
    for seed in range(n_seeds):
        X, Y, W, tau_vec = nonlinear_cate_dgp(N=N, P=6, seed=seed)
        
        # intercept and X0 as basis
        X_infer = np.column_stack([np.ones(N), X[:, 0]])
        
        model = TargetedBayesianXLearner(
            contamination_severity="none", 
            n_splits=2,
            num_warmup=400,
            num_samples=800,
            num_chains=2,
            random_state=seed,
        )
        
        model.fit(X, Y, W, X_infer=X_infer)
        
        beta_samples = model.mcmc_samples["beta"]
        cate_samples = np.dot(X_infer, beta_samples.T)
        ci_lo, ci_hi = np.percentile(cate_samples, [2.5, 97.5], axis=1)
        
        covered = (ci_lo <= tau_vec) & (tau_vec <= ci_hi)
        
        # Stratify by X0 quintiles
        quintiles = pd.qcut(X[:, 0], 5, labels=False)
        
        for q in range(5):
            mask = (quintiles == q)
            q_cov = covered[mask].mean()
            rows.append({
                "seed": seed,
                "quintile": q,
                "coverage": q_cov
            })
        print(f"Seed {seed} done.")
            
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "stratified_coverage_raw.csv", index=False)
    
    agg = df.groupby("quintile")["coverage"].mean().reset_index()
    print("\n--- Stratified Coverage ---")
    print(agg)

if __name__ == "__main__":
    run_stratified()
