import numpy as np
import pandas as pd
from pathlib import Path
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from benchmarks.dgps import imbalance_dgp, whale_dgp
import time

RESULTS_DIR = Path(__file__).parent / "results"

def run_overlap_isolation():
    rows = []
    N = 1000
    n_seeds = 10
    
    # 1. Base imbalance DGP
    # 2. Imbalance + Whales
    
    print("Running Overlap Isolation...")
    for seed in range(n_seeds):
        # Good overlap, no whales
        X1, Y1, W1, tau1 = whale_dgp(N=N, P=6, tau=2.0, n_whales=0, seed=seed)
        
        # Good overlap, whales
        n_whales = int(N * 0.20)
        X2, Y2, W2, tau2 = whale_dgp(N=N, P=6, tau=2.0, n_whales=n_whales, seed=seed)
        
        # Poor overlap, no whales
        X3, Y3, W3, tau3 = imbalance_dgp(N=N, P=6, tau=2.0, treatment_prob=0.95, seed=seed)
        
        # Poor overlap, whales
        rng = np.random.RandomState(seed + 10_000)
        Y4 = Y3.copy()
        ctrl = np.where(W3 == 0)[0]
        n_whales_poor = max(1, len(ctrl) // 2)
        if len(ctrl) > 0:
            whale_idx = rng.choice(ctrl, size=n_whales_poor, replace=False)
            Y4[whale_idx] += 5000.0
            
        dgps = [
            ("Good Overlap, No Whales", X1, Y1, W1, tau1),
            ("Good Overlap, Whales", X2, Y2, W2, tau2),
            ("Poor Overlap, No Whales", X3, Y3, W3, tau3),
            ("Poor Overlap, Whales", X3, Y4, W3, tau3),
        ]
        
        for name, X, Y, W, tau in dgps:
            for use_overlap in [False, True]:
                model = TargetedBayesianXLearner(
                    contamination_severity="severe",
                    n_splits=2,
                    num_warmup=400,
                    num_samples=800,
                    num_chains=2,
                    use_overlap=use_overlap,
                    random_state=seed
                )
                
                t0 = time.time()
                try:
                    model.fit(X, Y, W)
                    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
                    ate = float(np.mean(beta))
                    lo, hi = np.percentile(beta, [2.5, 97.5])
                except Exception as e:
                    print(e)
                    ate, lo, hi = np.nan, np.nan, np.nan
                rt = time.time() - t0
                
                rows.append({
                    "seed": seed,
                    "dgp": name,
                    "overlap_weights": use_overlap,
                    "ate": ate,
                    "lo": lo,
                    "hi": hi,
                    "ci_width": hi - lo if not np.isnan(hi) else np.nan,
                    "cov": lo <= tau <= hi if not np.isnan(hi) else False,
                    "runtime": rt
                })
        print(f"Seed {seed} done.")
        
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "overlap_isolation_raw.csv", index=False)

if __name__ == "__main__":
    run_overlap_isolation()
