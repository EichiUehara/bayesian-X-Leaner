import numpy as np
import pandas as pd
from pathlib import Path
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
from benchmarks.dgps import whale_dgp
import time

RESULTS_DIR = Path(__file__).parent / "results"

def run_mad_sweep():
    rows = []
    c_whales = [0.5, 1.0, 1.34, 2.0]
    huber_deltas = [0.5, 1.0, 1.345, 2.0]
    N = 1000
    density = 0.20
    n_whales = int(N * density)
    
    print("Running MAD/c/delta sensitivity sweep...")
    X, Y, W, tau = whale_dgp(N=N, P=6, tau=2.0, n_whales=n_whales, seed=0)
    
    for mad_rescale in [True, False]:
        for c in c_whales:
            for delta in huber_deltas:
                outcome_params = {"depth": 4, "iterations": 150, "loss_function": f"Huber:delta={delta}"}
                model = TargetedBayesianXLearner(
                    outcome_model_params=outcome_params,
                    nuisance_method="catboost",
                    n_splits=2,
                    num_warmup=400,
                    num_samples=800,
                    num_chains=2,
                    robust=True,
                    c_whale=c,
                    mad_rescale=mad_rescale,
                    random_state=0
                )
                
                t0 = time.time()
                try:
                    model.fit(X, Y, W)
                    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
                    ate = float(np.mean(beta))
                    lo, hi = np.percentile(beta, [2.5, 97.5])
                except Exception as e:
                    ate, lo, hi = np.nan, np.nan, np.nan
                    
                rt = time.time() - t0
                
                row = {
                    "mad_rescale": mad_rescale,
                    "c_whale": c,
                    "huber_delta": delta,
                    "ate": ate,
                    "lo": lo,
                    "hi": hi,
                    "ci_width": hi - lo if not np.isnan(hi) else np.nan,
                    "cov": lo <= tau <= hi if not np.isnan(hi) else False,
                    "runtime": rt
                }
                print(f"MAD={mad_rescale}, c={c}, delta={delta} -> ATE={ate:.3f}")
                rows.append(row)
                
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "c_delta_mad_sweep_raw.csv", index=False)

if __name__ == "__main__":
    run_mad_sweep()
