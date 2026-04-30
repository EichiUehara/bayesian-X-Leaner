import numpy as np
import pandas as pd
from pathlib import Path
from benchmarks.dgps import whale_dgp
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
import time

RESULTS_DIR = Path(__file__).parent / "results"

def run_eigenvalue_and_ridge():
    rows = []
    N = 1000
    density = 0.20
    n_whales = int(N * density)
    c_whale = 1.34
    
    # We will test p=6 (standard), p=21, and p=50
    # For p>6, we will generate extra features
    print("Running Eigenvalue and Ridge...")
    for p in [6, 21, 50]:
        for seed in range(5):
            X, Y, W, tau = whale_dgp(N=N, P=p, tau=2.0, n_whales=n_whales, seed=seed)
            
            # Fit nuisance and get residuals
            model = TargetedBayesianXLearner(
                contamination_severity="severe",
                n_splits=2,
                num_warmup=100, # fast for just getting to imputation
                num_samples=100,
                num_chains=1,
                random_state=seed
            )
            
            # We don't need full MCMC, we just need the imputation residuals
            from sert_xlearner.core.orthogonalization import impute_and_debias
            out_pred_mu0, out_pred_mu1, out_pred_pi = model.nuisance_estimator.fit_predict(X, Y, W)
            
            treated_mask, control_mask, D1, D0, W_D1, W_D0 = impute_and_debias(
                Y, W, out_pred_mu0, out_pred_mu1, out_pred_pi,
                robust=True, tail_threshold=None, tail_alpha=None, use_overlap=False
            )
            
            # Combine
            X_all = np.vstack([X[treated_mask], X[control_mask]])
            # Add intercept
            X_all = np.hstack([np.ones((X_all.shape[0], 1)), X_all])
            
            # We need residuals relative to the true effect (or an OLS estimate)
            # For simplicity, we can use the robust D residuals relative to an intercept model
            # actually tau(x) = 2.0 (for all) since standard DGP has constant tau
            res_all = np.concatenate([D1 - 2.0, D0 - 2.0])
            
            # w(r) = (1 - r^2/c^2) * exp(-r^2 / 2c^2) / c^2
            # wait, c_whale in targeted_model is the scale. welsch_loss = c^2/2 * (1 - exp(-(r/c)^2))
            # so the first derivative w.r.t r is r * exp(-(r/c)^2)
            # second derivative is (1 - 2(r/c)^2) * exp(-(r/c)^2)
            # Let's compute weights
            r = res_all
            c = c_whale
            weights = (1 - 2 * (r/c)**2) * np.exp(-(r/c)**2)
            
            I_hat = (X_all.T @ np.diag(weights) @ X_all) / len(X_all)
            
            # eigenvalues
            evals = np.linalg.eigvalsh(I_hat)
            min_eval = evals.min()
            
            # Ridge bias
            lam = 0.01
            tr_I = np.trace(I_hat)
            I_ridge = I_hat + lam * tr_I * np.eye(X_all.shape[1]) / X_all.shape[1]
            
            # Fraction of residuals > c/sqrt(2)
            frac_exceed = np.mean(np.abs(r) > c / np.sqrt(2))
            
            # Let's simulate some eta by taking inverse
            # eta is trace(I^-1 J I^-1) ... just check inverse norm
            try:
                inv_I = np.linalg.inv(I_hat)
                eta_I = np.trace(inv_I)
            except:
                eta_I = np.nan
                
            try:
                inv_ridge = np.linalg.inv(I_ridge)
                eta_ridge = np.trace(inv_ridge)
            except:
                eta_ridge = np.nan
                
            bias = np.abs(eta_ridge - eta_I) / np.abs(eta_I) if not np.isnan(eta_I) and eta_I > 0 else np.nan
            
            rows.append({
                "p": p,
                "seed": seed,
                "min_eval": min_eval,
                "frac_exceed": frac_exceed,
                "bias": bias,
                "eta_I": eta_I,
                "eta_ridge": eta_ridge
            })
            print(f"p={p}, seed={seed}, min_eval={min_eval:.4f}, bias={bias:.4f}")
            
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "eigenvalue_ridge_bias_raw.csv", index=False)

if __name__ == "__main__":
    run_eigenvalue_and_ridge()
