import time
import numpy as np
import xgboost as xgb
from sert_xlearner.benchmarks.simulate_dgp import (
    simulate_level1_algebraic_sanity,
    simulate_level2_sparsity_stress,
    get_level3_ihdp,
    get_level4_acic_hostile,
    simulate_level5_imbalance,
    simulate_level6_unobserved_confounding,
    simulate_level7_heteroskedasticity,
    simulate_level8_weak_signal,
    simulate_level10_null_effect,
    simulate_level11_discontinuity
)
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

try:
    from econml.metalearners import XLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

try:
    import pymc as pm
    import pymc_bart as pmb
    PYMC_BART_AVAILABLE = True
except ImportError:
    PYMC_BART_AVAILABLE = False

def calculate_pehe(true_cate, pred_cate):
    return np.sqrt(np.mean((true_cate - pred_cate)**2))

def evaluate_coverage(true_cate, ci_lower, ci_upper):
    return np.mean((true_cate >= ci_lower) & (true_cate <= ci_upper))

def run_baselines(X, Y, W, true_cate):
    results = {}
    if ECONML_AVAILABLE:
        print("    Running EconML XLearner baseline (Frequentist Competitor)...")
        frequentist_xlearner = XLearner(
            models=xgb.XGBRegressor(max_depth=3, learning_rate=0.1),
            propensity_model=xgb.XGBClassifier(max_depth=3, learning_rate=0.1)
        )
        start_time = time.time()
        frequentist_xlearner.fit(Y, W, X=X)
        frequentist_cate_preds = frequentist_xlearner.effect(X)
        pehe = calculate_pehe(true_cate, frequentist_cate_preds)
        time_taken = time.time() - start_time
        print(f"    EconML PEHE: {pehe:.4f}, Time: {time_taken:.2f}s")
        results['EconML_XLearner'] = {'PEHE': pehe, 'Time': time_taken}
    else:
        print("    EconML not installed. Skipping EconML baseline.")
        
    if PYMC_BART_AVAILABLE:
        print("    Running PyMC-BART baseline (Traditional Bayesian)...")
        start_time = time.time()
        X_with_W = np.column_stack([X, W])
        with pm.Model() as model:
            mu = pmb.BART("mu", X_with_W, Y, m=50) # 50 trees
            sigma = pm.HalfNormal("sigma", sigma=1)
            pm.Normal("y", mu=mu, sigma=sigma, observed=Y)
            idata = pm.sample(500, tune=500, chains=2, compute_convergence_checks=False, progressbar=False)
        time_taken = time.time() - start_time
        print(f"    BART Time: {time_taken:.2f}s")
        results['PyMC_BART'] = {'Time': time_taken}
    else:
        print("    PyMC-BART not installed. Skipping PyMC-BART baseline.")
    return results

def run_sert_xlearner(X, Y, W, true_cate, name, X_infer=None, outcome_params=None, propensity_params=None, prior_scale=10.0, return_variance=False):
    print(f"\\n================ Scenario: {name} ================")
    
    out_p = outcome_params if outcome_params is not None else {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 50}
    prop_p = propensity_params if propensity_params is not None else {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 50}
    
    model = TargetedBayesianXLearner(
        outcome_model_params=out_p,
        propensity_model_params=prop_p,
        n_splits=2,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        prior_scale=prior_scale
    )

    print("    Running Sert-Targeted Bayesian X-Learner...")
    start_time = time.time()
    
    model.fit(X, Y, W, X_infer=X_infer)
    
    N = len(X)
    predict_target = X_infer if X_infer is not None else np.ones((N, 1))
    mean_cate, ci_lower, ci_upper = model.predict(predict_target)
    
    time_taken = time.time() - start_time

    pehe = calculate_pehe(true_cate, mean_cate)
    coverage = evaluate_coverage(true_cate, ci_lower, ci_upper)

    print(f"    Sert X-Learner PEHE:            {pehe:.5f}")
    if np.any(true_cate == 0.0) and "Null" not in name:
        pass
    print(f"    Sert X-Learner Exact Coverage:  {coverage:.4f} (Ideal: ~0.95)")
    print(f"    Sert X-Learner Execution Time:  {time_taken:.2f}s")
    
    if return_variance:
        cate_samples = np.dot(predict_target, model.mcmc_samples["beta"].T)
        posterior_variance = np.mean(np.var(cate_samples, axis=1)) 
        return posterior_variance

def benchmark_standard_ladder():
    print("\\n\\n*** PHASE 1: THE FOUR-LEVEL LADDER ***\\n")
    X, Y, W, tau = simulate_level1_algebraic_sanity(N=1000)
    run_sert_xlearner(X, Y, W, tau, "Level 1: Algebraic Sanity Check")
    run_baselines(X, Y, W, tau)

    X, Y, W, tau = simulate_level2_sparsity_stress(N=500, P=2000)
    X_infer_l2 = np.column_stack((np.ones(X.shape[0]), X[:, 3], X[:, 4]**2))
    run_sert_xlearner(X, Y, W, tau, "Level 2: The Sparsity Stress Test (P >> N)", X_infer=X_infer_l2)

    X, Y, W, tau = get_level3_ihdp()
    run_sert_xlearner(X, Y, W, tau, "Level 3: The Academic Standard (IHDP)", X_infer=X)

    X, Y, W, tau = get_level4_acic_hostile()
    X_infer_l4 = np.column_stack((np.ones(X.shape[0]), X[:, 0]*X[:, 1], X[:, 3]))
    run_sert_xlearner(X, Y, W, tau, "Level 4: The 'Reviewer 2' Hostile Suite (ACIC Proxy)", X_infer=X_infer_l4)

def benchmark_boundary_gauntlet():
    print("\\n\\n*** PHASE 2: THE BOUNDARY GAUNTLET ***\\n")
    X, Y, W, tau = simulate_level5_imbalance(N=1000)
    run_sert_xlearner(X, Y, W, tau, "Level 5: Severe Treatment Imbalance")
    
    X, Y, W, tau = simulate_level6_unobserved_confounding(N=1000, confounding_strength=2.0)
    run_sert_xlearner(X, Y, W, tau, "Level 6: Unobserved Confounding (Expected Bias!)")
    
    X, Y, W, tau = simulate_level7_heteroskedasticity(N=1000)
    run_sert_xlearner(X, Y, W, tau, "Level 7: Severe Heteroskedasticity")
    
    X, Y, W, tau = simulate_level8_weak_signal(N=200)
    print("    -> Run A: Tight Prior N(0, 0.1)")
    run_sert_xlearner(X, Y, W, tau, "Level 8-A: Weak Signal (Tight Prior)", prior_scale=0.1)
    print("    -> Run B: Flat Prior N(0, 10.0)")
    run_sert_xlearner(X, Y, W, tau, "Level 8-B: Weak Signal (Flat Prior)", prior_scale=10.0)

def benchmark_architects_crucible():
    print("\\n\\n*** PHASE 3: THE ARCHITECT'S CRUCIBLE ***\\n")
    # LEVEL 9
    X, Y, W, tau = simulate_level2_sparsity_stress(N=500, P=2000)
    X_infer_l2 = np.column_stack((np.ones(X.shape[0]), X[:, 3], X[:, 4]**2))
    sabotaged_params = {'max_depth': 1, 'n_estimators': 10, 'learning_rate': 0.01}
    perfect_params = {'max_depth': 5, 'n_estimators': 200, 'learning_rate': 0.1}
    run_sert_xlearner(X, Y, W, tau, "Level 9-A: Double Robustness (Sabotaged Outcome)", X_infer=X_infer_l2, outcome_params=sabotaged_params, propensity_params=perfect_params)
    run_sert_xlearner(X, Y, W, tau, "Level 9-B: Double Robustness (Sabotaged Propensity)", X_infer=X_infer_l2, outcome_params=perfect_params, propensity_params=sabotaged_params)
    
    # LEVEL 10
    X, Y, W, tau = simulate_level10_null_effect(N=1000)
    run_sert_xlearner(X, Y, W, tau, "Level 10: The Null Effect")
    
    # LEVEL 11
    X, Y, W, tau = simulate_level11_discontinuity(N=1000)
    X_infer_l11 = np.column_stack((np.ones(X.shape[0]), np.where(X[:, 0] > 0, 1, 0))) 
    run_sert_xlearner(X, Y, W, tau, "Level 11: The Discontinuity", X_infer=X_infer_l11)
    
    # LEVEL 12
    print("\\n================ Scenario: Level 12: Empirical Bernstein-von Mises ================")
    ns = [100, 500, 1000, 5000]
    variances = []
    for n in ns:
        X, Y, W, tau = simulate_level1_algebraic_sanity(N=n)
        v = run_sert_xlearner(X, Y, W, tau, f"BvM Scaling N={n}", return_variance=True)
        variances.append(v)
        
    print("\\n    --- BvM Scaling Law Proof ---")
    for n, v in zip(ns, variances):
        print(f"    N = {n:<5} => Posterior Variance: {v:.6f}")
    
    log_v = np.log(variances)
    log_inv_N = np.log(1 / np.array(ns))
    slope, intercept = np.polyfit(log_inv_N, log_v, 1)
    print(f"    BvM Slope: {slope:.4f} (Ideal: ~1.000)")

if __name__ == "__main__":
    benchmark_standard_ladder()
    benchmark_boundary_gauntlet()
    benchmark_architects_crucible()
