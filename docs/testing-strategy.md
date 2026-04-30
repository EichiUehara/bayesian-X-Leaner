# Targeted Bayesian X-Learner: Experimentation & Testing Strategy

This document outlines the systematic Simulation-Driven Development (SDD) strategy designed to empirically validate the Sert-Targeted Bayesian X-Learner. The experimentation framework guarantees properties involving regularization robustness, structural debiasing, Heavy-Tail (Whale) resilience, and uncertainty quantification.

## 1. Unit Testing Suite (`tests/`)

Three critical tests evaluate individual mechanistic layers of the framework:

1. **Algebraic Orthogonalization Test (`test_algebraic_orthogonalization.py`)**  
   Validates that baseline sample-splitting and phase estimation function without errors. It functions as a base sanity check ensuring valid dimensional mapping when evaluating causal pseudo-outcomes analytically mapped to simple linear DGP environments.

2. **MCMC Convergence Diagnostics (`test_mcmc_convergence.py`)**  
   Ensures the validity of Bayesian uncertainty intervals. Using standard Gelman-Rubin ($\hat{R}$) convergence checks and Effective Sample Size (ESS) metrics, it validates that `numpyro`'s NUTS trajectory successfully explores the low-dimensional structural parameter space without pathological mixing behavior.

3. **Regularization Leakage Test (`test_regularization_leakage.py`)**  
   Assesses double robustness limits under extreme underfitting scenarios. This test heavily penalizes the XGBoost parameters (`max_depth=1`, `n_estimators=3`) for both outcome and propensity learners to enforce structural failure in the initial learning phases. This aims to explicitly surface finite-sample boundaries on the unidentifiability of CATE when no remaining covariate relationships are maintained in Phase 3.

---

## 2. The Comprehensive Benchmark Suite (`sert_xlearner/benchmarks/run_benchmarks.py`)

A grueling 12-level progressive benchmark suite is utilized to progressively stress the targeted architecture against baseline models like standard `EconML XLearner` and regular `PyMC-BART`.

### Phase 1: The Four-Level Ladder
* **Level 1: Algebraic Sanity Check**  
  A completely linear, noise-free domain verifying optimal PEHE (Precision in Estimating Heterogeneous Effects).
* **Level 2: The Sparsity Stress Test (P >> N)**  
  Evaluates algorithm degradation under high-dimensional data (e.g., $P=2000$, $N=500$), demonstrating the benefit of isolating nuisance extraction from MCMC convergence properties.
* **Level 3: The Academic Standard (IHDP)**  
  Simulated bounds rooted in infant hazard developmental outcomes representing traditional NeurIPS/KDD methodological evaluations.
* **Level 4: The 'Reviewer 2' Hostile Suite (ACIC Proxy)**  
  High-noise non-linear mapping environment testing whether structural MCMC properly models variance despite significant initial measurement volatility.

### Phase 2: The Boundary Gauntlet
* **Level 5: Severe Treatment Imbalance**  
  A stress test pushing observation limits where $W=1$ dominates to ensure inverse propensity weights remain stable under Phase 2 targeted updates.
* **Level 6: Unobserved Confounding (Expected Bias)**  
  Evaluates breakdown bounds: verifying estimator collapse against strict identifiability assumptions.
* **Level 7: Severe Heteroskedasticity**  
  Tests the ability of the Bayesian framework to recover conditional variance over the sample dimensions optimally without shattering Hamiltonian trajectories.
* **Level 8: Weak Signal Identification**  
  Varies Bayesian priors (Tight $N(0, 0.1)$ vs. Weak $N(0, 10.0)$) to quantify sensitivity bounds against minute real-world CATE distributions.

### Phase 3: The Architect's Crucible
* **Level 9: Double Robustness Isolation**  
  Iteratively sabotages the outcome model, and sequentially the propensity model, ensuring the TMLE-infused pseudo-outcomes debias properly on a standalone basis if the counterpart maintains functional accuracy.
* **Level 10: The Null Effect**  
  Checks precision targeting centered on EXACTLY $0.0$, a strict test ensuring artificial CATE variance isn't injected structurally by pseudo-outcome conversions.
* **Level 11: The Discontinuity**  
  Replicates Regression Discontinuity behavior by mapping distinct causal clusters to abrupt feature barriers. 
* **Level 12: Empirical Bernstein-von Mises (BvM)**  
  Proves theoretically that Posterior Variance explicitly follows identical scaling laws ($O(N^{-1})$) akin to normal maximum distributions, serving as robust guarantees against extreme structural mis-specifications.

---

## 3. Supplementary Outlier & Tail Handling (Validation)

A self-contained validation methodology specifically validates performance regarding Heavy-Tailed distributions via dynamic `c_whale` sizing and Welsch-divergence loss.

* Tests explicitly inject structurally non-convective anomalies (+100.0) mapping the outcome surface away from local equilibria. 
* Validates that standard estimation tracks exponentially biased pseudo-outcomes whereas tuning robust Pseudo-Likelihood settings limits gradient accumulation relative to median scaling. Covariates exhibiting unbounded growth must be mitigated using external Peak-Over-Threshold (POT) pre-estimators prior to structural analysis.
