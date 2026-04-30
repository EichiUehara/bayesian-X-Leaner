# Research Expansion Plan: Algorithm Replacements and Stress Testing

This document tracks the strategic roadmap for swapping current baseline algorithms with state-of-the-art causal and probabilistic frameworks to elevate the *Sert-Targeted Bayesian X-Learner* for final NeurIPS/KDD submission. It defines the required mathematical stress tests necessary to empirically validate each architectural upgrade.

## Phase 1: Nuisance Quarantine
* **Objective:** Estimate nuisance parameters without overfitting.
* **Current Baseline:** XGBoost with K-Fold Cross-Fitting.
* **Potential Replacements:** BART (PyMC-BART), Causal Forests, Dragonnet (`causalml`), TabPFN.
* **The Validation Test:** "RIC Stress Test" (High-Dimensional Confounding)
    * **Execution:** Generate synthetic data with 100+ covariates (only 3 true confounders, rest noise).
    * **Metrics:** Covariate Balance, First-Stage MSE.
    * **Failure Condition:** Algorithm over-regularizes and drops true confounders, causing Regularization-Induced Confounding (RIC) and biased CATE.
* **Required Paper Framing:** "To validate our choice of base learner in Phase 1, we conduct a High-Dimensional Confounding stress test. We demonstrate that [New Algorithm] achieves superior out-of-fold accuracy compared to standard Gradient Boosting, ensuring the cross-imputed residuals passed to Phase 2 are free from prior dogmatism."

## Phase 2: Imputation & Debiasing
* **Objective:** Impute counterfactuals in imbalanced datasets.
* **Current Baseline:** X-Learner with Density Ratio Weighting.
* **Potential Replacements:** R-Learner (EconML), DR-Learner (AIPW), Overlap Weights (DoWhy), Entropy Balancing.
* **The Validation Test:** "Few Placebo" Imbalance Test
    * **Execution:** Artificially skew dataset so treatment group is 99% and control is 1% (propensities approach 0.99 or 0.01).
    * **Metrics:** Variance of Pseudo-Outcomes ($D_1$, $D_0$).
    * **Failure Condition:** Inverse propensity weights cause pseudo-outcomes to shoot to infinity when common support is dangerously thin.
* **Required Paper Framing:** "We subject the Phase 2 imputation mechanism to a 'Few Placebo' extreme imbalance test. By replacing standard density weights with [New Algorithm], we show that the pseudo-outcome variance remains bounded even when propensity scores approach the deterministic limits of 0 or 1."

## Phase 3: The Bayesian Update
* **Objective:** Translate frequentist outcomes to probabilistic representation.
* **Current Baseline:** NUTS MCMC (NumPyro) with Welsch/$\gamma$-divergence.
* **Potential Replacements:** Stochastic Variational Inference (SVI) via NumPyro, Student-t Likelihoods, Density Power Divergence (DPD).
* **The Validation Test:** "Whale Injection" Test (Outlier Smearing)
    * **Execution:** Multiply the outcome of 3 control units by 1,000 ("Whales"). 
    * **Metrics:** Posterior Shift, Credible Interval Coverage.
    * **Failure Condition:** MCMC posterior mean shifts significantly toward the outliers due to an unbounded influence function.
* **Required Paper Framing:** "To prove the structural robustness of Phase 3, we execute a 'Whale Injection' test. We replace the standard Gaussian likelihood with [New Algorithm] and inject synthetic extreme outliers into the minority class. We demonstrate that our chosen algorithm strictly bounds the influence function, resulting in a zero posterior shift (i.e., eliminating Outlier Smearing)."

## Phase 4: Tail Refinement (Extreme Value Theory)
* **Objective:** Predict treatment effects in unobserved extreme domains.
* **Current Baseline:** Hill Estimator (power-law representation).
* **Potential Replacements:** Peaks-Over-Threshold (POT-GPD via PyExtremes), Block Maxima (GEV), TIEE Framework.
* **The Validation Test:** "Deep Tail Extrapolation" Test
    * **Execution:** Train tail parameters on 90th-95th percentile data.
    * **Metrics:** Extrapolation MSE vs. Winsorization baseline in the >99.9th percentile.
    * **Failure Condition:** Model fails to extrapolate extreme tail shape, performing worse than naive threshold clipping (Winsorization).
* **Required Paper Framing:** "Finally, we benchmark the Phase 4 EVT layer using a Deep Tail Extrapolation task. Relying on [New Algorithm], we estimate the tail shape parameters using moderate-frequency data. We show that this algorithm successfully predicts treatment effects in the unobserved >99th percentile, reducing Extrapolation MSE by X% compared to standard threshold clipping."
