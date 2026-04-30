# Sert-Targeted Bayesian X-Learner Implementation Details

This document outlines the current technical state of the robust "Sert-Targeted Bayesian X-Learner" implementation, detailing the mathematics, algorithms, and structure of the estimation pipeline.

## 1. High-Level Architecture Overview

The `TargetedBayesianXLearner` implements a three-phase procedure to isolate high-dimensional nuisance learning from low-dimensional targeted Bayesian parameter inference.

* **Phase 1: Nuisance Quarantine (Cross-Fitting)**
* **Phase 2: Orthogonalization & Robust Tail Normalization**
* **Phase 3: Targeted Bayesian Update via Normalized MCMC**

---

## 2. Phase 1: Nuisance Quarantine (`sert_xlearner/models/nuisance.py`)

To prevent overfitting and allow for correct asymptotic properties, all nuisance models are cross-fitted using a typical `K-fold` approach.

1. **Covariate and Treatment Space:** Models operate over $(X, Y, W)$.
2. **Base Learners:** We utilize `xgboost.XGBRegressor` to model conditional means $\mu_w(x) = \mathbb{E}[Y|X=x, W=w]$ and `xgboost.XGBClassifier` to model the propensity score $\pi(x) = P(W=1|X=x)$.
3. **Cross-fitting:** The dataset is split into $K=2$ folds by default. Nuisance parameters are fit on one fold and predicted on the other. Final out-of-fold estimates $\hat{\mu}_0$, $\hat{\mu}_1$, and $\hat{\pi}$ are propagated forward.

---

## 3. Phase 2: Orthogonalization (`sert_xlearner/core/orthogonalization.py`)

Once out-of-fold nuisance parameters are retrieved, we synthesize the targeted pseudo-outcomes $D_1$ and $D_0$ across treated and control units respectively.

### 3.1 Doubly Robust (TMLE/Kennedy-style) Transformation
To immunize the estimates against regularization leakage (often occurring when XGBoost models underfit), we modify the standard X-Learner pseudo-outcomes with IPW residual corrections to form fundamentally doubly robust targets:
* **Treated Target:** $D_1 = \hat{\mu}_1 - \hat{\mu}_0 + \frac{Y_{W=1} - \hat{\mu}_1}{\hat{\pi}}$
* **Control Target:** $D_0 = \hat{\mu}_1 - \hat{\mu}_0 - \frac{Y_{W=0} - \hat{\mu}_0}{1 - \hat{\pi}}$

### 3.2 Extreme Tail Normalization (EVT)
If `robust=True`, the pseudo-outcomes undergo Extreme Value Theory (EVT) handling. "Whale" observations that cross a pre-estimated `tail_threshold` ($t$) are structurally scaled down by the tail index $\alpha$:
* $D_{robust} = D / t^\alpha$ for any $|D| > t$.

*(Note: $\alpha$ and $t$ should theoretically be derived upstream using a Hill Estimator or POT).*

---

## 4. Phase 3: Bayesian Update (`sert_xlearner/inference/bayesian.py`)

The orthogonalized endpoints ($D_1$ and $D_0$) serve as conditionally independent signals used to update a low-dimensional structural parameter $\beta$.

### 4.1 Structural MCMC formulation
A `numpyro` model parameterizes an expected Conditional Average Treatment Effect (CATE) surface over structurally identifiable low-dimensional variables (`X_infer`):
$$ \hat{\tau}_1(X) = X_{\text{infer}} \beta $$
$$ \hat{\tau}_0(X) = X_{\text{infer}} \beta $$

### 4.2 Dynamic Welsch Loss / Gamma-Divergence
If `robust=True`, the model breaks away from classical L2 Gaussian minimization to explicitly bypass "Outlier Smearing". Rather than standard likelihood samples, a **Pseudo-Likelihood** evaluates via Welsch Loss ($\gamma$-divergence):
$$ \text{Loss}(Residual) = \frac{c^2}{2} \left[ 1 - \exp\left(-\left(\frac{Residual}{c}\right)^2\right) \right] $$

To preserve scale invariance:
1. The global Median Absolute Deviation (MAD) of the aggregated pseudo-outcomes is computed.
2. The Welsch tuning constant $c_\text{whale}$ is dynamically multiplied by `MAD / 0.6745` (the scale adjustment factor).
This ensures identical sampling geometries regardless of whether $Y$ represents single integers or billions of dollars.

---

## 5. Current Unresolved Edge Cases (Testing)

**Regularization Leakage Limit Theorem:**
The unit test `tests/test_regularization_leakage.py` consistently fails to estimate the correct ATE (predicting CATE $\approx$ 5.5 vs target 2.0).
* **Cause**: The test deliberately sets both `max_depth=1` and `n_estimators=3` for both outcome models AND propensity models. Because the structural MCMC is currently configured to fit only the intercept (`X_infer=None`), it completely drops structural covariates. 
* **Statistical Limitation**: When both Phase-1 nuisance learners are crippled, and the Phase-3 structural model contains no covariates to capture remaining confounding gradients, true Deconfounding is mathematically unidentifiable. No orthogonalization or weighting parameterization can recover unbiased outcomes when all layers of a causal pipeline completely fail to trace the observational dependencies.
