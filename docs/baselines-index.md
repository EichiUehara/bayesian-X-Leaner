# Master Index: Algorithmic Baselines and Robustness Framework

This document serves as the comprehensive repository of state-of-the-art (SOTA) baselines and validation metrics for the **Sert-Targeted Bayesian X-Learner**. Algorithms are categorized by their implementation phase (Phase 1–4) and mapped to specific robustness stress tests.

---

## 🏗 Phase 1: Nuisance Quarantine
*Objective: Isolate nuisance parameters ($\mu_0, \mu_1, \pi$) in high-dimensional settings without succumbing to Regularization-Induced Confounding (RIC).*

| Algorithm | Description | Purpose |
| :--- | :--- | :--- |
| **BART / BCF** | Bayesian Additive Regression Trees / Bayesian Causal Forests. Non-parametric sum-of-trees. | Prevents RIC and prior dogmatism via Bayesian backfitting. |
| **Dragonnet / TARNet** | Deep NN architectures with targeted regularization and shared representation layers. | Extracts confounders via representation learning in non-linear spaces. |
| **Causal Forests** | Extension of Random Forests with honest sample splitting and local orthogonalization. | Captures heterogeneity while preserving out-of-fold validity. |
| **ACE Procedures** | Higher-Order Robust Estimators utilizing structure-agnostic cumulant estimators. | Minimax guarantees for insensitivity to non-Gaussian nuisance errors. |
| **Spatial Deconfounders** | Conditional Variational Autoencoder (C-VAE) with spatial priors. | Reconstructs substitute confounders from multi-cause neighborhoods. |
| **CausalFM** | Causal Foundation Models leveraging large-scale pretraining. | Structure-agnostic mapping of high-dimensional observational data. |
| **BDML** | Bayesian Double Machine Learning. | Recovers effects from reduced-form covariance of $(Y, D)$ on $X$. |
| **LTCF** | Latent Causal Forest integrating efficient influence functions (EIF). | Identification under severe measurement error or misclassification. |
| **DP-CATE** | Neyman-orthogonal framework for $(\epsilon, \delta)$-differential privacy. | Ensures individual privacy while maintaining quasi-oracle efficiency. |

---

## ⚖ Phase 2: Imputation & Debiasing
*Objective: Impute counterfactuals and apply targeted debiasing (DR/Overlap) to stabilize weights under extreme imbalance.*

| Algorithm | Description | Purpose |
| :--- | :--- | :--- |
| **R-Learner** | Robinson Transformation-based semiparametric meta-learner. | Residualizes treatment space to bypass weighting instability. |
| **Entropy Balancing** | Convex optimization matching covariate moments (Kullback-Leibler divergence). | Prevents weight explosion under extreme imbalance. |
| **DR Pseudo-outcomes** | Augmented Inverse Probability Weighting (AIPW) formulation. | Quasi-oracle convergence if *either* nuisance model is valid. |
| **BICauseTree** | Interpretable balancing method based on decision trees. | Excludes positivity-violating subgroups; defines inferable target population. |
| **Graph ML DR** | Integrates graph ML with double machine learning for network data. | Estimates direct and peer effects while addressing spillover confounding. |
| **DVDS Estimators** | Doubly-Valid / Doubly-Sharp Estimators for marginal sensitivity bounds. | Consistent estimates for sharp bounds under binary outcomes. |
| **UPO** | Unified Propensity Optimization minimax framework. | Decouples bias, variance, and calibration for stable sparse weighting. |
| **APWCF** | Adversarial Propensity Weighting utilizing dynamic factors. | Clears selection bias and reduces variance during cross-fill imputation. |
| **Coarse IPW (CIPW)** | Space partitioning into disjoint clusters for "coarse" propensity scores. | Stabilizes estimates when scores $\pi(X) \to \{0, 1\}$. |
| **LBC** | Local Balance with Calibration via Neural Networks. | Enforces local covariate balance with flexible function approximation. |

---

## 🔮 Phase 3: The Bayesian Update
*Objective: Approximate posterior distributions of CATE while strictly bounding the influence of extreme outliers.*

| Algorithm | Description | Purpose |
| :--- | :--- | :--- |
| **SVI** | Stochastic Variational Inference for scalable posterior approximation. | Accelerates convergence and bypasses MCMC bottlenecks. |
| **Student-t Likelihoods** | Robust heavy-tailed parameterization (fixed or learned $\nu$). | Absorbs symmetric outliers without shifting the posterior mean. |
| **Welsch Loss ($\alpha, \beta, \gamma$)** | Robust pseudo-Bayesian loss (e.g., redescending influence). | Strictly bounds the influence function to zero-out "Whales." |
| **GBI** | Generalised Bayesian Inference using a Gibbs posterior. | Scales loss discrepancy via a tuned learning rate $\omega$. |
| **Posterior Coupling** | Independent posteriors coupled via entropic tilting. | Prevents "feedback problems" in doubly robust moment conditions. |
| **RVI / RNPs** | Robust Variational Inference / Rényi Neural Processes. | Scaled density ratios dampening the effect of misspecified priors. |

---

## 🐋 Phase 4: Tail Refinement (EVT)
*Objective: Model asymptotic decay and capture treatment effects in the unobserved deep tail.*

| Algorithm | Description | Purpose |
| :--- | :--- | :--- |
| **POT via GPD** | Peaks-Over-Threshold modeling excesses above threshold $t$. | Accurately parameters asymptotic decay for deep extrapolation. |
| **Block Maxima (GEV)** | Modeling maximums within temporal/spatial segments. | Defines GEV distribution bounds for clustered extrema. |
| **NETE Estimator** | Normalized Extreme Treatment Effect via Multivariate Regular Variation. | Measures treatment effects conditionally on $\|U\| > t$. |
| **EVO** | Extreme Value Policy Optimization. | Safety-critical quantile optimization for cost tail distributions. |
| **EFDiff** | Extreme-Frequency Diffusion via phase alignment modules. | Generative modeling for data augmentation of extreme distributions. |
| **Liquid Tail** | Mutable prefix for continuous autoregressive predictive tails. | Suppresses degeneration and improves diversity in predictive tails. |

---

## 🔗 Repository Reference List

### 📦 Phase 1: Nuisance & Identification
* **BART / BCF**: [jaredsmurray/bcf](https://github.com/jaredsmurray/bcf) \| [skdeshpande91/flexBCF](https://github.com/skdeshpande91/flexBCF)
* **Dragonnet / TARNet**: [claudiashi57/dragonnet](https://github.com/claudiashi57/dragonnet) \| [farazmah/dragonnet-pytorch](https://github.com/farazmah/dragonnet-pytorch)
* **Causal Forests**: [haytug/causalfe](https://github.com/haytug/causalfe) \| [microsoft/EconML](https://github.com/microsoft/EconML)
* **ACE Procedures**: [JikaiJin/ACE](https://github.com/JikaiJin/ACE)
* **Causal Foundation Models**: [yccm/CausalFM](https://github.com/yccm/CausalFM) \| [yccm/CausalFM-toolkit](https://github.com/yccm/CausalFM-toolkit)

### 📦 Phase 2: Weighting & Balancing
* **R-Learner**: [dscolby/CausalELM.jl](https://github.com/dscolby/CausalELM.jl) \| [uber/causalml](https://github.com/uber/causalml) \| [s-sairam/final-stage-bottleneck](https://github.com/s-sairam/final-stage-bottleneck)
* **Entropy Balancing**: [uscensusbureau/entropy-balance-weighting](https://github.com/uscensusbureau/entropy-balance-weighting) \| [EddieYang211/ebal-py](https://github.com/EddieYang211/ebal-py) \| [bvegetabile/entbal](https://github.com/bvegetabile/entbal)
* **BICauseTree**: [IBM-HRL-MLHLS/BICause-Trees](https://github.com/IBM-HRL-MLHLS/BICause-Trees)
* **Graph ML DR**: [BaharanKh/GDML](https://github.com/BaharanKh/GDML)
* **Unified Propensity Optimization (UPO)**: [yhc-666/UPO](https://github.com/yhc-666/UPO)
* **balnet (CBPS)**: [erikcs/balnet](https://github.com/erikcs/balnet)

### 📦 Phase 3: Robust Inference
* **Mixture Weights Alpha-VI**: [kdaudel/MixtureWeightsAlphaVI](https://github.com/kdaudel/MixtureWeightsAlphaVI)
* **Adversarial Alpha-VI (AADM)**: [simonrsantana/AADM](https://github.com/simonrsantana/AADM)
* **Generalised Bayesian Inference (GBI)**: [aggelisalexopoulos/robust-bayesian-causal-inference](https://gitlab.com/aggelisalexopoulos/robust-bayesian-causal-inference)
* **Rényi Neural Processes (RNPs)**: [csiro-funml/renyineuralprocesses](https://github.com/csiro-funml/renyineuralprocesses)

### 📦 Phase 4: Extreme Value Theory
* **pyextremes (POT/GEV/GPD)**: [georgebv/pyextremes](https://github.com/georgebv/pyextremes)
* **EVO**: [ShiqingGao/EVO](https://github.com/ShiqingGao/EVO)

### 🛠 General Causal Ecosystems
* **DoubleML**: [DoubleML/doubleml-for-py](https://github.com/DoubleML/doubleml-for-py)
* **Causallib (IBM)**: [IBM/causallib](https://github.com/IBM/causallib)
* **Causal Inference Laboratory (Vector Institute)**: [VectorInstitute/Causal_Inference_Laboratory](https://github.com/VectorInstitute/Causal_Inference_Laboratory)

---

## 🛡 Robustness Stress Test Requirements

| Test Name | Focus | Key Metric | Failure Condition |
| :--- | :--- | :--- | :--- |
| **RIC Stress Test** | Phase 1 | Covariate Balance / MSE | Drop of true confounders in high-dim noise. |
| **"Few Placebo" Test** | Phase 2 | Pseudo-outcome Variance | Weight explosion when $\pi(X) \to 1.0$. |
| **"Whale Injection" Test** | Phase 3 | Posterior Shift & CI Coverage | Posterior mean follows extreme asymmetric outliers. |
| **Deep Tail Extrapolation** | Phase 4 | Extrapolation MSE | Structural collapse on $>99^{\text{th}}$ percentile data. |

---

## 📝 Paper Framing for Ablation Strategies

* **Phase 1 (Nuisance)**: *"To validate our choice of base learner, we demonstrate that [Algorithm] achieves superior out-of-fold accuracy in high-dimensional confounding tests (RIC Stress), ensuring residuals are free from prior dogmatism."*
* **Phase 2 (Weighting)**: *"By replacing standard density weights with [Algorithm], we show that pseudo-outcome variance remains bounded even when propensity scores approach deterministic limits."*
* **Phase 3 (Bayesian)**: *"We demonstrate that our chosen Bayesian solver strictly bounds the influence function (Welsch redescending pseudo-likelihood), resulting in a zero posterior shift despite synthetic extreme outliers (Whale Injection)."*
* **Phase 4 (EVT)**: *"Relying on [Algorithm], we show successful prediction of treatment effects in the unobserved $>99$th percentile, reducing Extrapolation MSE compared to threshold clipping."*
