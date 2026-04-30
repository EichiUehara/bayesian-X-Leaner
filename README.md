# Bayesian X-Learner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Bayesian X-Learner with calibrated posterior CATE inference under
heavy-tailed outcomes. The package synthesises the standard X-Learner
with cross-fitted doubly robust pseudo-outcomes
\([Kennedy 2020](https://arxiv.org/abs/2004.14497)\) and a Welsch
redescending pseudo-likelihood (generalised Bayes posterior with
trace-formula learning-rate calibration) to deliver
heterogeneous-effect point estimates, calibrated uncertainty, and
bounded-influence robustness in a single estimator.

## Paper

> *Bayesian X-Learner: Calibrated Posterior Inference for
> Heterogeneous Treatment Effects under Heavy-Tailed Outcomes.*
> Eichi Uehara. arXiv preprint, 2026.

The full LaTeX source is in [paper/](paper/) and the camera-ready
arXiv bundle in [bayesian_xlearner_arxiv.zip](bayesian_xlearner_arxiv.zip).
Reproduce all empirical figures with the scripts in
[benchmarks/](benchmarks/) (see [REPRODUCE.md](REPRODUCE.md)).

## Architecture

The project is structured into three phases, modularized to prevent data leakage and facilitate testing:

1. **Phase 1: Nuisance Quarantine (`sert_xlearner/models/nuisance.py`)**  
   Uses high-dimensional ML algorithms (e.g., XGBoost) constrained within a K-fold cross-fitting process to estimate $\hat{\mu}_0$, $\hat{\mu}_1$, and the propensity score $\hat{\pi}(X)$.

2. **Phase 2: Imputation & Targeted Debiasing (`sert_xlearner/core/orthogonalization.py`)**  
   Computes the X-Learner pseudo-outcomes ($D_1$, $D_0$) on the hold-out fold and injects the Sert et al. density ratio weights based on $\hat{\pi}(X)$ to orthogonalize the regularization bias.

3. **Phase 3: Targeted Bayesian Update (`sert_xlearner/inference/bayesian.py`)**  
   Deploys `numpyro`'s NUTS sampler on the lower-dimensional parameter space $\beta$ of the targeted likelihood, utilizing the debiased pseudo-outcomes and weights from Phase 2.

## Installation

You can install the package and its dependencies locally via pip:

```bash
pip install -e '.[test]'          # core + unit tests
pip install -e '.[test,benchmark]' # also installs econml / pymc-bart for full benchmark suite
```

## Running Tests

The test suite is split into a deterministic regression core and a
broader set of stress / probe tests:

```bash
# Fast, deterministic regression suite (9 tests, ~25 s).
# Pins the default configuration and the contamination_severity API.
pytest tests/test_default_config_regression.py

# Full suite (~3 min). Includes scenario stress tests
# (whale injection, extreme imbalance, deep tail, etc.) — some of
# these have aggressive thresholds and may flake on unusual hardware.
pytest tests/
```

Three crucial mechanism-level tests guarantee statistical validity:

1. **Algebraic Orthogonalization Test:** weight ratios correctly cancel under perfect specification.
2. **Regularization Leakage Test:** Phase 3 posterior centers correctly despite artificial underfitting / L1-penalties in Phase 1 models.
3. **MCMC Convergence Diagnostics:** Gelman-Rubin ($\hat{R}$) and Effective Sample Size (ESS) thresholds.

## Empirical validation

Sixteen benchmark experiments back the implementation's stability and
positioning claims. See
[benchmarks/results/README.md](benchmarks/results/README.md) for an
index and [REPRODUCE.md](REPRODUCE.md) for the exact commands.

| Headline claim | Evidence |
|---|---|
| Stable & converged (100 % R̂ < 1.05, ESS > 200) | [STABILITY_SUMMARY.md](benchmarks/results/STABILITY_SUMMARY.md) |
| Robust machinery is load-bearing (RMSE 0.08 on whale vs 100+ for every Gaussian-likelihood baseline) | [STABILITY_SUMMARY.md](benchmarks/results/STABILITY_SUMMARY.md), [bart_comparison.md](benchmarks/results/bart_comparison.md) |
| Overlap weights close the imbalance coverage gap | [overlap_experiment.md](benchmarks/results/overlap_experiment.md) |
| Basis-sensitive on CATE — strong when correctly specified | [nonlinear_cate.md](benchmarks/results/nonlinear_cate.md) |
| Competitive on real data (IHDP ε_ATE 0.16) | [ihdp_benchmark.md](benchmarks/results/ihdp_benchmark.md) |
| Welsch ≈ Student-T under clean Huber nuisance (§7's strict ordering was an XGB-MSE artefact); **Gaussian** still fails on whale, so *some* robust likelihood is needed | [component_ablation_catboost_huber.md](benchmarks/results/component_ablation_catboost_huber.md) |
| The `c_whale = 1.34` default sits at the calibrated plateau; `c ≥ 2` collapses whale coverage | [c_whale_sensitivity.md](benchmarks/results/c_whale_sensitivity.md) |
| **√N consistency on whale is restored by Huber nuisance** (slope −0.59 vs +1.29 under XGB-MSE); §9's breakdown was nuisance-contamination, not architectural | [sample_size_scaling_catboost_huber.md](benchmarks/results/sample_size_scaling_catboost_huber.md) |
| **Contamination tolerance is ~20-25 % whale density** under the CatBoost-Huber default (was ≤ 1 % with XGB-MSE); breakdown begins at 30 % as per-leaf whale concentration exceeds Huber's clipping capacity | [whale_density_catboost_huber.md](benchmarks/results/whale_density_catboost_huber.md) |
| **Three levers restore robustness at high contamination (XGB-MSE fallback)**: deeper nuisance trees (`max_depth=10`), disabling MAD rescaling of `c_whale`, and `prior_scale=2.0` — together recover coverage 1.00 at 20 % whale density | [nuisance_depth.md](benchmarks/results/nuisance_depth.md), [mad_rescaling_and_prior.md](benchmarks/results/mad_rescaling_and_prior.md) |
| **Implementation finding**: MAD rescaling of `c_whale` is the silent mechanism behind §11's catastrophic numbers under XGB-MSE — MAD itself gets contaminated, inflating effective c to ≈3670 and defeating Welsch clipping. Now opt-out via `mad_rescale=False` | [mad_rescaling_and_prior.md](benchmarks/results/mad_rescaling_and_prior.md) |
| **CatBoost + Huber nuisance subsumes the downstream tangle under contamination** — RMSE 0.06 at 20 % whale density (vs 1543 XGB-MSE, 25000×) with zero Bayesian-layer changes | [robust_nuisance.md](benchmarks/results/robust_nuisance.md), [whale_density_catboost_huber.md](benchmarks/results/whale_density_catboost_huber.md) |
| **Huber δ tuning** — `loss_function="Huber:delta=0.5"` gives bias −0.009, RMSE 0.058, coverage **1.00** at 20 % whale density | [nuisance_loss_sweep.md](benchmarks/results/nuisance_loss_sweep.md) |
| **§17 caveat: Huber costs accuracy on clean data (theory: 27 %, empirically 3× √PEHE on IHDP)** — Huber 1964 ARE at δ=0.5 is 0.79 (27 % variance penalty); amplified to 3× √PEHE on IHDP by bias-coherence + finite-N + tree-structure distortion. Our δ=0.5 is tuned for ~40 % contamination; canonical Huber (δ=1.345) is right for ~5 %; loosen δ if mild contamination expected | [ihdp_benchmark.md](benchmarks/results/ihdp_benchmark.md), [EXTENSIONS.md §17.1-§17.4](benchmarks/results/EXTENSIONS.md) |

## Usage

```python
import numpy as np
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

# Generate synthetic dataset
N, P = 500, 10
X = np.random.normal(0, 1, size=(N, P))
W = np.random.binomial(1, 0.5, size=N)
Y = W * 2.0 + X[:, 0] + np.random.normal(0, 0.1, size=N)

# Defaults: CatBoost nuisance with Huber(delta=0.5), depth=4, 150 trees.
# See §16 of benchmarks/results/EXTENSIONS.md — bias -0.009, coverage
# 1.00 at 20 % whale density. Falls back to XGBoost if CatBoost isn't
# installed. Pass robust=True for the full Welsch pseudo-likelihood.
#
# §17 caveat: Huber trades clean-data accuracy for contamination
# robustness. On IHDP (clean), XGB-MSE RX wins (√PEHE 0.562 vs 1.795).
# If you expect well-behaved outcomes, pass nuisance_method="xgboost".
model = TargetedBayesianXLearner(robust=True)

model.fit(X, Y, W)

X_new = np.random.normal(0, 1, size=(5, P))
mean_cate, ci_lower, ci_upper = model.predict(X_new)

print("Mean CATE:", mean_cate)
print("95% CI Lower:", ci_lower)
print("95% CI Upper:", ci_upper)
```

To use the prior XGBoost defaults for backwards compatibility, pass
`nuisance_method="xgboost"` with explicit `outcome_model_params` /
`propensity_model_params`.

## Repository layout

```
sert_xlearner/        Library source (Phase 1/2/3 pipeline)
benchmarks/           Reproduction scripts (round-1 … round-13)
benchmarks/data/      Public benchmark datasets (IHDP, Hillstrom, Lalonde)
benchmarks/results/   CSV outputs and result-summary markdown
tests/                Pytest test suite (regression + stress probes)
paper/                LaTeX source for the arXiv preprint
docs/                 Internal design and development notes
baselines/            Vendored third-party baselines (see each subdir's LICENSE)
```

For deeper architecture / implementation notes see [docs/](docs/).

## Citation

```bibtex
@article{uehara2026bayesianxlearner,
  title   = {Bayesian X-Learner: Calibrated Posterior Inference for
             Heterogeneous Treatment Effects under Heavy-Tailed Outcomes},
  author  = {Uehara, Eichi},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

Vendored baselines under [baselines/](baselines/) ship under their
own licenses (e.g. [baselines/ebal_py/LICENSE](baselines/ebal_py/LICENSE)).

