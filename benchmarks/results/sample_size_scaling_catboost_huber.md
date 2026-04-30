# Sample-size scaling

Seeds: [0, 1, 2]. N ∈ [200, 500, 1000, 2000, 5000].

Classical Bayesian consistency predicts RMSE ∝ 1/√N, i.e. a slope of **−0.5** in log-log. A slope near 0 means non-convergence.

## DGP: `standard`

| Variant | N=200 | N=500 | N=1000 | N=2000 | N=5000 | log-log slope |
|---|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 0.401 | 0.201 | 0.132 | 0.064 | 0.080 | -0.55 |
| RX-Learner (std) | 0.351 | 0.071 | 0.036 | 0.037 | 0.014 | -0.91 |

## DGP: `whale`

| Variant | N=200 | N=500 | N=1000 | N=2000 | N=5000 | log-log slope |
|---|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 0.430 | 0.195 | 0.130 | 0.060 | 0.075 | -0.59 |
| RX-Learner (std) | 22.775 | 33.237 | 40.037 | 48.843 | 62.572 | +0.31 |

## Interpretation

- **Slope near −0.5** → √N-consistent (canonical Bayesian rate).
- **Slope near 0 or positive** → non-convergent; extra data does not help (expected for RX-Learner std on whale: 1 % whale density is preserved as N grows).

## §17 finding — §9's √N breakdown is a nuisance-contamination artefact

Under the legacy XGB-MSE nuisance default, §9
([sample_size_scaling.md](sample_size_scaling.md)) reported
slope **+1.29** on whale — RMSE *grew* with N, flagging a fundamental
consistency break. That finding was load-bearing in the README's
caveats list.

Under the §16 CatBoost + Huber(δ=0.5) default run here, the robust
variant's whale slope is **−0.59** — indistinguishable from classical
√N consistency and matching the clean-DGP slope of −0.55. The absolute
RMSE at N=2000 drops from order-100 (XGB-MSE) to **0.06** (CatBoost-
Huber).

This confirms §14's diagnosis + §15-16's fix:  §9's breakdown was
**nuisance leakage inflating the DR pseudo-outcome variance faster
than MCMC could absorb it**, not an architectural limit of the
estimator. Once µ̂₀ stays clean (Huber-absorbed whales at the leaf
level), the pseudo-outcome scale shrinks with N at the classical rate
and the Bayesian layer inherits that.

The non-robust ("std") variant still fails on whale (slope +0.31),
confirming the robust loss remains necessary even with clean nuisance
— there is still a small amount of whale leakage post-Huber that the
Welsch layer removes.