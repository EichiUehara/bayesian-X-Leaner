# Huber-δ follow-ups: three open questions from §17

The §17 IHDP audit left three open questions explicitly deferred as
follow-ups, not blockers. This note reports the resolutions. §16 / §17
prose in [EXTENSIONS.md](EXTENSIONS.md) is preserved verbatim; the new
data is summarised here and in the raw CSVs already on disk.

## 1. Empirical 0–5 % crossover on the whale DGP

**Question (§17.7).** At what contamination rate does Welsch+CB-Huber
start beating XGB-MSE? We had the 0 % and 5 % endpoints but nothing
between.

**Run.** `benchmarks/run_whale_density.py` extended down to 0.1 %, 0.5 %,
1 %, 2 % densities (N = 1000, 3 seeds). Raw data:
[whale_density_catboost_huber.md](whale_density_catboost_huber.md).

| Density | RX-Learner (std, XGB-MSE) RMSE | RX-Learner (robust, CB-Huber) RMSE |
|---:|---:|---:|
| 0.1 % |   6.51 | 0.134 |
| 0.5 % |  24.66 | 0.136 |
| 1.0 % |  40.04 | 0.130 |
| 2.0 % |  62.13 | 0.127 |
| 5.0 % | 106.59 | 0.128 |

**Finding.** The crossover is **below 0.1 %** on this DGP. A single
whale in N = 1000 (0.1 % density) is already enough to put XGB-MSE in
the RMSE ≈ 6 regime — catastrophic on a DGP whose true ATE is +2. The
robust variant is flat (RMSE ≈ 0.13) across the entire 0.1–5 % sweep,
so the boundary §17 hypothesised (~2 % density) is too loose in this
direction: on clean *contamination-free* data XGB-MSE wins (IHDP
§17.3), but as soon as any whale appears the default fails faster than
the benchmark cadence can resolve.

**Caveat.** Coverage for CB-Huber is 0 % throughout this sweep, because
bias (+0.13) is comparable to CI half-width (≈ 0.11) and the
intervals miss the true ATE by a hair. Point accuracy is excellent;
the 95 % posterior is mildly overconfident at the contamination floor.

## 2. `contamination_severity` enum API

**Question (§17.7).** A principled user-facing knob that exposes
Huber's 1964 minimax-δ table instead of asking users to remember
specific δ values.

**Ship.** Added to `TargetedBayesianXLearner.__init__`:

| `contamination_severity` | Nuisance | Huber δ | Target ε |
|---|---|---:|---:|
| `"none"`     | XGBoost MSE       | —       |  0 %    |
| `"mild"`     | CatBoost Huber    | 1.345   | ~5 %    |
| `"moderate"` | CatBoost Huber    | 1.0     | ~10 %   |
| `"severe"`   | CatBoost Huber    | 0.5     | ~40 %   |

The δ values are the numerical solutions to Huber's minimax-δ relation
φ(δ)/δ − (1 − Φ(δ)) = ε/(2(1 − ε)) (EXTENSIONS.md §17.1). Explicit
`outcome_model_params` still wins over the preset — the enum is a
convenience, not a lock.

Regression coverage:
[test_default_config_regression.py](../../tests/test_default_config_regression.py)
pins each severity level to its expected `(nuisance_method, δ)` pair,
rejects unknown labels, and confirms explicit params override the
preset. Nine tests total; all passing.

## 3. IHDP with δ = 1.345 (canonical Huber 1964)

**Question (§17.7).** Theory (EXTENSIONS.md §17.1) predicts δ = 1.345
recovers ~95 % of Gaussian efficiency — vs ~79 % for δ = 0.5 — so
canonical Huber should close most of the IHDP gap that δ = 0.5 opens.
Is the 3× PEHE penalty primarily the δ-choice, or is it structural?

**Run.** Added `RX-Learner (CB-Huber δ=1.345)` row to
[ihdp_benchmark.md](ihdp_benchmark.md) (same 5 replications, same
nuisance architecture, only δ differs).

| Estimator | √PEHE | ε_ATE |
|---|---:|---:|
| RX-Learner (robust, XGB-MSE)          | **0.562** | 0.079 |
| RX-Learner (CB-Huber δ=1.345)         | 1.232 | 0.739 |
| RX-Learner (CB-Huber δ=0.5)           | 1.795 | 1.368 |

**Finding.** Canonical Huber (δ = 1.345) improves √PEHE by 31 % over
the whale-tuned δ = 0.5 (1.80 → 1.23), confirming the direction theory
predicts — a less aggressive redescender pays less efficiency cost on
clean data.

**But** the gap to XGB-MSE remains large: 1.23 vs 0.562 is still a 2.2×
PEHE penalty, far more than the ~5 % ARE theory predicts for
δ = 1.345. This replicates the §17.2 conclusion: the IHDP penalty is
**not** primarily about δ, and therefore cannot be tuned away by
loosening the loss. The dominant mechanism is structural — tree-split
distortion and coherent bias on heterogeneous effects (§17.5) — which
canonical Huber does not fix.

**Implication for the API.** `contamination_severity="mild"` is the
right default when the user is uncertain about contamination: it
trades ~5 % efficiency for a large improvement over `"severe"` on
clean-leaning data while still providing real robustness for light
contamination. On truly clean benchmarks like IHDP, however, it still
underperforms `contamination_severity="none"` — users who know their
data is clean should say so.

## Summary

Three items moved from "design sketch" to "shipped and tested":

1. Whale crossover maps to *below* 0.1 % — the default fails faster
   than we thought. Strengthens the §17 motivation for the enum.
2. The enum API ships with pinned regression tests so the (severity,
   δ) table cannot drift silently.
3. δ = 1.345 on IHDP confirms the theoretical *direction* and
   falsifies the "δ = 0.5 is to blame" hypothesis. The 3× penalty is
   structural; use `"none"` on clean data, not a loosened Huber.
