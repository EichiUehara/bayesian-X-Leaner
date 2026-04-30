# Benchmark results — reading order

Three summary documents cover six experiments. Start at the top and walk
down as deep as you need.

## Top-level reports

1. **[STABILITY_SUMMARY.md](STABILITY_SUMMARY.md)** — the headline
   verdict. MCMC convergence, 15-seed Monte Carlo across 4 DGPs, and the
   mechanism behind robust > standard. **Read this first.**

2. **[EXTENSIONS.md](EXTENSIONS.md)** — five follow-up experiments:
   overlap weights, CATE heterogeneity, Causal BART baseline, nonlinear CATE
   (basis sensitivity), and the IHDP semi-synthetic benchmark.

3. **[pseudo_outcome_diagnostics.md](pseudo_outcome_diagnostics.md)** —
   empirical verification (excess kurtosis, skew, influence curves) of the
   three claims about why the non-robust variant fails even on clean data.

## Per-experiment results

| File | DGP | Metric | Headline |
|---|---|---|---|
| [results_summary.md](results_summary.md) | 4 DGPs × 10 estimators | ATE RMSE, coverage, CI width | RX-Learner (robust) RMSE 0.08 on whale; every Gaussian-likelihood method ≥ 67 |
| [stability_report.md](stability_report.md) | standard + whale × 6 MCMC seeds × 6 data seeds | R̂, ESS, MCMC-vs-data-noise ratio | 100% of fits pass R̂ < 1.05 and ESS > 200 |
| [overlap_experiment.md](overlap_experiment.md) | imbalance | ATE + coverage | `use_overlap=True` lifts coverage 0.75 → 1.00 |
| [cate_benchmark.md](cate_benchmark.md) | heterogeneous_cate (linear τ) | PEHE | RX-Learner PEHE 0.083, 3× better than S-Learner |
| [nonlinear_cate.md](nonlinear_cate.md) | nonlinear_cate (sin τ) | PEHE × basis | Fourier basis wins (0.097); linear/polynomial misspecified (0.66/0.70) |
| [bart_comparison.md](bart_comparison.md) | standard + whale | ATE RMSE, coverage | BART fails on whale (RMSE 101, 0% coverage); RX-Learner (robust) 0.072 |
| [ihdp_benchmark.md](ihdp_benchmark.md) | IHDP (real covariates) | √PEHE, ε_ATE | T-Learner wins (1.37), RX-Learner (robust) 2nd (1.95), std fails (6.25) |
| [ihdp_basis_ablation.md](ihdp_basis_ablation.md) | IHDP × 4 bases | PEHE by basis | Interactions close 72% of gap to T-Learner (1.53 vs 1.37); Nyström RBF *worse* than linear |
| [component_ablation.md](component_ablation.md) | 3 DGPs × {Gaussian, Student-T, Welsch} × 15 seeds | RMSE, coverage | Student-T and Welsch match on RMSE; only Welsch keeps coverage calibrated on clean data (0.87 vs 0.47) |
| [c_whale_sensitivity.md](c_whale_sensitivity.md) | standard + whale × c_whale ∈ {0.5..20} × 8 seeds | RMSE, coverage | Huber default c=1.34 is the calibrated sweet spot; c ≥ 2 collapses whale coverage (1.00 → 0.12) |
| [sample_size_scaling.md](sample_size_scaling.md) | standard + whale × N ∈ {200..5000} × 8 seeds | log-log RMSE slope | √N consistency holds for both variants on standard; std diverges on whale (slope ≥ 0) |
| [whale_density.md](whale_density.md) | whale × density ∈ {0.5 %..20 %} × 8 seeds | RMSE, coverage | Practical contamination tolerance ≤ 1 %; robust catastrophically fails at ≥ 2 % |
| [nuisance_depth.md](nuisance_depth.md) | standard + whale × max_depth ∈ {2..10} × 8 seeds | RMSE, bias | Deeper trees monotonically cure whale bias (depth 10 RMSE 0.041 vs depth 4 RMSE 3.76); flat on clean DGP |
| [n_splits_sensitivity.md](n_splits_sensitivity.md) | standard + whale × n_splits ∈ {2, 3, 5, 10} × 8 seeds | RMSE, bias | More folds *worsen* whale bias; default `n_splits=2` is optimal |
| [prior_scale_sensitivity.md](prior_scale_sensitivity.md) | whale × density ∈ {5 %, 10 %, 20 %} × prior_scale ∈ {0.5..10} × 8 seeds | RMSE, bias, mode-flip rate | Initial mode-flip sweep — later shown (§14) to be contaminated by the MAD rescaling bug |
| [mad_rescaling_and_prior.md](mad_rescaling_and_prior.md) | whale × density × `mad_rescale` ∈ {on, off} × prior_scale ∈ {0.01..10} × 8 seeds | RMSE, bias, coverage, effective c | **MAD rescaling is the §11 culprit** — disabling it + `prior_scale=2.0` recovers coverage 1.00 at 20 % density (RMSE 1.98 vs 1543) |
| [robust_nuisance.md](robust_nuisance.md) | whale × density × {xgb, catboost} × {mse, huber} × 8 seeds | RMSE, bias, coverage | **CatBoost-Huber subsumes §12 and §14** — RMSE 0.20 at 20 % density with zero Bayesian-layer changes (7600× better than XGB-MSE baseline) |
| [nuisance_loss_sweep.md](nuisance_loss_sweep.md) | whale 20 % density × loss ∈ {Huber:0.1/0.5/1.0, Quantile:0.5, RMSE} × depth ∈ {4, 8} × 8 seeds | RMSE, bias, coverage | **Huber:δ=0.5, depth 4 closes §15's coverage gap** — bias −0.018, RMSE 0.058, coverage **1.00** at 20 % whale density (U-shaped δ response; δ=0.1 too tight, δ=1.0 too loose) |

Each `*.md` has a matching `*_raw.csv` (e.g.
[cate_benchmark_raw.csv](cate_benchmark_raw.csv)) with per-seed values —
use these to recompute statistics or rebuild plots.

## Figures

| File | Source |
|---|---|
| [figures/mcmc_diagnostics.png](figures/mcmc_diagnostics.png) | `stability_check` |
| [figures/bias_by_dgp.png](figures/bias_by_dgp.png) | `plot_results` |
| [figures/rmse_comparison.png](figures/rmse_comparison.png) | `plot_results` |
| [figures/coverage_vs_width.png](figures/coverage_vs_width.png) | `plot_results` |
| [figures/runtime_comparison.png](figures/runtime_comparison.png) | `plot_results` |
| [figures/pseudo_outcome_tails.png](figures/pseudo_outcome_tails.png) | `verify_pseudo_outcome` |
| [figures/loss_influence.png](figures/loss_influence.png) | `verify_pseudo_outcome` |

## Reproducing

See [../../REPRODUCE.md](../../REPRODUCE.md) for the full recipe.
