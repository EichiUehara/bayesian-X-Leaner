# RX-Learner Stability & Reproducibility Report

Total MCMC fits: **48**
Fits with R̂ < 1.05:   **48/48**  (100 %)
Fits with ESS > 200:  **48/48**  (100 %)

## Per-condition summary

| Condition | n | Mean ATE | Std ATE | Bias | RMSE | Worst R̂ | Mean R̂ | Worst ESS | Mean ESS | R̂-pass | ESS-pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCMC noise · standard · std | 6 | +1.1238 | 0.4104 | -0.8762 | 0.9530 | 1.005 | 1.001 | 984 | 1188 | 6/6 | 6/6 |
| Data variance · standard · std | 6 | +1.9361 | 0.4933 | -0.0639 | 0.4548 | 1.002 | 1.001 | 1161 | 1257 | 6/6 | 6/6 |
| MCMC noise · standard · robust | 6 | +2.0499 | 0.0605 | +0.0499 | 0.0744 | 1.002 | 1.000 | 401 | 597 | 6/6 | 6/6 |
| Data variance · standard · robust | 6 | +2.0502 | 0.0537 | +0.0502 | 0.0701 | 1.005 | 1.001 | 522 | 590 | 6/6 | 6/6 |
| MCMC noise · whale · std | 6 | +1.2703 | 24.1649 | -0.7297 | 22.0715 | 1.012 | 1.004 | 404 | 568 | 6/6 | 6/6 |
| Data variance · whale · std | 6 | +6.0535 | 17.5728 | +4.0535 | 16.5459 | 1.007 | 1.003 | 326 | 482 | 6/6 | 6/6 |
| MCMC noise · whale · robust | 6 | +2.1338 | 0.1140 | +0.1338 | 0.1695 | 1.004 | 1.001 | 393 | 590 | 6/6 | 6/6 |
| Data variance · whale · robust | 6 | +2.0754 | 0.0344 | +0.0754 | 0.0816 | 1.001 | 1.000 | 626 | 679 | 6/6 | 6/6 |

## Interpretation

- **Std ATE across MCMC seeds** (with data fixed) quantifies *Monte-Carlo noise* in the posterior. It should be ≪ RMSE across data seeds — otherwise, the reported point estimate is not reproducible and `num_samples` must be increased.
- **R̂ ≥ 1.05 or ESS < 200** in any fit signals under-converged MCMC; increase `num_warmup` / `num_samples`.
- **Bias close to 0 under whale DGP** confirms the Welsch redescending loss (`robust=True`) neutralises the outlier. Note: when `robust=True`, `use_student_t` is ignored by the inference backend — see EXTENSIONS.md §10.

Figures in `benchmarks/results/figures/mcmc_diagnostics.png`.