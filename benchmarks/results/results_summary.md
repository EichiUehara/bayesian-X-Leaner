# Pipeline Comparison — Monte Carlo Results

Seeds per DGP: **15**   |   Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`

Metrics are aggregated over seeds:

- **Mean ATE** — average point estimate across seeds
- **Bias** — Mean ATE − true ATE (positive = over-estimate)
- **RMSE** — √E[(ATE − τ)²], lower is better
- **Coverage** — fraction of seeds where 95 % CI contains τ (target ≈ 0.95)
- **Mean CI Width** — efficiency (narrower is better, if coverage ≥ 0.90)
- **Runtime (s)** — mean wall-clock seconds per fit
- **Success** — `n_ok/n_total` (failure = exception in wrapper)

## DGP: `standard`   (true ATE = 2.0)

| Estimator | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| CausalML X-Learner | +2.015 | +0.015 | 0.056 | — | — | 0.54 | 15/15 |
| X-Learner (std) | +2.019 | +0.019 | 0.059 | — | — | 0.50 | 15/15 |
| S-Learner | +2.025 | +0.025 | 0.073 | — | — | 0.13 | 15/15 |
| RX-Learner (robust) | +2.076 | +0.076 | 0.095 | 0.87 | 0.269 | 2.61 | 15/15 |
| T-Learner | +2.084 | +0.084 | 0.101 | — | — | 0.21 | 15/15 |
| EconML Forest | +1.918 | -0.082 | 0.111 | 1.00 | 0.486 | 1.20 | 15/15 |
| R-Learner | +1.904 | -0.096 | 0.115 | — | — | 0.38 | 15/15 |
| DoubleML IRM | +1.462 | -0.538 | 0.684 | 0.87 | 1.839 | 0.73 | 15/15 |
| RX-Learner (std) | +1.577 | -0.423 | 0.713 | 1.00 | 2.893 | 3.18 | 15/15 |
| DR-Learner (AIPW) | +1.382 | -0.618 | 0.794 | 1.00 | 2.084 | 0.52 | 15/15 |

## DGP: `whale`   (true ATE = 2.0)

| Estimator | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | +2.059 | +0.059 | 0.081 | 0.93 | 0.260 | 5.01 | 15/15 |
| RX-Learner (std) | +7.213 | +5.213 | 20.790 | 0.60 | 38.318 | 6.90 | 15/15 |
| S-Learner | -60.449 | -62.449 | 66.382 | — | — | 0.24 | 15/15 |
| EconML Forest | -66.985 | -68.985 | 75.418 | 1.00 | 407.785 | 2.06 | 15/15 |
| R-Learner | -90.530 | -92.530 | 103.436 | — | — | 0.86 | 15/15 |
| T-Learner | -108.824 | -110.824 | 114.560 | — | — | 0.39 | 15/15 |
| CausalML X-Learner | -112.252 | -114.252 | 119.054 | — | — | 1.09 | 15/15 |
| X-Learner (std) | -112.854 | -114.854 | 119.839 | — | — | 1.13 | 15/15 |
| DoubleML IRM | -6.643 | -8.643 | 168.817 | 0.87 | 664.610 | 1.19 | 15/15 |
| DR-Learner (AIPW) | -129.545 | -131.545 | 322.502 | 0.93 | 986.949 | 0.96 | 15/15 |

## DGP: `imbalance`   (true ATE = 2.0)

| Estimator | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| CausalML X-Learner | +1.989 | -0.011 | 0.084 | — | — | 1.05 | 15/15 |
| X-Learner (std) | +1.990 | -0.010 | 0.084 | — | — | 1.12 | 15/15 |
| EconML Forest | +1.926 | -0.074 | 0.115 | 1.00 | 0.959 | 2.55 | 15/15 |
| T-Learner | +1.965 | -0.035 | 0.116 | — | — | 0.43 | 15/15 |
| RX-Learner (robust) | +2.064 | +0.064 | 0.121 | 0.67 | 0.217 | 4.96 | 15/15 |
| R-Learner | +1.923 | -0.077 | 0.136 | — | — | 1.01 | 15/15 |
| DoubleML IRM | +1.939 | -0.061 | 0.193 | 1.00 | 2.897 | 1.15 | 15/15 |
| S-Learner | +1.837 | -0.163 | 0.198 | — | — | 0.33 | 15/15 |
| DR-Learner (AIPW) | +2.021 | +0.021 | 0.322 | 1.00 | 2.791 | 1.20 | 15/15 |
| RX-Learner (std) | +1.614 | -0.386 | 0.678 | 0.80 | 1.846 | 5.99 | 15/15 |

## DGP: `sharp_null`   (true ATE = 0.0)

| Estimator | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| S-Learner | +0.003 | +0.003 | 0.010 | — | — | 0.25 | 15/15 |
| CausalML X-Learner | +0.004 | +0.004 | 0.021 | — | — | 1.31 | 15/15 |
| X-Learner (std) | +0.004 | +0.004 | 0.021 | — | — | 1.15 | 15/15 |
| EconML Forest | +0.001 | +0.001 | 0.024 | 1.00 | 0.344 | 2.57 | 15/15 |
| R-Learner | +0.001 | +0.001 | 0.026 | — | — | 0.99 | 15/15 |
| T-Learner | +0.013 | +0.013 | 0.027 | — | — | 0.50 | 15/15 |
| RX-Learner (robust) | +0.037 | +0.037 | 0.043 | 1.00 | 0.174 | 5.15 | 15/15 |
| DoubleML IRM | -0.028 | -0.028 | 0.100 | 1.00 | 0.435 | 1.51 | 15/15 |
| DR-Learner (AIPW) | -0.028 | -0.028 | 0.129 | 0.93 | 0.499 | 1.16 | 15/15 |
| RX-Learner (std) | -0.143 | -0.143 | 0.291 | 0.93 | 1.174 | 5.57 | 15/15 |
