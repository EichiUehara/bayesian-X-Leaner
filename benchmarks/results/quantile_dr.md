# Quantile DR-learner — paradigm (iii) `tail is the target' alternative

Whale DGP, N = 1000, true mean ATE = 2.0, 3 seeds.
QuantileRegressor on DR pseudo-outcomes (intercept-only basis).

Estimand: q-th quantile of the treatment-effect distribution.
Note: under whale contamination the q-th quantile of the DR pseudo-outcomes
drifts with density; its meaning is *not* the contaminated mean.

| density | quantile | n | mean estimate | std |
|---:|---:|---:|---:|---:|
| 0.00 | 0.50 | 3 | +1.980 | 0.042 |
| 0.00 | 0.75 | 3 | +2.424 | 0.057 |
| 0.00 | 0.95 | 3 | +3.164 | 0.090 |
| 0.01 | 0.50 | 3 | +0.489 | 4.068 |
| 0.01 | 0.75 | 3 | +46.265 | 8.694 |
| 0.01 | 0.95 | 3 | +194.754 | 10.306 |
| 0.05 | 0.50 | 3 | -13.754 | 8.428 |
| 0.05 | 0.75 | 3 | +98.272 | 9.562 |
| 0.05 | 0.95 | 3 | +468.471 | 61.352 |
| 0.20 | 0.50 | 3 | -1772.306 | 107.172 |
| 0.20 | 0.75 | 3 | +72.128 | 17.987 |
| 0.20 | 0.95 | 3 | +643.176 | 16.401 |