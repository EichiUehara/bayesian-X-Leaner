# `c_whale` sensitivity

Seeds: [0, 1, 2, 3, 4, 5, 6, 7]. Welsch loss tuning constant `c_whale` swept over [0.5, 1.0, 1.34, 2.0, 5.0, 20.0]. All other settings fixed at RX-Learner (robust).

Prediction: RMSE is U-shaped on `whale` (c too small ignores good residuals; c too large ≈ L²) and flat on `standard` (no outliers).

## DGP: `standard` (true ATE = 2.00)

| c_whale | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 8 | +0.041 | 0.061 | 1.00 | 0.623 |
| 1.00 | 8 | +0.040 | 0.066 | 1.00 | 0.321 |
| 1.34 | 8 | +0.045 | 0.070 | 0.88 | 0.269 |
| 2.00 | 8 | +0.049 | 0.076 | 0.88 | 0.230 |
| 5.00 | 8 | +0.052 | 0.085 | 0.62 | 0.193 |
| 20.00 | 8 | +0.014 | 0.093 | 0.62 | 0.176 |

## DGP: `whale` (true ATE = 2.00)

| c_whale | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 8 | +0.033 | 0.066 | 1.00 | 0.368 |
| 1.00 | 8 | +0.052 | 0.078 | 1.00 | 0.272 |
| 1.34 | 8 | +0.058 | 0.077 | 1.00 | 0.258 |
| 2.00 | 8 | +0.078 | 0.106 | 0.75 | 0.237 |
| 5.00 | 8 | +0.141 | 0.199 | 0.38 | 0.214 |
| 20.00 | 8 | +0.106 | 0.460 | 0.12 | 0.187 |
