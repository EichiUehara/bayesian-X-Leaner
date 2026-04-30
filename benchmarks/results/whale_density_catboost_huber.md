# Whale-density breakdown boundary

N fixed at 1000. Whale density swept over ['0.1%', '0.5%', '1%', '2%', '5%'] ([1, 5, 10, 20, 50] whales). Seeds: [0, 1, 2].

Prediction: robust variant holds through density ~2-5 % and fails above, set by when leaf-level whale concentration exceeds what XGBoost can isolate in single leaves. `std` variant fails at every density.

## RX-Learner (robust)

| density | n_whales | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 1 | 3 | +0.134 | 0.134 | 0.00 | 0.216 |
| 0.5% | 5 | 3 | +0.136 | 0.136 | 0.00 | 0.215 |
| 1.0% | 10 | 3 | +0.130 | 0.130 | 0.00 | 0.221 |
| 2.0% | 20 | 3 | +0.127 | 0.127 | 0.00 | 0.218 |
| 5.0% | 50 | 3 | +0.128 | 0.128 | 0.00 | 0.216 |

## RX-Learner (std)

| density | n_whales | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 1 | 3 | -6.496 | 6.507 | 1.00 | 19.029 |
| 0.5% | 5 | 3 | -24.528 | 24.656 | 0.00 | 29.313 |
| 1.0% | 10 | 3 | -39.996 | 40.037 | 0.00 | 31.322 |
| 2.0% | 20 | 3 | -62.121 | 62.131 | 0.00 | 33.047 |
| 5.0% | 50 | 3 | -106.577 | 106.588 | 0.00 | 35.652 |
