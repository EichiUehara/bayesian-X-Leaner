# Whale-density breakdown boundary

N fixed at 1000. Whale density swept over ['0.1%', '0.5%', '1%', '2%', '5%'] ([1, 5, 10, 20, 50] whales). Seeds: [0, 1, 2].

Prediction: robust variant holds through density ~2-5 % and fails above, set by when leaf-level whale concentration exceeds what XGBoost can isolate in single leaves. `std` variant fails at every density.

## RX-Learner (robust)

| density | n_whales | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 1 | 3 | -0.003 | 0.038 | 1.00 | 0.209 |
| 0.5% | 5 | 3 | -0.018 | 0.052 | 1.00 | 0.209 |
| 1.0% | 10 | 3 | -0.030 | 0.138 | 0.33 | 0.200 |
| 2.0% | 20 | 3 | +4.013 | 5.800 | 0.00 | 0.197 |
| 5.0% | 50 | 3 | -37.270 | 42.827 | 0.00 | 0.095 |

## RX-Learner (std)

| density | n_whales | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 1 | 3 | -7.108 | 7.892 | 1.00 | 25.393 |
| 0.5% | 5 | 3 | +6.032 | 18.525 | 0.33 | 34.681 |
| 1.0% | 10 | 3 | +15.504 | 19.333 | 0.33 | 36.791 |
| 2.0% | 20 | 3 | +0.535 | 26.597 | 0.33 | 37.884 |
| 5.0% | 50 | 3 | +7.685 | 9.781 | 1.00 | 37.984 |
