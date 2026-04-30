# `n_splits` sensitivity — tests the Sert-framework claim

N fixed at 2000. Cross-fitting folds `n_splits` swept over [2, 3, 5, 10]. All other settings fixed at RX-Learner (robust). Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

Prediction: on whale, higher `n_splits` reduces bias (smaller training folds → more whale concentration per fold relative to leaf capacity → tighter whale isolation). Plateau expected by `n_splits=5`. Standard DGP should be flat.

## DGP: `standard` (true ATE = 2.00)

| n_splits | n | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | -0.019 | 0.027 | 1.00 | 0.149 | 4.54 |
| 3 | 8 | -0.009 | 0.030 | 1.00 | 0.151 | 3.93 |
| 5 | 8 | -0.023 | 0.036 | 1.00 | 0.151 | 4.55 |
| 10 | 8 | -0.010 | 0.023 | 1.00 | 0.151 | 6.34 |

## DGP: `whale` (true ATE = 2.00)

| n_splits | n | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | +3.380 | 3.756 | 0.00 | 0.145 | 3.40 |
| 3 | 8 | +3.587 | 3.756 | 0.00 | 0.144 | 3.87 |
| 5 | 8 | +5.240 | 5.560 | 0.00 | 0.127 | 6.90 |
| 10 | 8 | +6.340 | 6.665 | 0.00 | 0.138 | 9.22 |
