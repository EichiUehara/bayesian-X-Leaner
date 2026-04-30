# Nuisance tree-depth on whale — §9 mechanism check

N fixed at 2000. XGBoost `max_depth` swept over [2, 4, 6, 8, 10] for both outcome and propensity nuisance models. All other settings fixed at RX-Learner (robust), `n_splits=2`, `c_whale=1.34`. Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

Prediction: deeper trees (more leaves → fewer whales per leaf) should reduce RMSE on whale, directly verifying the §9 mechanism hypothesis. Standard DGP should be flat-to-worse with depth.

## DGP: `standard` (true ATE = 2.00)

| max_depth | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | -0.026 | 0.031 | 1.00 | 0.153 |
| 4 | 8 | -0.019 | 0.027 | 1.00 | 0.149 |
| 6 | 8 | +0.008 | 0.036 | 1.00 | 0.150 |
| 8 | 8 | +0.013 | 0.038 | 1.00 | 0.149 |
| 10 | 8 | +0.005 | 0.031 | 1.00 | 0.147 |

## DGP: `whale` (true ATE = 2.00)

| max_depth | n | Bias | RMSE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | +9.117 | 9.739 | 0.00 | 0.099 |
| 4 | 8 | +3.380 | 3.756 | 0.00 | 0.145 |
| 6 | 8 | +0.789 | 1.093 | 0.00 | 0.140 |
| 8 | 8 | +0.017 | 0.065 | 0.75 | 0.140 |
| 10 | 8 | -0.001 | 0.041 | 0.88 | 0.144 |
