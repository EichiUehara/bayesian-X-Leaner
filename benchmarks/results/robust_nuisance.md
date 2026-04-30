# Robust nuisance subsumption test

N fixed at 1000. Whale density ∈ ['1%', '5%', '10%', '20%']. Nuisance outcome learner ∈ ['xgb_mse', 'xgb_huber', 'catboost_mse', 'catboost_huber']. Production code path elsewhere (robust=True, MAD rescale on, `prior_scale=10.0`, `n_splits=2`, depth-4). Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

Question: if the nuisance outcome learner is natively robust (Huber loss), does the downstream Welsch/MAD/prior tangle (§14) become unnecessary?

## density = 1% (8 seeds × 4 configs)

| nuisance | Bias | RMSE | Median ATE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| catboost_huber | +0.112 | 0.114 | +2.108 | 0.62 | 0.220 | 3.2 |
| catboost_mse | +1.365 | 2.346 | +3.454 | 0.00 | 0.205 | 3.5 |
| xgb_huber | -0.060 | 0.084 | +1.943 | 0.75 | 0.208 | 2.7 |
| xgb_mse | +0.103 | 0.230 | +2.086 | 0.38 | 0.195 | 3.7 |

## density = 5% (8 seeds × 4 configs)

| nuisance | Bias | RMSE | Median ATE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| catboost_huber | +0.104 | 0.106 | +2.099 | 0.62 | 0.218 | 3.7 |
| catboost_mse | -23.783 | 28.971 | -27.206 | 0.00 | 0.331 | 9.8 |
| xgb_huber | -0.152 | 0.177 | +1.866 | 0.38 | 0.209 | 2.7 |
| xgb_mse | -26.887 | 32.096 | -26.152 | 0.00 | 0.249 | 8.9 |

## density = 10% (8 seeds × 4 configs)

| nuisance | Bias | RMSE | Median ATE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| catboost_huber | +0.076 | 0.083 | +2.077 | 0.88 | 0.214 | 3.3 |
| catboost_mse | -245.548 | 248.285 | -227.680 | 0.00 | 0.962 | 9.4 |
| xgb_huber | -0.556 | 0.636 | +1.540 | 0.00 | 0.207 | 2.9 |
| xgb_mse | -280.094 | 285.775 | -279.979 | 0.00 | 0.166 | 8.9 |

## density = 20% (8 seeds × 4 configs)

| nuisance | Bias | RMSE | Median ATE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| catboost_huber | -0.186 | 0.203 | +1.819 | 0.12 | 0.201 | 1.9 |
| catboost_mse | -1320.440 | 1325.438 | -1333.607 | 0.00 | 1.475 | 4.4 |
| xgb_huber | -12.543 | 13.101 | -10.360 | 0.00 | 0.205 | 1.6 |
| xgb_mse | -1534.418 | 1543.047 | -1557.936 | 0.00 | 1.189 | 6.0 |
