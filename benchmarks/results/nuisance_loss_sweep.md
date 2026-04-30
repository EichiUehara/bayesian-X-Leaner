# Nuisance-loss sweep — eliminating residual contamination at 20 % whale density

N=1000, whale density=20 %, nuisance=catboost, robust=True, MAD rescale on, `prior_scale=10.0`, `n_splits=2`. Outcome `loss_function` ∈ ['Huber:delta=0.1', 'Huber:delta=0.5', 'Huber:delta=1.0', 'Quantile:alpha=0.5', 'RMSE'], `depth` ∈ [4, 8]. Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

Context: §15's CatBoost-Huber(δ=1) config delivered RMSE 0.20 at this density but coverage 0.12 — a systematic bias of ~0.19 comes from μ̂₀ contamination the Huber L¹ tail can't fully remove when 40 % of the control training set is whales. This sweep tests whether tighter-δ Huber, median regression (Quantile:0.5), or deeper trees cures it.

| loss | depth | Bias | RMSE | Median ATE | Coverage | CI Width |
|---|---:|---:|---:|---:|---:|---:|
| Huber:delta=0.1 | 4 | +0.230 | 0.236 | +2.235 | 0.00 | 0.199 |
| Huber:delta=0.1 | 8 | +0.627 | 0.629 | +2.603 | 0.00 | 0.199 |
| Huber:delta=0.5 | 4 | -0.018 | 0.058 | +1.994 | 1.00 | 0.200 |
| Huber:delta=0.5 | 8 | +0.239 | 0.241 | +2.241 | 0.00 | 0.197 |
| Huber:delta=1.0 | 4 | -0.186 | 0.203 | +1.819 | 0.12 | 0.201 |
| Huber:delta=1.0 | 8 | +0.084 | 0.101 | +2.094 | 0.50 | 0.199 |
| Quantile:alpha=0.5 | 4 | -466.516 | 474.138 | -489.776 | 0.00 | 0.151 |
| Quantile:alpha=0.5 | 8 | -797.017 | 801.560 | -800.209 | 0.00 | 0.158 |
| RMSE | 4 | -1320.440 | 1325.438 | -1333.607 | 0.00 | 1.475 |
| RMSE | 8 | -1314.504 | 1319.962 | -1350.604 | 0.00 | 1.956 |
