# MAD rescaling × prior-scale — revisiting §14

N fixed at 1000. Whale density ∈ ['5%', '10%', '20%'], `mad_rescale` ∈ ['on', 'off'], `prior_scale` ∈ [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]. Robust (Welsch) variant only, base `c_whale=1.34`. Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

Motivation. The initial §14 conclusion (*prior is not load-bearing*) was produced with the production-default MAD rescaling of `c_whale` (lines 85-92 of [targeted_bayesian_xlearner.py](../../sert_xlearner/targeted_bayesian_xlearner.py)). When whales inflate pseudo-outcome spread, MAD rises to ~1100, so the effective Welsch constant becomes ~1500 — Welsch no longer clips whales and the pseudo-likelihood is itself peaked at the biased value, hiding any effect of the prior. Turning MAD rescaling off restores Welsch's clipping behaviour and lets the prior act.

## density=5%, MAD rescale **on** (effective c ≈ 585.24)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8 | -2.868 | 2.950 | -0.691 | 0.00 | 0.213 |
| 0.1 | 8 | -21.819 | 25.980 | -20.786 | 0.00 | 0.186 |
| 0.5 | 8 | -26.549 | 31.808 | -25.886 | 0.00 | 0.409 |
| 1.0 | 8 | -26.663 | 31.994 | -26.103 | 0.00 | 0.379 |
| 2.0 | 8 | -26.748 | 32.045 | -26.099 | 0.00 | 0.412 |
| 10.0 | 8 | -26.887 | 32.096 | -26.152 | 0.00 | 0.249 |

## density=5%, MAD rescale **off** (effective c ≈ 1.34)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8 | -2.000 | 2.000 | -0.000 | 0.00 | 0.040 |
| 0.1 | 8 | -1.978 | 1.978 | +0.027 | 0.00 | 0.403 |
| 0.5 | 8 | -1.571 | 1.620 | +0.487 | 0.00 | 1.731 |
| 1.0 | 8 | -1.124 | 1.316 | +1.039 | 0.75 | 2.645 |
| 2.0 | 8 | -0.261 | 0.776 | +1.774 | 0.88 | 4.114 |
| 10.0 | 8 | +0.841 | 1.507 | +2.656 | 0.75 | 4.606 |

## density=10%, MAD rescale **on** (effective c ≈ 1578.45)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8 | -11.632 | 11.855 | -9.653 | 0.00 | 0.694 |
| 0.1 | 8 | -220.919 | 225.255 | -220.357 | 0.00 | 0.476 |
| 0.5 | 8 | -277.261 | 282.893 | -277.004 | 0.00 | 0.416 |
| 1.0 | 8 | -279.446 | 285.119 | -279.297 | 0.00 | 0.316 |
| 2.0 | 8 | -279.942 | 285.625 | -279.813 | 0.00 | 0.213 |
| 10.0 | 8 | -280.094 | 285.775 | -279.979 | 0.00 | 0.166 |

## density=10%, MAD rescale **off** (effective c ≈ 1.34)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8 | -2.001 | 2.001 | -0.001 | 0.00 | 0.040 |
| 0.1 | 8 | -2.010 | 2.010 | -0.007 | 0.00 | 0.401 |
| 0.5 | 8 | -2.057 | 2.069 | -0.086 | 0.00 | 2.130 |
| 1.0 | 8 | -1.713 | 1.818 | +0.082 | 0.75 | 4.325 |
| 2.0 | 8 | -0.276 | 1.508 | +2.039 | 1.00 | 7.165 |
| 10.0 | 8 | +4.608 | 5.588 | +6.458 | 0.62 | 14.601 |

## density=20%, MAD rescale **on** (effective c ≈ 3670.76)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8 | -50.629 | 50.940 | -48.602 | 0.00 | 0.624 |
| 0.1 | 8 | -1185.263 | 1192.681 | -1202.501 | 0.00 | 0.735 |
| 0.5 | 8 | -1516.201 | 1524.713 | -1539.185 | 0.00 | 2.789 |
| 1.0 | 8 | -1529.973 | 1538.600 | -1554.884 | 0.00 | 1.894 |
| 2.0 | 8 | -1533.557 | 1542.185 | -1557.631 | 0.00 | 1.170 |
| 10.0 | 8 | -1534.418 | 1543.047 | -1557.936 | 0.00 | 1.189 |

## density=20%, MAD rescale **off** (effective c ≈ 1.34)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 8 | -2.001 | 2.001 | -0.001 | 0.00 | 0.040 |
| 0.1 | 8 | -2.006 | 2.006 | -0.005 | 0.00 | 0.404 |
| 0.5 | 8 | -2.005 | 2.012 | +0.061 | 0.00 | 2.028 |
| 1.0 | 8 | -1.948 | 1.991 | +0.121 | 0.38 | 3.802 |
| 2.0 | 8 | -1.871 | 1.977 | +0.184 | 1.00 | 7.254 |
| 10.0 | 8 | +6.492 | 7.830 | +7.973 | 1.00 | 39.645 |
