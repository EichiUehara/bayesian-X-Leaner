# Coverage comparison: RX-Learner MCMC posterior vs Huber-DR bootstrap CI

Whale DGP, N = 1000, true ATE = 2.0, 3 seeds. 95 % CI level.
Huber-DR bootstrap is on the full (nuisance + Huber) pipeline (200 replicates).

| density | estimator | n | mean ATE | bias | 95% coverage | CI width |
|---:|---|---:|---:|---:|---:|---:|
| 0.00 | Conformal-DR (split-CP) | 3 | +1.930 | -0.070 | 1.00 | 12.716 |
| 0.00 | Huber-DR (bootstrap CI) | 3 | +1.974 | -0.026 | 1.00 | 0.149 |
| 0.00 | RX-Gaussian (severity=none) | 3 | +1.529 | -0.471 | 1.00 | 1.462 |
| 0.00 | RX-StudentT (severity=none) | 3 | +2.026 | +0.026 | 1.00 | 0.127 |
| 0.00 | RX-Welsch (severity=none) | 3 | +2.021 | +0.021 | 1.00 | 0.215 |
| 0.00 | RX-Welsch (severity=severe) | 3 | +2.134 | +0.134 | 0.00 | 0.222 |
| 0.01 | Conformal-DR (split-CP) | 3 | -89.196 | -91.196 | 1.00 | 2090.305 |
| 0.01 | Huber-DR (bootstrap CI) | 3 | -6.187 | -8.187 | 1.00 | 22.008 |
| 0.01 | RX-Gaussian (severity=none) | 3 | -8.994 | -10.994 | 0.67 | 39.678 |
| 0.01 | RX-StudentT (severity=none) | 3 | +2.359 | +0.359 | 0.67 | 0.635 |
| 0.01 | RX-Welsch (severity=none) | 3 | +1.919 | -0.081 | 1.00 | 0.400 |
| 0.01 | RX-Welsch (severity=severe) | 3 | +2.128 | +0.128 | 0.00 | 0.219 |
| 0.05 | Conformal-DR (split-CP) | 3 | -409.941 | -411.941 | 1.00 | 12368.544 |
| 0.05 | Huber-DR (bootstrap CI) | 3 | -111.393 | -113.393 | 0.00 | 181.074 |
| 0.05 | RX-Gaussian (severity=none) | 3 | -3.672 | -5.672 | 0.67 | 38.339 |
| 0.05 | RX-StudentT (severity=none) | 3 | +7.767 | +5.767 | 0.00 | 8.930 |
| 0.05 | RX-Welsch (severity=none) | 3 | +5.313 | +3.313 | 0.00 | 1.558 |
| 0.05 | RX-Welsch (severity=severe) | 3 | +2.130 | +0.130 | 0.00 | 0.228 |
| 0.20 | Conformal-DR (split-CP) | 3 | -2269.353 | -2271.353 | 1.00 | 25886.042 |
| 0.20 | Huber-DR (bootstrap CI) | 3 | -1760.488 | -1762.488 | 0.00 | 672.708 |
| 0.20 | RX-Gaussian (severity=none) | 3 | -27.444 | -29.444 | 0.00 | 39.632 |
| 0.20 | RX-StudentT (severity=none) | 3 | +26.769 | +24.769 | 0.00 | 16.141 |
| 0.20 | RX-Welsch (severity=none) | 3 | +8.487 | +6.487 | 1.00 | 32.288 |
| 0.20 | RX-Welsch (severity=severe) | 3 | +1.985 | -0.015 | 1.00 | 0.292 |