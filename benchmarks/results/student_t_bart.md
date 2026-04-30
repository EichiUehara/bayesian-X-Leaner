# Student-t-likelihood T-BART on IHDP

Heavy-tailed Bayesian tree backbone: T-BART with Student-t residuals
(ν learned via prior Gamma(2, 0.1)). 5 reps, m=200, 1000 total draws.

| rep | √PEHE | ε_ATE | runtime |
|---:|---:|---:|---:|
| 1 | 0.515 | 0.348 | 358.0s |
| 2 | 0.411 | 0.016 | 289.0s |
| 3 | 0.706 | 0.588 | 281.6s |
| 4 | 0.782 | 0.273 | 283.1s |
| 5 | 1.000 | 0.162 | 294.5s |

**Mean √PEHE = 0.683 ± 0.231**, mean ε_ATE = 0.278.