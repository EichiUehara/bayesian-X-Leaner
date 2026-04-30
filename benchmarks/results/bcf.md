# BCF (Hahn 2020) on IHDP — proper Bayesian Causal Forest baseline

5 replications, BCF prognostic mu(x, pi) BART (m=200) + tau(x) BART (m=50),
1000 total draws across 2 chains, cores=1. Pi-hat from LogisticRegression.

| rep | √PEHE | ε_ATE | runtime |
|---:|---:|---:|---:|
| 1 | 0.604 | 0.023 | 150.8s |
| 2 | 0.605 | 0.089 | 142.4s |
| 3 | 0.573 | 0.051 | 144.4s |
| 4 | 1.545 | 0.201 | 143.3s |
| 5 | 1.865 | 0.021 | 124.9s |

**Mean √PEHE = 1.038 ± 0.619**, mean ε_ATE = 0.077.