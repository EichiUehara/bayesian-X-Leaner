# IHDP semi-synthetic benchmark

Dataset: Hill (2011) IHDP, replications [1, 2, 3, 4, 5] (from CEVAE preprocessing).
Covariates: 25 real features (N=747 per rep). Outcome simulated per Hill's response surface B.

Metrics:
- **√ε_PEHE** = √mean((τ̂ − τ)²)   — heterogeneous-effect recovery (lower is better)
- **ε_ATE** = |mean(τ̂) − mean(τ)| — average-effect error

| Estimator | n | √PEHE | std(√PEHE) | ε_ATE | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 5 | 0.562 | 0.200 | 0.079 | 42.07 |
| Huber-DR (point) | 5 | 0.575 | 0.153 | 0.037 | 51.57 |
| Causal BART (Hill 2011) | 5 | 0.597 | 0.239 | 0.140 | 595.31 |
| S-Learner | 5 | 0.720 | 0.363 | 0.091 | 2.48 |
| RX-Learner (robust+overlap) | 5 | 0.761 | 0.193 | 0.047 | 4.78 |
| T-Learner | 5 | 0.788 | 0.049 | 0.040 | 0.26 |
| X-Learner (std) | 5 | 0.936 | 0.361 | 0.028 | 6.48 |
| EconML Forest | 5 | 1.056 | 0.536 | 0.315 | 2.51 |
| RX-Learner (CB-Huber δ=1.345) | 5 | 1.232 | 0.457 | 0.739 | 4.22 |
| RX-Learner (CB-Huber δ=0.5) | 5 | 1.795 | 0.745 | 1.368 | 4.18 |
| RX-Learner (std) | 5 | 5.217 | 0.423 | 1.192 | 15.11 |