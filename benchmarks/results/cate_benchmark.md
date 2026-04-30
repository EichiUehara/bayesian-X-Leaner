# CATE (heterogeneous-effect) benchmark

DGP: `heterogeneous_cate_dgp`, τ(x) = 2 + x₀, N=1000.
Seeds: [0, 1, 2, 3, 4, 5, 6, 7]

**PEHE** (Precision in Estimating Heterogeneous Effects) = √E[(τ̂ − τ)²] — lower is better.
**corr** = Pearson correlation between τ̂(x) and τ(x) — 1.0 is best.

| Estimator | n | Mean PEHE | Std PEHE | Mean Bias | Mean Corr | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 8 | 0.083 | 0.053 | +0.034 | +1.000 | 6.51 |
| S-Learner | 8 | 0.271 | 0.037 | -0.019 | +0.965 | 0.41 |
| EconML Forest | 8 | 0.291 | 0.060 | -0.057 | +0.973 | 2.30 |
| X-Learner (std) | 8 | 0.326 | 0.024 | +0.004 | +0.950 | 1.15 |
| T-Learner | 8 | 0.432 | 0.027 | +0.019 | +0.915 | 0.52 |
| RX-Learner (std) | 8 | 0.589 | 0.348 | -0.348 | +1.000 | 6.44 |