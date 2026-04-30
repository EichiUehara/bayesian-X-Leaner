# Nonlinear CATE benchmark

DGP: `nonlinear_cate_dgp`, τ(x) = 2 + sin(2·x₀), N=1000.
Seeds: [0, 1, 2, 3, 4, 5, 6, 7]

The DGP's CATE has curvature *not* in the linear basis [1, x₀]. RX-Learner variants differ only in the `X_infer` basis passed to the Bayesian CATE regression — same nuisance, same MCMC, same likelihood.

| Estimator | n | Mean PEHE | Std PEHE | Mean Bias | Mean Corr | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| RX-Learner (Fourier basis) | 8 | 0.097 | 0.039 | +0.033 | +0.992 | 6.19 |
| S-Learner | 8 | 0.257 | 0.017 | +0.002 | +0.933 | 0.27 |
| EconML Forest | 8 | 0.296 | 0.023 | -0.052 | +0.941 | 2.24 |
| X-Learner (std) | 8 | 0.302 | 0.016 | +0.009 | +0.914 | 0.93 |
| T-Learner | 8 | 0.404 | 0.022 | +0.021 | +0.863 | 0.39 |
| RX-Learner (linear basis) | 8 | 0.662 | 0.012 | +0.028 | +0.381 | 6.36 |
| RX-Learner (polynomial basis) | 8 | 0.697 | 0.084 | +0.049 | +0.358 | 5.54 |