# Robust BART (contaminated-normal residuals) on IHDP

Heavy-tailed Bayesian tree backbone with two-component Gaussian mixture
residuals (eps_w ~ Beta(1,9), sigma_out free).
Note: BART splitting rule unchanged; only the leaf-likelihood is robustified.

| rep | √PEHE | ε_ATE | runtime |
|---:|---:|---:|---:|
| 1 | nan | nan | 4.4s |
| 2 | nan | nan | 0.0s |
| 3 | nan | nan | 0.0s |
| 4 | nan | nan | 0.1s |
| 5 | nan | nan | 0.0s |