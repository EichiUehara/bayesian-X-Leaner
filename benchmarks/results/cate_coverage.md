# CATE-level coverage under contamination variants

Tail-heterogeneous DGP (τ_bulk=2.0, τ_tail=10.0, whale=|X₀|>1.96), N = 1000, 5 seeds.
Tail-aware basis $[1, \mathbf{1}(|X_0|>1.96)]$, `severity=severe`, MCMC posterior credible intervals.

Pointwise coverage: fraction of units i where τ(x_i) ∈ CI_i.

| setup | n | √PEHE | cov pointwise | cov whale | cov bulk | mean CI width |
|---|---:|---:|---:|---:|---:|---:|
| clean | 5 | 1.029 | 0.97 | 0.40 | 1.00 | 0.855 |
| contam_5% | 5 | 1.108 | 0.60 | 0.60 | 0.60 | 0.947 |
| contam_20% | 5 | 1.194 | 0.79 | 0.60 | 0.80 | 1.161 |
| low_overlap | 5 | 1.488 | 1.00 | 1.00 | 1.00 | 2.203 |
| t_nu2_noise | 5 | 1.426 | 0.97 | 0.40 | 1.00 | 0.905 |