# Basis-sensitivity ablation on the tail-heterogeneous DGP

DGP: tau_bulk = 2, tau_tail = 10, whale = 1(|X_0|>1.96), N = 1000, seeds = [0, 1, 2, 3, 4].
Welsch likelihood held fixed; only the CATE basis varies.

| basis | n | PEHE | std | ε(mixed) | ε(whale) | ε(bulk) | cov(mix) | cov(whale) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| intercept | 5 | 1.814 | 0.119 | 0.351 | 7.934 | 0.085 | 0.20 | 0.00 |
| linear_X0 | 5 | 1.822 | 0.121 | 0.355 | 7.953 | 0.084 | 0.20 | 0.00 |
| poly2 | 5 | 1.721 | 0.197 | 0.335 | 7.467 | 0.077 | 0.60 | 0.00 |
| tail_t15 | 5 | 1.767 | 0.162 | 0.342 | 7.704 | 0.081 | 0.20 | 0.00 |
| tail_t196 | 5 | 0.202 | 0.129 | 0.115 | 0.742 | 0.084 | 1.00 | 1.00 |
| tail_t25 | 5 | 1.592 | 0.139 | 0.266 | 6.219 | 0.080 | 0.20 | 0.00 |