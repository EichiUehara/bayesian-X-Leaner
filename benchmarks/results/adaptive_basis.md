# Adaptive-basis BMA for tails-as-signal

Candidate thresholds: [1.0, 1.25, 1.5, 1.75, 1.96, 2.25, 2.5, 3.0]. True threshold = 1.96.
BMA over discrete thresholds, weights ∝ softmax(-PEHE_proxy / temperature).
Tail-heterogeneous DGP (τ_bulk=2.0, τ_tail=10.0), N = 1000, 5 seeds.

| seed | PEHE | cov_pw | cov_whale | cov_bulk | best_c | w(1.96) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.590 | 0.97 | 0.31 | 1.00 | 1.96 | 0.57 |
| 1 | 0.410 | 0.05 | 1.00 | 0.00 | 1.96 | 0.62 |
| 2 | 0.609 | 0.95 | 0.19 | 1.00 | 1.96 | 0.53 |

**Mean PEHE = 0.536 ± 0.110**, mean pointwise coverage = 0.66, mean weight on true threshold (1.96) = 0.58.