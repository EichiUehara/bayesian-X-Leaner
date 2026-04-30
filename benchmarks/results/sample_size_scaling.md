# Sample-size scaling

Seeds: [0, 1, 2, 3, 4, 5, 6, 7]. N ∈ [200, 500, 1000, 2000, 5000].

Classical Bayesian consistency predicts RMSE ∝ 1/√N, i.e. a slope of **−0.5** in log-log. A slope near 0 means non-convergence.

## DGP: `standard`

| Variant | N=200 | N=500 | N=1000 | N=2000 | N=5000 | log-log slope |
|---|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 0.267 | 0.071 | 0.035 | 0.027 | 0.024 | -0.74 |
| RX-Learner (std) | 3.499 | 0.796 | 0.333 | 0.291 | 0.057 | -1.19 |

## DGP: `whale`

| Variant | N=200 | N=500 | N=1000 | N=2000 | N=5000 | log-log slope |
|---|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 0.266 | 0.092 | 0.230 | 3.756 | 7.289 | +1.29 |
| RX-Learner (std) | 24.832 | 22.988 | 21.789 | 14.130 | 19.984 | -0.11 |

## Interpretation

- **Slope near −0.5** → √N-consistent (canonical Bayesian rate).
- **Slope near 0 or positive** → non-convergent; extra data does not help (expected for RX-Learner std on whale: 1 % whale density is preserved as N grows).