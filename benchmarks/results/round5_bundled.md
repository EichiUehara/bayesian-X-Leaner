# Round-5 reviewer-response bundled experiments

## 1. Arm-specific contamination (whales on treated arm only)

| seed | density | contamination | ATE | coverage | CI width |
|---:|---:|---|---:|---:|---:|
| 0 | 0.05 | symmetric | +2.162 | 0 | 0.233 |
| 0 | 0.05 | treated_only | +2.144 | 0 | 0.228 |
| 0 | 0.20 | symmetric | +2.266 | 0 | 0.256 |
| 0 | 0.20 | treated_only | +2.236 | 0 | 0.257 |
| 1 | 0.05 | symmetric | +2.074 | 1 | 0.237 |
| 1 | 0.05 | treated_only | +2.093 | 1 | 0.227 |
| 1 | 0.20 | symmetric | +2.121 | 1 | 0.258 |
| 1 | 0.20 | treated_only | +2.086 | 1 | 0.253 |
| 2 | 0.05 | symmetric | +2.234 | 0 | 0.247 |
| 2 | 0.05 | treated_only | +2.208 | 0 | 0.241 |
| 2 | 0.20 | symmetric | +2.245 | 0 | 0.264 |
| 2 | 0.20 | treated_only | +2.216 | 0 | 0.283 |

## 4. Continuous τ(x) CATE coverage (smooth heterogeneity)

| seed | √PEHE | cov pointwise | τ̂ range | true τ range |
|---:|---:|---:|---|---|
| 0 | 0.264 | 0.65 | [0.94, 6.03] | [2.01, 4.99] |
| 1 | 0.473 | 0.81 | [-0.29, 8.47] | [2.01, 5.00] |
| 2 | 0.300 | 0.63 | [0.40, 6.68] | [2.00, 4.99] |

## 5. Contaminated-normal Phase-3 likelihood

| seed | density | severity | ATE | coverage | CI width |
|---:|---:|---|---:|---:|---:|
| 0 | 0.00 | none | +2.082 | 0 | 0.144 |
| 0 | 0.00 | severe | +2.143 | 0 | 0.128 |
| 0 | 0.05 | none | +409.758 | 0 | 23.140 |
| 0 | 0.05 | severe | -516.506 | 0 | 0.904 |
| 0 | 0.20 | none | -1855.059 | 0 | 20.776 |
| 0 | 0.20 | severe | -2075.152 | 0 | 1.067 |
| 1 | 0.00 | none | +2.015 | 1 | 0.150 |
| 1 | 0.00 | severe | +2.107 | 0 | 0.131 |
| 1 | 0.05 | none | -631.052 | 0 | 0.137 |
| 1 | 0.05 | severe | -469.830 | 0 | 3.143 |
| 1 | 0.20 | none | -1876.047 | 0 | 12.277 |
| 1 | 0.20 | severe | -1984.505 | 0 | 1.147 |
| 2 | 0.00 | none | +2.009 | 1 | 0.148 |
| 2 | 0.00 | severe | +2.126 | 0 | 0.137 |
| 2 | 0.05 | none | -97.708 | 0 | 0.682 |
| 2 | 0.05 | severe | -485.147 | 0 | 0.158 |
| 2 | 0.20 | none | -2010.732 | 0 | 122.400 |
| 2 | 0.20 | severe | -1932.376 | 0 | 10.739 |

## 7. Spline-basis CATE

| seed | PEHE | τ̂_whale | τ̂_bulk |
|---:|---:|---:|---:|
| 0 | 1.265 | 4.62 | 1.94 |
| 1 | 1.391 | 9.09 | 2.30 |
| 2 | 1.710 | 3.09 | 2.01 |