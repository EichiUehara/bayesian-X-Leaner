# Modular-Bayes nuisance uncertainty propagation

Bayesian-bootstrap M = 10 nuisance draws on the whale DGP, N = 1000, true ATE = 2.0.

Two pooling rules: (a) modular-cut concatenation; (b) Rubin's rules with
(1 + 1/M) between-imputation correction.

| density | severity | n | concat coverage | concat width | Rubin coverage | Rubin width |
|---:|---|---:|---:|---:|---:|---:|
| 0.00 | none | 3 | 1.00 | 0.289 | 1.00 | 0.306 |
| 0.00 | severe | 3 | 1.00 | 0.327 | 1.00 | 0.348 |
| 0.05 | none | 3 | 1.00 | 0.423 | 1.00 | 0.447 |
| 0.05 | severe | 3 | 1.00 | 0.335 | 1.00 | 0.354 |
| 0.20 | none | 3 | 1.00 | 0.563 | 1.00 | 0.584 |
| 0.20 | severe | 3 | 1.00 | 0.630 | 1.00 | 0.667 |