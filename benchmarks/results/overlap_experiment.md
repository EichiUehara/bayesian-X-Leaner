# Overlap-weights experiment on imbalance DGP

DGP: `imbalance_dgp` (treatment_prob = 0.95, ~50 controls of 1000). True ATE = 2.0.
Seeds: [0, 1, 2, 3, 4, 5, 6, 7]

Tests whether `use_overlap=True` (bounded overlap weights, Li 2018) closes the coverage gap of the DR-AIPW variant.

| Variant | n | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust+overlap) | 8 | +1.964 | -0.036 | 0.087 | 1.00 | 0.818 | 5.54 |
| RX-Learner (robust) | 8 | +2.038 | +0.038 | 0.111 | 0.75 | 0.217 | 4.90 |
| RX-Learner (std) | 8 | +1.627 | -0.373 | 0.716 | 0.88 | 1.801 | 6.00 |

## Verdict

**Overlap weights improve coverage** by +0.25 (0.75 → 1.00). The limitation flagged in STABILITY_SUMMARY.md is resolvable.
