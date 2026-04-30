# Component ablation — is each robust piece necessary?

Decomposes the "RX-Learner (robust)" claim that Welsch loss and Student-T likelihood are each load-bearing. Inspection of `sert_xlearner/inference/bayesian.py` reveals **the architecture has three modes, not four** — when `robust=True`, the model uses `numpyro.factor(welsch_loss)` directly and ignores `use_student_t`. So the previously-documented "Welsch + Student-T" variant was never actually tested; this ablation tests the three modes the code does produce.

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]. DR pseudo-outcome targeting is on in every variant (it's architectural, not optional).

## DGP: `sharp_null` (true ATE = 0.00)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Welsch only | 15 | +0.037 | +0.037 | 0.043 | 1.00 | 0.174 | 6.49 |
| Student-T only | 15 | +0.038 | +0.038 | 0.043 | 0.67 | 0.095 | 9.68 |
| Gaussian (std) | 15 | -0.143 | -0.143 | 0.291 | 0.93 | 1.174 | 7.34 |

## DGP: `standard` (true ATE = 2.00)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Welsch only | 15 | +2.076 | +0.076 | 0.095 | 0.87 | 0.269 | 4.02 |
| Student-T only | 15 | +2.081 | +0.081 | 0.099 | 0.47 | 0.175 | 6.81 |
| Gaussian (std) | 15 | +1.577 | -0.423 | 0.713 | 1.00 | 2.893 | 5.72 |

## DGP: `whale` (true ATE = 2.00)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Welsch only | 15 | +2.059 | +0.059 | 0.081 | 0.93 | 0.260 | 3.65 |
| Student-T only | 15 | +2.053 | +0.053 | 0.086 | 0.87 | 0.224 | 5.61 |
| Gaussian (std) | 15 | +7.213 | +5.213 | 20.790 | 0.60 | 38.318 | 4.85 |

## Interpretation guide

- **If Welsch and Student-T both handle whale ≈ equally well** → either is sufficient and our docs over-claim by calling both necessary.
- **If only Welsch handles whale** → Welsch is the load-bearing robustness piece; Student-T's contribution is narrower CIs via thick-tail posterior, not point-estimate robustness.
- **If neither alone handles whale well but the implicit "combined" variant we can't directly test would** → motivates a code change to enable true Welsch+StudentT combined mode.