# Bayesian baseline: RX-Learner vs Causal BART

Seeds: [0, 1, 2, 3, 4].  Causal BART = BART T-Learner (separate BART regressions for treated and control, ATE is posterior mean of μ̂₁ − μ̂₀).

## DGP: `standard`   (true ATE = 2.0)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 5 | +2.049 | +0.049 | 0.073 | 0.80 | 0.269 | 1.43 |
| Causal BART (T-Learner) | 5 | +2.112 | +0.112 | 0.126 | 0.80 | 0.271 | 21.39 |
| RX-Learner (std) | 5 | +1.907 | -0.093 | 0.497 | 1.00 | 2.571 | 1.78 |

## DGP: `whale`   (true ATE = 2.0)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | Mean CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RX-Learner (robust) | 5 | +2.067 | +0.067 | 0.072 | 1.00 | 0.258 | 1.54 |
| RX-Learner (std) | 5 | +5.637 | +3.637 | 17.916 | 0.80 | 38.250 | 2.07 |
| Causal BART (T-Learner) | 5 | -95.674 | -97.674 | 98.453 | 0.00 | 82.146 | 19.61 |
