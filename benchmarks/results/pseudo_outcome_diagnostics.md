# Pseudo-outcome distribution — empirical verification

Claim 1: DR-X-Learner pseudo-outcomes are heavy-tailed even when Y is Gaussian.

Under a Gaussian null hypothesis: excess kurtosis ≈ 0, Jarque-Bera p ≫ 0.05.
Heavy tails → positive excess kurtosis and very small JB p-value.

| Variable | n | mean | std | skew | excess kurt | max \|z\| | JB p-value | P99 \|r\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standard · raw Y − μ̂ | 600 | -0.010 | 0.681 | +0.11 | +0.94 | 4.5σ | 7.9e-06 | 1.90 |
| standard · D₁ (DR pseudo) | 295 | +0.221 | 15.481 | -8.64 | +88.32 | 12.1σ | 0.0e+00 | 55.44 |
| standard · D₀ (DR pseudo) | 305 | +0.818 | 11.880 | -6.81 | +57.86 | 10.5σ | 0.0e+00 | 71.46 |
| whale · raw Y − μ̂ | 600 | +0.666 | 535.303 | +7.10 | +65.99 | 9.9σ | 0.0e+00 | 1893.79 |
| whale · D₁ (DR pseudo) | 295 | -99.364 | 284.298 | -1.25 | +2.78 | 4.4σ | 3.3e-38 | 1020.83 |
| whale · D₀ (DR pseudo) | 305 | -147.569 | 2654.488 | -4.44 | +52.00 | 10.6σ | 0.0e+00 | 12745.85 |

## Propensity range (source of 1/π̂ blow-up)

- **standard**: π̂ ∈ [0.0100, 0.9900], min(π̂, 1−π̂) = 0.0100
- **whale**: π̂ ∈ [0.0100, 0.9900], min(π̂, 1−π̂) = 0.0100

## Interpretation

- If the pseudo-outcome rows have **much larger excess kurtosis** than the raw `Y − μ̂` rows, claim 1 is confirmed: the DR division by π̂ and (1−π̂) generates tails that the raw outcome didn't have.
- If `max |z|` is ≥ 5 on a standard DGP where Y itself is Gaussian, the Gaussian likelihood will pay ≥ 25× per observation at the tail — claim 2's mechanism.
- JB p ≪ 0.05 rejects Gaussianity at standard significance. A Gaussian likelihood assumed by non-robust MCMC is the wrong model.

## Why this causes a *systematic* bias (not just variance)

The DR numerator `(Y − μ̂)/π̂` is **asymmetrically distributed** when π̂ is skewed — i.e., when treated and control regions differ. Under `standard_dgp` the propensity is sigmoid-structured, so the rare `π̂ ≈ 0.02` cases land disproportionately on control units with negative residuals (or vice versa), producing the observed **−0.42 mean bias** in the non-robust RX-Learner.

## Figures

- `figures/pseudo_outcome_tails.png` — histogram + Q-Q plot vs Normal
- `figures/loss_influence.png` — Gaussian L2 vs Welsch influence function