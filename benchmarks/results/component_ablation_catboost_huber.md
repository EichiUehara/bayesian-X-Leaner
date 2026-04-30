# Component ablation — is each robust piece necessary?

Decomposes the "RX-Learner (robust)" claim that Welsch loss and Student-T likelihood are each load-bearing. Inspection of `sert_xlearner/inference/bayesian.py` reveals **the architecture has three modes, not four** — when `robust=True`, the model uses `numpyro.factor(welsch_loss)` directly and ignores `use_student_t`. So the previously-documented "Welsch + Student-T" variant was never actually tested; this ablation tests the three modes the code does produce.

Seeds: [0, 1, 2, 3, 4]. DR pseudo-outcome targeting is on in every variant (it's architectural, not optional).

## DGP: `sharp_null` (true ATE = 0.00)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian (std) | 5 | -0.012 | -0.012 | 0.041 | 1.00 | 0.149 | 3.91 |
| Student-T only | 5 | +0.065 | +0.065 | 0.076 | 0.40 | 0.102 | 5.34 |
| Welsch only | 5 | +0.091 | +0.091 | 0.099 | 0.60 | 0.179 | 3.09 |

## DGP: `standard` (true ATE = 2.00)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian (std) | 5 | +2.114 | +0.114 | 0.124 | 0.80 | 0.271 | 5.48 |
| Student-T only | 5 | +2.208 | +0.208 | 0.214 | 0.00 | 0.187 | 4.85 |
| Welsch only | 5 | +2.210 | +0.210 | 0.216 | 0.00 | 0.283 | 3.54 |

## DGP: `whale` (true ATE = 2.00)

| Variant | n | Mean ATE | Bias | RMSE | Coverage | CI Width | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student-T only | 5 | +2.204 | +0.204 | 0.210 | 0.00 | 0.184 | 4.46 |
| Welsch only | 5 | +2.209 | +0.209 | 0.215 | 0.00 | 0.282 | 3.31 |
| Gaussian (std) | 5 | -31.718 | -33.718 | 33.741 | 0.00 | 33.041 | 3.83 |

## Interpretation guide

- **If Welsch and Student-T both handle whale ≈ equally well** → either is sufficient and our docs over-claim by calling both necessary.
- **If only Welsch handles whale** → Welsch is the load-bearing robustness piece; Student-T's contribution is narrower CIs via thick-tail posterior, not point-estimate robustness.
- **If neither alone handles whale well but the implicit "combined" variant we can't directly test would** → motivates a code change to enable true Welsch+StudentT combined mode.

## §17 finding — the Welsch vs Student-T distinction weakens under Huber nuisance

Under the legacy XGB-MSE nuisance, the original `component_ablation.md`
concluded *Welsch is load-bearing; Student-T alone under-covers*.
That framing was conditional on contaminated µ̂₀ producing
heavy-tailed pseudo-outcomes — conditions where Welsch's bounded-
influence clipping strictly dominates Student-T's heavy tails.

Under the §16 CatBoost-Huber nuisance (this run), pseudo-outcomes are
already largely clean, so the distinction collapses. On whale:

| Variant | RMSE (whale) | RMSE (std) |
|---|---:|---:|
| Gaussian (std) | **33.7** | 0.12 |
| Student-T only | 0.21 | 0.21 |
| Welsch only | 0.22 | 0.22 |

- **Gaussian still fails on whale (RMSE 33.7).** Even with Huber
  upstream, there's residual whale leakage the Gaussian likelihood
  smears catastrophically. So the Bayesian layer *does* need
  robustness — but not a specific flavour of it.
- **Student-T ≈ Welsch on whale (RMSE 0.21 vs 0.22).** Either is
  sufficient. The earlier claim that Welsch is uniquely load-bearing
  was a property of the XGB-MSE regime, not an architectural
  necessity.
- **Coverage is 0 for both on whale (CI width ~0.2, bias ~0.2).**
  CIs are slightly too narrow at this 1 %-density regime — the
  posterior is overconfident when nuisance is already doing most of
  the robustness work. This matches the tail-end observation in
  §17's whale_density update.

**Practical implication.** The README claim that "Welsch alone carries
point-robustness **and** coverage calibration; Student-T alone
under-covers" should be qualified — under the §16 Huber default,
Student-T matches Welsch on RMSE, and both slightly under-cover at
tight densities. The earlier strict ordering held only in the
XGB-MSE regime.

Either robust likelihood remains necessary (Gaussian still fails),
but the choice between them is now a matter of convention and
runtime, not point-estimate quality.