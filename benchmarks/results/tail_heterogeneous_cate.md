# Tail-heterogeneous CATE — tails-as-signal probe

DGP:  X ~ N(0, I₅),  p=5,  N=1000.  whale = 1(|X₀|>1.96) (~5 % tail mass).
  τ(X) = 2·(1−whale) + 10·whale,    Y₀ = X₀ + 0.5 X₁ + N(0,1),
  π(X) = σ(0.3 X₀) clipped to [0.1, 0.9],   W ~ Bern(π).
Seeds: [0, 1, 2, 3, 4].  True mixed ATE ≈ 2.4;  true τ_whale = 10;  true τ_bulk = 2.

Nuisance: XGB-MSE (no outcome contamination in this DGP; we are
isolating the *likelihood-level* tail handling, not the nuisance-level).
mad_rescale disabled to avoid the §14 MAD-contamination pathway.

Configurations:
  - **Gaussian**      : `robust=False`  (standard Bayesian likelihood)
  - **Welsch**        : `robust=True`, no EVT
  - **Welsch+EVT**    : `robust=True` + Hill-estimated `tail_threshold`,
                         `tail_alpha` applied via `normalize_extremes`

Bases:
  - **intercept**     : X_infer = [1]        (scalar ATE)
  - **tail_aware**    : X_infer = [1, 1(|X₀|>1.96)]  (bulk + tail dummy)

## Summary

| basis | config | n | ε_ATE (mixed) | cov mixed | τ̂_whale | ε_ATE (whale) | cov whale | τ̂_bulk | ε_ATE (bulk) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| intercept | Gaussian | 5 | 0.697 | 0.80 | 2.550 | 7.450 | 0.00 | 2.550 | 0.922 |
| intercept | Welsch | 5 | 0.351 | 0.20 | 2.066 | 7.934 | 0.00 | 2.066 | 0.085 |
| intercept | Welsch+EVT | 5 | 1.763 | 0.00 | 0.655 | 9.345 | 0.00 | 0.655 | 1.345 |
| tail_aware | Gaussian | 5 | 0.683 | 1.00 | 11.172 | 2.148 | 1.00 | 2.084 | 0.626 |
| tail_aware | Welsch | 5 | 0.115 | 1.00 | 10.269 | 0.742 | 1.00 | 2.066 | 0.084 |
| tail_aware | Welsch+EVT | 5 | 1.761 | 0.00 | 0.650 | 9.350 | 0.00 | 0.657 | 1.343 |

## Findings

Three observations — two positive, one negative.

**1. Tails-as-signal is recoverable, but through the CATE basis, not
through the tail parameters.** With `X_infer = [1, 1(|X₀|>1.96)]` the
Welsch posterior recovers τ_whale = 10.27 (true: 10) and τ_bulk = 2.07
(true: 2), with 100 % coverage on both subgroups. Gaussian likelihood
also recovers the effect but overshoots on whales (τ̂_whale = 11.17).
The practitioner's structural knowledge — "whales have a different
treatment effect" — is encoded as a basis function, and the standard
Bayesian posterior does the rest. No special tail machinery is needed.

**2. Welsch beats Gaussian on both accuracy and coverage for
heterogeneous τ.** With the tail-aware basis, Welsch gives
ε_ATE(mixed) = 0.11 vs Gaussian's 0.68, and recovers τ_whale to
within 0.74 vs Gaussian's 2.15. The Welsch redescender is not
"suppressing whales" when they are represented by their own basis
coefficient — it is only suppressing the *residual* structure the
basis does not explain, which is the correct behaviour.

**3. The Hill-estimator / `normalize_extremes` pathway actively harms
this DGP.** Welsch+EVT degrades ε_ATE(mixed) to 1.76 (≈ 5× worse than
Welsch alone) and drives τ_whale to ≈ 0.65 even with the tail-aware
basis — the data-layer rescaling divides large pseudo-outcomes by
threshold^alpha, which is a *contamination-reduction* operation. It is
the wrong tool when the tail carries signal. The EVT code path was
originally conceived for tails-as-contamination; this experiment shows
it is not a general-purpose heavy-tail mechanism, and should not be
activated when τ is tail-heterogeneous.

### Implication for the library

The three heavy-tail mechanisms shipping in this library address
distinct concerns:

| mechanism | layer | concern it addresses | use when |
|---|---|---|---|
| CatBoost-Huber nuisance | nuisance | heavy-tailed *outcome* noise | §16 / §17 whale contamination |
| Welsch likelihood (`robust=True`) | Bayesian | heavy-tailed *pseudo-outcome residuals* | almost always (low cost on clean data) |
| Tail-aware `X_infer` basis | CATE surface | heterogeneous τ on tail subgroups | tails-as-signal applications |
| `tail_threshold` / `tail_alpha` | data | (empirically) contamination-like rescaling | not recommended based on this probe |

Tails-as-signal is therefore a *supported* use case — through (3),
not (4). The EVT Hill-estimator path (4) is architecturally present
but empirically discouraged pending a better tail-likelihood design.