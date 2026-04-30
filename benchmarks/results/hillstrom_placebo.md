# Hillstrom placebo & symmetry sanity checks

Hillstrom RCT, N = 42613. RX-Welsch, intercept-only basis.
Placebo: W permuted at random; expected ATE ≈ 0.
Flipped: W → 1 − W; expected ATE = −original ATE.

| Setup | Severity | ATE | 95% CI |
|---|---|---:|---|
| Original W | none | -0.0026 | [-0.0292, +0.0201] |
| Original W | severe | -0.0001 | [-0.0092, +0.0092] |
| Placebo seed=0 | none | -0.0092 | [-0.0360, +0.0138] |
| Placebo seed=0 | severe | -0.0001 | [-0.0103, +0.0087] |
| Placebo seed=1 | none | -0.0079 | [-0.0339, +0.0171] |
| Placebo seed=1 | severe | +0.0000 | [-0.0098, +0.0095] |
| Placebo seed=2 | none | +0.0012 | [-0.0249, +0.0296] |
| Placebo seed=2 | severe | +0.0001 | [-0.0093, +0.0100] |
| Flipped W | none | -0.0168 | [-0.0416, +0.0081] |
| Flipped W | severe | -0.0002 | [-0.0091, +0.0092] |