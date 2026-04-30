# IHDP basis ablation

Tests whether RX-Learner's IHDP loss to T-Learner is entirely a matter of CATE basis misspecification — the assumption raised in [EXTENSIONS.md § 5](EXTENSIONS.md). All RX-Learner variants use identical nuisance, MCMC, and robust likelihood; they differ only in the `X_infer` basis passed to the Bayesian CATE regression.

Replications: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (Hill 2011 / CEVAE, N=747).

| Variant | n | √PEHE | std(√PEHE) | ε_ATE | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| T-Learner (control) | 10 | 1.373 | 1.634 | 0.110 | 0.18 |
| RX-Learner (interactions basis) | 10 | 1.531 | 2.663 | 0.162 | 10.91 |
| RX-Learner (quadratic basis) | 10 | 1.769 | 3.159 | 0.204 | 18.30 |
| RX-Learner (linear basis) | 10 | 1.951 | 3.584 | 0.252 | 2.91 |
| RX-Learner (Nyström RBF basis) | 10 | 2.394 | 3.982 | 0.212 | 2.86 |

## Interpretation

- **If Nyström RBF ≤ T-Learner** → assumption verified; the IHDP gap is purely basis, and RX-Learner's machinery is fine once the functional form is flexible enough.
- **If richer bases close most but not all of the gap** → basis dominates but there is a residual efficiency gap.
- **If no basis closes the gap** → the limitation is elsewhere (shrinkage, effective sample size, DR residual variance under high-dim basis).