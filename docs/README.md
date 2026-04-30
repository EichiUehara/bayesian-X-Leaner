# Documentation

Internal design and development notes. The headline docs at the
project root are [README.md](../README.md) (overview, install, usage)
and [REPRODUCE.md](../REPRODUCE.md) (exact commands to reproduce all
paper results).

## Contents

- [motivation.md](motivation.md) — why the Bayesian X-Learner; the
  joint heterogeneity / calibration / robustness problem.
- [architecture.md](architecture.md) — three-phase pipeline deep
  dive (nuisance quarantine, X-learner imputation + targeted
  debiasing, Bayesian update).
- [implementation.md](implementation.md) — concrete mathematical and
  algorithmic specifics of the implementation.
- [baselines-index.md](baselines-index.md) — index of comparison
  algorithms across the SOTA causal-inference / robustness landscape.
- [testing-strategy.md](testing-strategy.md) — simulation-driven
  development and the regression vs stress test split.
- [research-roadmap.md](research-roadmap.md) — open questions and
  planned extensions.

These notes were originally working documents; they have been
preserved for transparency but have not been polished as public
documentation.
