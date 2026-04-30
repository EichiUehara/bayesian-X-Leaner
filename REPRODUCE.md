# Reproducing the RX-Learner benchmark results

Every published number in [benchmarks/results/](benchmarks/results/) comes
from one of the scripts below. All scripts are deterministic given a fixed
seed list; runtime is measured on a 2-chain MCMC laptop setup.

## 0 · Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e '.[test,benchmark]'
```

Full install is required for the BART comparison
(`pymc-bart`) and the SOTA baselines (`econml`, `doubleml`, `causalml`).

## 1 · Full reproduction (~90 min total)

```bash
source venv/bin/activate

# Core evidence — stability + Monte Carlo across 4 DGPs (45 min)
python -m benchmarks.run_pipeline_comparison --seeds 15 \
    --dgps standard whale imbalance sharp_null
python -m benchmarks.stability_check --n-mcmc 6 --n-data 6
python -m benchmarks.plot_results

# Mechanism check — why robust beats standard (< 1 min)
python -m benchmarks.verify_pseudo_outcome

# Follow-up experiments (~40 min)
python -m benchmarks.run_overlap_experiment --seeds 8      # ~1 min
python -m benchmarks.run_cate_benchmark --seeds 8          # ~5 min
python -m benchmarks.run_bart_comparison --seeds 5         # ~15 min
python -m benchmarks.run_nonlinear_cate --seeds 8          # ~8 min
python -m benchmarks.run_ihdp_benchmark --replications 10  # ~15 min
python -m benchmarks.run_ihdp_basis_ablation --replications 10  # ~10 min

# Robustness ablations — decompose the "robust" machinery (~45 min)
python -m benchmarks.run_component_ablation --seeds 15     # ~15 min
python -m benchmarks.run_c_whale_sensitivity --seeds 8     # ~8 min
python -m benchmarks.run_sample_size_scaling --seeds 8     # ~20 min

# Breakdown boundary + §9 mechanism verification + §14 MAD-rescaling bug (~100 min)
python -m benchmarks.run_whale_density --seeds 8           # ~14 min
python -m benchmarks.run_nuisance_depth --seeds 8          # ~7 min
python -m benchmarks.run_n_splits_sensitivity --seeds 8    # ~6 min
python -m benchmarks.run_prior_scale_sensitivity --seeds 8 # ~18 min  (§14 initial sweep)
python -m benchmarks.run_mad_rescaling_and_prior --seeds 8 # ~35 min  (§14 corrected sweep)
python -m benchmarks.run_robust_nuisance --seeds 8         # ~15 min  (§15 subsumption test)
python -m benchmarks.run_nuisance_loss_sweep --seeds 8     # ~12 min  (§16 Huber-δ tuning)
```

Each script writes a markdown report and a raw CSV into
[benchmarks/results/](benchmarks/results/).

## 2 · Where each reported number lives

| Report | Script | Output |
|---|---|---|
| [STABILITY_SUMMARY.md](benchmarks/results/STABILITY_SUMMARY.md) | `run_pipeline_comparison`, `stability_check`, `plot_results` | `results_raw.csv`, `results_summary.md`, `stability_report.md`, `figures/*.png` |
| [pseudo_outcome_diagnostics.md](benchmarks/results/pseudo_outcome_diagnostics.md) | `verify_pseudo_outcome` | `figures/pseudo_outcome_tails.png`, `figures/loss_influence.png` |
| [overlap_experiment.md](benchmarks/results/overlap_experiment.md) | `run_overlap_experiment` | `overlap_experiment_raw.csv` |
| [cate_benchmark.md](benchmarks/results/cate_benchmark.md) | `run_cate_benchmark` | `cate_benchmark_raw.csv` |
| [bart_comparison.md](benchmarks/results/bart_comparison.md) | `run_bart_comparison` | `bart_comparison_raw.csv` |
| [nonlinear_cate.md](benchmarks/results/nonlinear_cate.md) | `run_nonlinear_cate` | `nonlinear_cate_raw.csv` |
| [ihdp_benchmark.md](benchmarks/results/ihdp_benchmark.md) | `run_ihdp_benchmark` | `ihdp_benchmark_raw.csv` |
| [ihdp_basis_ablation.md](benchmarks/results/ihdp_basis_ablation.md) | `run_ihdp_basis_ablation` | `ihdp_basis_ablation_raw.csv` |
| [component_ablation.md](benchmarks/results/component_ablation.md) | `run_component_ablation` | `component_ablation_raw.csv` |
| [c_whale_sensitivity.md](benchmarks/results/c_whale_sensitivity.md) | `run_c_whale_sensitivity` | `c_whale_sensitivity_raw.csv` |
| [sample_size_scaling.md](benchmarks/results/sample_size_scaling.md) | `run_sample_size_scaling` | `sample_size_scaling_raw.csv` |
| [whale_density.md](benchmarks/results/whale_density.md) | `run_whale_density` | `whale_density_raw.csv` |
| [nuisance_depth.md](benchmarks/results/nuisance_depth.md) | `run_nuisance_depth` | `nuisance_depth_raw.csv` |
| [n_splits_sensitivity.md](benchmarks/results/n_splits_sensitivity.md) | `run_n_splits_sensitivity` | `n_splits_sensitivity_raw.csv` |
| [prior_scale_sensitivity.md](benchmarks/results/prior_scale_sensitivity.md) | `run_prior_scale_sensitivity` | `prior_scale_sensitivity_raw.csv` |
| [mad_rescaling_and_prior.md](benchmarks/results/mad_rescaling_and_prior.md) | `run_mad_rescaling_and_prior` | `mad_rescaling_and_prior_raw.csv` |
| [robust_nuisance.md](benchmarks/results/robust_nuisance.md) | `run_robust_nuisance` | `robust_nuisance_raw.csv` |
| [nuisance_loss_sweep.md](benchmarks/results/nuisance_loss_sweep.md) | `run_nuisance_loss_sweep` | `nuisance_loss_sweep_raw.csv` |

## 3 · Unit tests

Statistical-validity tests (orthogonalization, regularization leakage, MCMC
convergence, stress DGPs) run in < 2 min:

```bash
pytest tests/
```

Multi-seed Monte-Carlo tests are marked `@pytest.mark.benchmark` and
deselected by default; enable with `pytest -m benchmark tests/`.

## 4 · Reproducibility notes

- **Seeds.** Every run script accepts `--seeds N` (or `--replications N` for
  IHDP). The seed list is `range(N)` — fixed and deterministic.
- **MCMC.** `TargetedBayesianXLearner(random_state=42)` fixes the NumPyro PRNG
  key. Different host CPU counts can still reorder chain interleaving; set
  `XLA_FLAGS="--xla_force_host_platform_device_count=2"` (the run scripts do
  this automatically) for identical trace output.
- **Thread caps.** If another CPU-bound job is running, results match but
  wall-clock numbers inflate. Set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  OPENBLAS_NUM_THREADS=2` for the runtime figures in the reports.
- **IHDP data.** Committed at [benchmarks/data/ihdp_*.csv](benchmarks/data/)
  — CEVAE preprocessing of Hill (2011) Response Surface B, 10 replications
  of N=747.
