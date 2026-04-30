#!/usr/bin/env bash
# Sequential reproduction driver — delete after use.
set -eo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=2"

log() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

log "run_cate_benchmark"
python -m benchmarks.run_cate_benchmark --seeds 8

log "run_nonlinear_cate"
python -m benchmarks.run_nonlinear_cate --seeds 8

log "run_pipeline_comparison"
python -m benchmarks.run_pipeline_comparison --seeds 15 \
    --dgps standard whale imbalance sharp_null

log "stability_check"
python -m benchmarks.stability_check --n-mcmc 6 --n-data 6

log "plot_results"
python -m benchmarks.plot_results

log "run_bart_comparison"
python -m benchmarks.run_bart_comparison --seeds 5

log "run_ihdp_benchmark"
python -m benchmarks.run_ihdp_benchmark --replications 10

log "DONE"
