"""
Monte-Carlo Pipeline Comparison Runner
======================================
Runs every estimator in ``benchmarks.estimators`` across every DGP in
``benchmarks.dgps`` with ``N_SEEDS`` random-seed replications and writes:

    benchmarks/results/results_raw.csv        one row per (estimator, DGP, seed)
    benchmarks/results/results_summary.md     aggregated markdown tables

Usage
-----
    python -m benchmarks.run_pipeline_comparison            # default 10 seeds
    python -m benchmarks.run_pipeline_comparison --seeds 30 --dgps whale standard
    python -m benchmarks.run_pipeline_comparison --fast     # quick smoke run
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

# Parallel MCMC chains (must be first)
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from benchmarks.dgps import DGPS
from benchmarks.estimators import ESTIMATORS
from benchmarks.metrics import aggregate, to_markdown_row, MARKDOWN_HEADER


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run(seeds: list[int], dgp_names: list[str], estimator_names: list[str], fast: bool):
    """Run the full Cartesian product and return raw + aggregated results."""
    raw_rows = []       # list of dicts for CSV
    summaries = {}      # (dgp_name, estimator_name) -> aggregated dict
    tau_per_dgp = {}    # dgp_name -> tau_true

    total = len(seeds) * len(dgp_names) * len(estimator_names)
    done = 0

    for dgp_name in dgp_names:
        dgp_fn = DGPS[dgp_name]
        per_est_runs = {est: [] for est in estimator_names}

        for seed in seeds:
            # Fast mode: tiny sample for smoke tests
            if fast:
                if dgp_name == "standard":
                    X, Y, W, tau = dgp_fn(N=200, P=4, tau=2.0, seed=seed)
                elif dgp_name == "whale":
                    X, Y, W, tau = dgp_fn(N=200, P=4, tau=2.0, n_whales=2, seed=seed)
                elif dgp_name == "imbalance":
                    X, Y, W, tau = dgp_fn(N=300, P=4, tau=2.0, seed=seed)
                else:  # sharp_null
                    X, Y, W, tau = dgp_fn(N=400, P=4, seed=seed)
            else:
                X, Y, W, tau = dgp_fn(seed=seed)
            tau_per_dgp[dgp_name] = tau

            for est_name in estimator_names:
                est_fn = ESTIMATORS[est_name]
                res = est_fn(X, Y, W)
                per_est_runs[est_name].append(res)
                raw_rows.append({
                    "dgp": dgp_name, "estimator": est_name, "seed": seed,
                    "tau_true": tau,
                    **{k: ("" if v is None else v) for k, v in res.items()},
                })
                done += 1
                status = "OK" if res["ate"] is not None else "FAIL"
                print(f"  [{done:>3d}/{total:>3d}] {dgp_name:<11s} {est_name:<22s} "
                      f"seed={seed:<3d} {status:<4s} "
                      f"ate={res['ate'] if res['ate'] is not None else '—':>8}  "
                      f"rt={res['runtime']:.2f}s")

        for est_name in estimator_names:
            summaries[(dgp_name, est_name)] = aggregate(
                per_est_runs[est_name], tau_true=tau_per_dgp[dgp_name]
            )

    return raw_rows, summaries, tau_per_dgp


def write_csv(raw_rows: list[dict], path: Path):
    if not raw_rows:
        return
    keys = list(raw_rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(raw_rows)


def write_markdown(summaries: dict, tau_per_dgp: dict, dgp_names: list[str],
                   estimator_names: list[str], seeds: list[int], path: Path):
    lines = [
        "# Pipeline Comparison — Monte Carlo Results",
        "",
        f"Seeds per DGP: **{len(seeds)}**   |   Seeds: `{seeds}`",
        "",
        "Metrics are aggregated over seeds:",
        "",
        "- **Mean ATE** — average point estimate across seeds",
        "- **Bias** — Mean ATE − true ATE (positive = over-estimate)",
        "- **RMSE** — √E[(ATE − τ)²], lower is better",
        "- **Coverage** — fraction of seeds where 95 % CI contains τ (target ≈ 0.95)",
        "- **Mean CI Width** — efficiency (narrower is better, if coverage ≥ 0.90)",
        "- **Runtime (s)** — mean wall-clock seconds per fit",
        "- **Success** — `n_ok/n_total` (failure = exception in wrapper)",
        "",
    ]

    for dgp_name in dgp_names:
        tau = tau_per_dgp.get(dgp_name, float("nan"))
        lines.append(f"## DGP: `{dgp_name}`   (true ATE = {tau})")
        lines.append("")
        lines.append(MARKDOWN_HEADER)

        # Sort by RMSE for the final table (best → worst)
        rows_with_rmse = []
        for est_name in estimator_names:
            m = summaries[(dgp_name, est_name)]
            rows_with_rmse.append((m["rmse"], est_name, m))
        rows_with_rmse.sort(key=lambda x: (x[0] if x[0] == x[0] else float("inf")))

        for _, est_name, m in rows_with_rmse:
            lines.append(to_markdown_row(est_name, m))
        lines.append("")

    path.write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of Monte Carlo replications (default 10)")
    ap.add_argument("--dgps", nargs="+", default=list(DGPS.keys()),
                    choices=list(DGPS.keys()),
                    help="DGPs to run (default: all)")
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATORS.keys()),
                    help="Estimators to run (default: all)")
    ap.add_argument("--fast", action="store_true",
                    help="Use tiny datasets — smoke test only")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="First seed (default 0)")
    args = ap.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print(f"\nPipeline Comparison — {len(seeds)} seeds × {len(args.dgps)} DGPs × "
          f"{len(args.estimators)} estimators = {len(seeds)*len(args.dgps)*len(args.estimators)} fits")
    print(f"Fast mode: {args.fast}\n")

    raw_rows, summaries, tau_per_dgp = run(seeds, args.dgps, args.estimators, args.fast)

    csv_path = RESULTS_DIR / "results_raw.csv"
    md_path = RESULTS_DIR / "results_summary.md"
    write_csv(raw_rows, csv_path)
    write_markdown(summaries, tau_per_dgp, args.dgps, args.estimators, seeds, md_path)

    print(f"\nWrote {len(raw_rows)} rows → {csv_path}")
    print(f"Wrote summary      → {md_path}\n")


if __name__ == "__main__":
    main()
