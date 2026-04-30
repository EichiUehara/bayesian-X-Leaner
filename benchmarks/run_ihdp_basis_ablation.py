"""
IHDP basis ablation — tests whether RX-Learner's IHDP loss to T-Learner
is *entirely* basis misspecification (falsifiable).

Runs RX-Learner (robust) on IHDP with four CATE bases of increasing
flexibility, plus T-Learner as the non-parametric control:

    linear        : [1, x_0, ..., x_24]                           (26 dims)
    quadratic     : linear + x_i²                                 (51 dims)
    interactions  : linear + pairwise products of top-5 features  (~40 dims)
    nystrom_rbf   : 50-dim Nyström RBF approximation              (50 dims)

Assumption under test (from EXTENSIONS.md § 5):
  "RX-Learner is 2nd on IHDP only because its linear-in-25-features basis
  cannot represent Hill's Response Surface B curvature. A richer basis
  should close (or close most of) the gap to T-Learner."

Verdict rule:
  - If nystrom_rbf PEHE ≤ T-Learner PEHE → assumption verified.
  - If all richer bases are within ~10% of T-Learner → mostly verified.
  - If no richer basis helps → assumption falsified; something else (e.g.
    Bayesian shrinkage at N=747 × high-dim) is limiting RX-Learner.

Usage:
    python -m benchmarks.run_ihdp_basis_ablation --replications 10
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

from benchmarks.dgps import load_ihdp


RESULTS_DIR = Path(__file__).parent / "results"


def _make_reg():
    return HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42)


def _make_clf():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)


def basis_linear(X):
    return np.column_stack([np.ones(len(X)), X])


def basis_quadratic(X):
    return np.column_stack([np.ones(len(X)), X, X ** 2])


def basis_interactions(X, Y=None, top_k=5):
    """Top-k features by mutual-information with Y, plus their pairwise products."""
    if Y is None:
        top = list(range(top_k))
    else:
        mi = mutual_info_regression(X, Y, random_state=42)
        top = np.argsort(mi)[-top_k:]
    main = X
    pairs = np.column_stack(
        [X[:, i] * X[:, j] for i in top for j in top if i <= j]
    )
    return np.column_stack([np.ones(len(X)), main, pairs])


def basis_nystrom_rbf(X, n_components=50):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    gamma = 1.0 / Xs.shape[1]
    feat = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components,
                    random_state=42).fit_transform(Xs)
    return np.column_stack([np.ones(len(X)), feat])


def cate_t_learner(X, Y, W):
    mu0 = _make_reg(); mu0.fit(X[W == 0], Y[W == 0])
    mu1 = _make_reg(); mu1.fit(X[W == 1], Y[W == 1])
    return mu1.predict(X) - mu0.predict(X)


def cate_rx_learner(X, Y, W, X_infer):
    from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner
    model = TargetedBayesianXLearner(
        outcome_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        propensity_model_params={"max_depth": 4, "n_estimators": 150, "verbosity": 0},
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        robust=True, c_whale=1.34,
        use_student_t=True, use_overlap=False,
        random_state=42,
    )
    model.fit(X, Y, W, X_infer=X_infer)
    cate, _, _ = model.predict(X_new_infer=X_infer)
    return np.asarray(cate).flatten()


VARIANTS = {
    "T-Learner (control)":               ("t_learner", None),
    "RX-Learner (linear basis)":         ("rx", basis_linear),
    "RX-Learner (quadratic basis)":      ("rx", basis_quadratic),
    "RX-Learner (interactions basis)":   ("rx", basis_interactions),
    "RX-Learner (Nyström RBF basis)":    ("rx", basis_nystrom_rbf),
}


def _pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def _ate_err(tau_hat, tau):
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def _run(replications):
    rows = []
    for rep in replications:
        X, Y, W, tau = load_ihdp(rep)
        print(f"\n── IHDP replication {rep} ──")
        for name, (kind, basis_fn) in VARIANTS.items():
            t0 = time.time()
            try:
                if kind == "t_learner":
                    tau_hat = cate_t_learner(X, Y, W)
                else:
                    X_infer = (
                        basis_fn(X, Y) if basis_fn is basis_interactions
                        else basis_fn(X)
                    )
                    tau_hat = cate_rx_learner(X, Y, W, X_infer)
                pehe = _pehe(tau_hat, tau)
                ate_err = _ate_err(tau_hat, tau)
                err = None
            except Exception as e:
                pehe = ate_err = float("nan")
                err = str(e)
            rt = time.time() - t0
            rows.append({
                "variant": name, "replication": rep,
                "pehe": pehe, "ate_err": ate_err,
                "runtime": rt, "error": err,
            })
            print(f"  {name:<34}  √PEHE={pehe:.3f}  ε_ATE={ate_err:.3f}  "
                  f"({rt:.1f}s)" + (f"  ERR={err}" if err else ""))
    return pd.DataFrame(rows)


def _summarise(df):
    return (df.dropna(subset=["pehe"]).groupby("variant")
              .agg(mean_pehe=("pehe", "mean"),
                   std_pehe=("pehe", "std"),
                   mean_ate_err=("ate_err", "mean"),
                   mean_rt=("runtime", "mean"),
                   n=("replication", "count"))
              .sort_values("mean_pehe"))


def _write_markdown(df, agg, reps):
    path = RESULTS_DIR / "ihdp_basis_ablation.md"
    lines = [
        "# IHDP basis ablation",
        "",
        "Tests whether RX-Learner's IHDP loss to T-Learner is entirely a "
        "matter of CATE basis misspecification — the assumption raised in "
        "[EXTENSIONS.md § 5](EXTENSIONS.md). All RX-Learner variants use "
        "identical nuisance, MCMC, and robust likelihood; they differ only "
        "in the `X_infer` basis passed to the Bayesian CATE regression.",
        "",
        f"Replications: {list(reps)} (Hill 2011 / CEVAE, N=747).",
        "",
        "| Variant | n | √PEHE | std(√PEHE) | ε_ATE | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in agg.iterrows():
        lines.append(
            f"| {name} | {int(r['n'])} | "
            f"{r['mean_pehe']:.3f} | {r['std_pehe']:.3f} | "
            f"{r['mean_ate_err']:.3f} | {r['mean_rt']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **If Nyström RBF ≤ T-Learner** → assumption verified; the IHDP "
        "gap is purely basis, and RX-Learner's machinery is fine once the "
        "functional form is flexible enough.",
        "- **If richer bases close most but not all of the gap** → basis "
        "dominates but there is a residual efficiency gap.",
        "- **If no basis closes the gap** → the limitation is elsewhere "
        "(shrinkage, effective sample size, DR residual variance under high-"
        "dim basis).",
    ]
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")
    df.to_csv(RESULTS_DIR / "ihdp_basis_ablation_raw.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'ihdp_basis_ablation_raw.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replications", type=int, default=10)
    args = ap.parse_args()
    reps = list(range(1, args.replications + 1))
    df = _run(reps)
    agg = _summarise(df)
    print("\n── Summary ──")
    print(agg.to_string())
    _write_markdown(df, agg, reps)


if __name__ == "__main__":
    main()
