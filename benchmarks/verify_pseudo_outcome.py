"""
Empirical verification of the three claims about why standard RX-Learner
underperforms the robust variant:

  Claim 1 (structural).  The X-Learner pseudo-outcomes D1, D0 are heavy-tailed
                         EVEN WHEN Y is Gaussian.  Source: the DR formula
                         divides residuals by π̂ (or 1−π̂), which is unbounded
                         as π̂ → 0 or 1.

  Claim 2 (mechanism).   The Gaussian log-likelihood penalises residuals by
                         r², so a small number of tail pseudo-outcomes
                         dominate the posterior — dragging the ATE estimate.

  Claim 3 (amplification). Already verified by stability_report.md
                         (MCMC-seed std = 24 under whale · std).  No further
                         test needed here.

Outputs:
    benchmarks/results/pseudo_outcome_diagnostics.md
    benchmarks/results/figures/pseudo_outcome_tails.png
    benchmarks/results/figures/loss_influence.png
"""

from __future__ import annotations
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from benchmarks.dgps import standard_dgp, whale_dgp
from sert_xlearner.core.orthogonalization import impute_and_debias


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def _cross_fit_nuisance(X, Y, W, seed=42):
    """Two-fold cross-fit: returns mu0, mu1, pi for every row (out-of-fold)."""
    N = len(X)
    mu0 = np.zeros(N)
    mu1 = np.zeros(N)
    pi = np.zeros(N)
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        Xtr, Ytr, Wtr = X[tr], Y[tr], W[tr]
        m0 = HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=seed)
        m0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
        m1 = HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=seed)
        m1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
        p = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=seed)
        p.fit(Xtr, Wtr)
        mu0[te] = m0.predict(X[te])
        mu1[te] = m1.predict(X[te])
        pi[te] = np.clip(p.predict_proba(X[te])[:, 1], 0.01, 0.99)
    return mu0, mu1, pi


def _moments(x, name):
    x = np.asarray(x)
    return {
        "name":      name,
        "n":         len(x),
        "mean":      float(np.mean(x)),
        "std":       float(np.std(x, ddof=1)),
        "skew":      float(stats.skew(x)),
        "kurtosis":  float(stats.kurtosis(x, fisher=True)),  # excess kurtosis
        "max_abs_z": float(np.max(np.abs((x - np.mean(x)) / np.std(x, ddof=1)))),
        "p_jb":      float(stats.jarque_bera(x).pvalue),
        "p99_abs":   float(np.percentile(np.abs(x - np.mean(x)), 99)),
    }


def _format_row(m):
    return (f"| {m['name']} | {m['n']} | {m['mean']:+.3f} | {m['std']:.3f} | "
            f"{m['skew']:+.2f} | {m['kurtosis']:+.2f} | "
            f"{m['max_abs_z']:.1f}σ | {m['p_jb']:.1e} | {m['p99_abs']:.2f} |")


def _qq_vs_normal(ax, data, title, colour):
    stats.probplot(data, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(colour)
    ax.get_lines()[0].set_markeredgecolor(colour)
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color("red")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def _plot_tails(records, figpath):
    n = len(records)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, data, colour) in enumerate(records):
        ax_hist, ax_qq = axes[0, i], axes[1, i]

        ax_hist.hist(data, bins=50, color=colour, alpha=0.75, edgecolor="black",
                     density=True)
        xs = np.linspace(data.min(), data.max(), 400)
        ax_hist.plot(xs, stats.norm.pdf(xs, np.mean(data), np.std(data)),
                     color="red", linestyle="--", linewidth=1.5, label="Gaussian fit")
        ax_hist.set_title(f"{name}\nhistogram vs Gaussian")
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)

        _qq_vs_normal(ax_qq, data, f"{name}\nQ-Q vs Normal", colour)

    fig.suptitle("Pseudo-outcome distribution — are residuals Gaussian?", fontsize=13)
    fig.tight_layout()
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")


def _plot_loss_influence(D1_std, figpath):
    """
    Claim 2: Gaussian likelihood (L2) vs Welsch redescender.

    Show: contribution to log-likelihood as a function of residual size.
          For L2, a 5σ residual contributes 25×; Welsch caps at ≈1.
    """
    c = 1.34 * np.std(D1_std, ddof=1) / 0.6745
    r = np.linspace(-6, 6, 400) * np.std(D1_std, ddof=1)
    r_scaled = r / max(c, 1e-6)

    l2 = r ** 2
    welsch = c ** 2 * (1 - np.exp(-r_scaled ** 2 / 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(r, l2, color="#1f77b4", label="Gaussian  L(r) = r²", linewidth=2)
    axes[0].plot(r, welsch, color="#d62728", label="Welsch  L(r) = c²(1 − e^{−(r/c)²/2})", linewidth=2)
    axes[0].set_xlabel("residual r (data units)")
    axes[0].set_ylabel("loss contribution")
    axes[0].set_title("Loss function (per-observation penalty)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Influence function (derivative) — the real diagnostic
    psi_l2 = 2 * r
    psi_welsch = r * np.exp(-r_scaled ** 2 / 2)
    axes[1].plot(r, psi_l2, color="#1f77b4", label="Gaussian  ψ(r) = 2r", linewidth=2)
    axes[1].plot(r, psi_welsch, color="#d62728", label="Welsch  ψ(r) = r·e^{−(r/c)²/2}", linewidth=2)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("residual r (data units)")
    axes[1].set_ylabel("influence ψ(r) = ∂L/∂r")
    axes[1].set_title("Influence function  (score contribution)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Why the Gaussian likelihood is sensitive to pseudo-outcome tails",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")


def run(dgp_name, dgp_fn, seed=42):
    print(f"\n── {dgp_name} DGP (seed={seed}) ──")
    X, Y, W, tau = dgp_fn(seed=seed)
    print(f"  N={len(X)}  true ATE={tau}")

    mu0, mu1, pi = _cross_fit_nuisance(X, Y, W, seed=seed)
    _, _, D1, D0, _, _ = impute_and_debias(Y, W, mu0, mu1, pi)

    # Compare: raw Y residual vs pseudo-outcome residual
    Y_res_treated = Y[W == 1] - mu1[W == 1]       # OOF residual, in-group
    Y_res_control = Y[W == 0] - mu0[W == 0]
    D1_centered = D1 - np.mean(D1)                 # demean for a fair tail comparison
    D0_centered = D0 - np.mean(D0)

    return {
        "dgp": dgp_name,
        "tau": tau,
        "raw_Y_res":       np.concatenate([Y_res_treated, Y_res_control]),
        "pseudo_D1":       D1,
        "pseudo_D0":       D0,
        "pseudo_D1_c":     D1_centered,
        "pseudo_D0_c":     D0_centered,
        "pi":              pi,
    }


def main():
    out = [
        run("standard", standard_dgp, seed=0),
        run("whale",    whale_dgp,    seed=0),
    ]

    # ── Diagnostics table ─────────────────────────────────────────────────
    moments = []
    for r in out:
        moments.append(_moments(r["raw_Y_res"],   f"{r['dgp']} · raw Y − μ̂"))
        moments.append(_moments(r["pseudo_D1"],   f"{r['dgp']} · D₁ (DR pseudo)"))
        moments.append(_moments(r["pseudo_D0"],   f"{r['dgp']} · D₀ (DR pseudo)"))

    lines = [
        "# Pseudo-outcome distribution — empirical verification",
        "",
        "Claim 1: DR-X-Learner pseudo-outcomes are heavy-tailed even when Y is Gaussian.",
        "",
        "Under a Gaussian null hypothesis: excess kurtosis ≈ 0, Jarque-Bera p ≫ 0.05.",
        "Heavy tails → positive excess kurtosis and very small JB p-value.",
        "",
        "| Variable | n | mean | std | skew | excess kurt | max \\|z\\| | JB p-value | P99 \\|r\\| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in moments:
        lines.append(_format_row(m))

    # Propensity range (explains where the mass comes from)
    lines += ["", "## Propensity range (source of 1/π̂ blow-up)", ""]
    for r in out:
        lines.append(f"- **{r['dgp']}**: π̂ ∈ [{r['pi'].min():.4f}, {r['pi'].max():.4f}], "
                     f"min(π̂, 1−π̂) = {min(r['pi'].min(), 1 - r['pi'].max()):.4f}")

    # Interpretation
    lines += [
        "",
        "## Interpretation",
        "",
        "- If the pseudo-outcome rows have **much larger excess kurtosis** than the raw "
        "`Y − μ̂` rows, claim 1 is confirmed: the DR division by π̂ and (1−π̂) "
        "generates tails that the raw outcome didn't have.",
        "- If `max |z|` is ≥ 5 on a standard DGP where Y itself is Gaussian, the Gaussian "
        "likelihood will pay ≥ 25× per observation at the tail — claim 2's mechanism.",
        "- JB p ≪ 0.05 rejects Gaussianity at standard significance. A Gaussian "
        "likelihood assumed by non-robust MCMC is the wrong model.",
        "",
        "## Why this causes a *systematic* bias (not just variance)",
        "",
        "The DR numerator `(Y − μ̂)/π̂` is **asymmetrically distributed** when π̂ is "
        "skewed — i.e., when treated and control regions differ. Under `standard_dgp` "
        "the propensity is sigmoid-structured, so the rare `π̂ ≈ 0.02` cases land "
        "disproportionately on control units with negative residuals (or vice versa), "
        "producing the observed **−0.42 mean bias** in the non-robust RX-Learner.",
        "",
        "## Figures",
        "",
        f"- `figures/pseudo_outcome_tails.png` — histogram + Q-Q plot vs Normal",
        f"- `figures/loss_influence.png` — Gaussian L2 vs Welsch influence function",
    ]

    path = RESULTS_DIR / "pseudo_outcome_diagnostics.md"
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")

    # ── Figures ───────────────────────────────────────────────────────────
    records = []
    for r in out:
        colour = "#1f77b4" if r["dgp"] == "standard" else "#d62728"
        records.append((f"{r['dgp']} · raw Y − μ̂",    r["raw_Y_res"],  colour))
        records.append((f"{r['dgp']} · D₁ (DR)",      r["pseudo_D1"],  colour))
    _plot_tails(records, FIG_DIR / "pseudo_outcome_tails.png")

    _plot_loss_influence(out[0]["pseudo_D1"], FIG_DIR / "loss_influence.png")


if __name__ == "__main__":
    main()
