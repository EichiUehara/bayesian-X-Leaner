# Why Bayesian X-Learner

## The problem

Conditional Average Treatment Effect (CATE) estimation sits at the
intersection of two demands that almost no tool simultaneously meets:

1. **Heterogeneous effects.** "Does this treatment work on average?"
   is the wrong question when the treatment helps one subpopulation
   and hurts another. Practitioners need τ(x), not just τ. Meta-
   learners (S, T, X) address this; classical RCT analysis doesn't.

2. **Calibrated uncertainty on τ(x).** A point estimate of "the ad
   campaign lifts revenue by $3 per user in segment A" is not
   actionable unless you know whether the confidence band crosses
   zero — and whether the band itself is trustworthy. Most meta-
   learners produce a number; a few wrap bootstrap intervals around
   it; almost none produce a Bayesian posterior you can integrate
   over for decisions.

The gap is specific and not hypothetical. Here's where existing tools
sit:

| Tool | Heterogeneous τ(x) | Calibrated uncertainty | Handles real-world tails |
|---|:---:|:---:|:---:|
| Classical S/T/X-Learner | ✓ | ✗ | ✗ |
| EconML Causal Forest | ✓ | bootstrap (approximate) | ✗ |
| Causal BART | ✓ | ✓ (Bayesian) | ✗ (Gaussian likelihood) |
| TMLE / DML | partial | ✓ (frequentist) | ✗ |
| **Bayesian X-Learner** | **✓** | **✓ (full MCMC posterior)** | **✓ (Welsch likelihood)** |

Every row except the last gives up something a working practitioner
actually needs. Causal BART comes closest, but its Gaussian-only
likelihood fails when outcomes have real tails — and real outcome
data almost always has real tails.

## What this library is (and isn't)

**It is** a Bayesian X-Learner: X-Learner's imputed-treatment-effect
architecture, fit with doubly-robust pseudo-outcomes, and a full
MCMC posterior over the CATE surface τ(x) via NumPyro.

**Its defining design choice** is the Welsch redescending
pseudo-likelihood at the Bayesian layer. On truly Gaussian data this
would be pure overhead, but real outcome surfaces — medical responses,
revenue distributions, Hill's IHDP simulator — have genuine tails
that Gaussian regression systematically misfits. Welsch lets the
posterior pay attention to the bulk and downweight what doesn't fit,
without manual outlier flagging.

**It is not** a universal CATE library. It doesn't do continuous
treatments, it runs MCMC (seconds per fit, not milliseconds), and it
asks the user to specify a basis for τ(x). Those constraints are real
and documented.

## The clean-data story: a competitive Bayesian X-Learner

On Hill (2011)'s IHDP semi-synthetic benchmark — the field's standard
clean CATE testbed, with real covariates and known ground-truth τ —
the default configuration (`robust=True`, XGB-MSE nuisance) produces
**√PEHE = 0.562**, beating every baseline tested:

| Estimator | √PEHE | ε_ATE |
|---|---:|---:|
| **Bayesian X-Learner (default)** | **0.562** | 0.079 |
| S-Learner | 0.720 | 0.091 |
| T-Learner | 0.788 | 0.040 |
| X-Learner (std, no Bayesian layer) | 0.936 | 0.028 |
| EconML Causal Forest | 1.056 | 0.315 |

Not by a lot — but it wins cleanly, on a benchmark nobody designed
for our method, against established baselines. That's the floor
claim: "on standard clean data, you are not paying for the Bayesian
machinery."

The stability claims reinforce this: 100 % of runs converge at
R̂ < 1.05 with ESS > 200 across every DGP tested.

## The contamination story: a robust extension that earns its keep

Standard benchmarks are clean; production data often isn't. Revenue
has whales. Medical outcomes have rare catastrophic events. Sensor
streams have glitches. We built a contamination benchmark — the
"whale DGP," where 1-30 % of units have outcomes shifted by 10× the
clean scale — to ask where the clean-data default breaks and what
brings it back.

What we found:

| Whale density | Default (XGB-MSE nuisance) | Extension (CatBoost-Huber nuisance) |
|---:|---:|---:|
| 0 % (clean IHDP) | √PEHE **0.56** (best) | √PEHE 1.80 (3× penalty) |
| 5 % | RMSE ≈ 32 (breaking) | RMSE **0.13** |
| 20 % | RMSE ≈ 1543 (catastrophic) | RMSE **0.06**, coverage **1.00** |
| 30 % | catastrophic | RMSE 2.5 (breakdown begins) |
| 50 % | catastrophic | fails (majority-contaminated) |

Two complementary findings. First, the clean-data default *does*
break under contamination — not gracefully. Second, a one-line
configuration change (`nuisance_method="catboost"` with Huber loss)
recovers performance up to ~20-25 % contamination, with properly
calibrated credible intervals. Beyond 30 %, no robust method survives
— that's Huber's theoretical breakdown point, not a library limit.

**The extension is not the default.** Huber's (1964) efficiency-
robustness theorem is unambiguous: bounded-influence losses cost
efficiency at clean data. Our IHDP penalty (3× √PEHE) is this cost
made concrete. The library ships the clean-data configuration as
default and documents the extension for users whose data actually
needs it.

## What this library matters for (use cases grounded in evidence)

- **Clinical trials with rare severe outcomes.** Mortality events in
  a treatment arm, adverse reactions, rare efficacy spikes. Welsch
  likelihood handles the tail structure; the Bayesian posterior
  supports go/no-go decisions where bootstrap CIs aren't good enough.
- **Marketing uplift with revenue whales.** Long-tail customer value.
  At up to 20 % whale density, the CatBoost-Huber extension delivers
  point accuracy *and* calibrated uncertainty — neither causal
  forests nor TMLE provide both together.
- **Policy evaluation with structured τ(x) hypotheses.** When you
  suspect τ depends on specific features (age, dose, covariate
  interaction), the `X_infer` basis encodes the hypothesis directly
  and returns a posterior over coefficients, not an opaque surface.
- **Research contexts where the full posterior matters.** Integrating
  over τ(x) to compute P(τ(x) > threshold), comparing posteriors
  across populations, propagating uncertainty to downstream
  decisions. Point estimates plus bootstrap give you the first;
  Bayesian X-Learner gives you all three with the same fit.

## Where the library is honest about its boundaries

Six boundaries, each mapped empirically:

| Boundary | Limit | Source |
|---|---|---|
| Contamination ceiling | 20-25 % whale density (CatBoost-Huber); nothing robust works at ≥ 50 % | [whale_density_catboost_huber.md](benchmarks/results/whale_density_catboost_huber.md) |
| Clean-data efficiency of robust nuisance | Huber(δ=0.5) costs 3× √PEHE on clean data; loosen to δ=1.345 or switch to XGB-MSE | [EXTENSIONS.md §17](benchmarks/results/EXTENSIONS.md) |
| Basis sensitivity | User must supply τ(x) functional form; wrong basis → PEHE degradation | [nonlinear_cate.md](benchmarks/results/nonlinear_cate.md) |
| Compute | 5-20 s per fit (MCMC); not millisecond-latency territory | all |
| Treatment type | Binary W ∈ {0, 1}; continuous dose un-validated | — |
| Sample size | Verified 200 ≤ N ≤ 5000; very-large-N extrapolated | [sample_size_scaling_catboost_huber.md](benchmarks/results/sample_size_scaling_catboost_huber.md) |

Outside these regimes, use a different tool. Inside them, the library
is validated with seed-level reproducibility.

## Why the story is trustable

Three independent pillars support every claim above:

1. **Theory.** Huber (1964) asymptotic relative efficiency and the
   minimax-δ relation are theorems, numerically verified against
   Huber & Ronchetti (2009) Table 4.1. The efficiency-robustness
   tradeoff is derivable before any experiment.

2. **Synthetic stress tests.** 16+ benchmark experiments
   ([EXTENSIONS.md](benchmarks/results/EXTENSIONS.md)) probe each
   axis independently: contamination density, sample-size scaling,
   overlap, nuisance-model choice, τ(x) basis, prior sensitivity.
   Designed to falsify as well as confirm.

3. **External benchmark.** IHDP (Hill 2011) is the field's standard
   clean CATE benchmark. We didn't design it; we reported against it.
   Results are reproducible via [REPRODUCE.md](REPRODUCE.md).

When the three pillars agreed, the claim went in the headline table
([README.md](README.md)). When they disagreed, the disagreement
itself became a finding (§17: the same Huber-nuisance that wins whale
loses IHDP — now documented, not hidden).

## The one-sentence summary

**Bayesian X-Learner is the tool you want when you need calibrated
posteriors over τ(x) on real outcome data, with a one-flag extension
for heavy-tailed contamination that earns its keep up to ~20-25 %
whale density.** Outside that envelope, we tell you to use something
else — and we tell you which.
