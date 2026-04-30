# Benchmark datasets

Three publicly-available datasets used by the experiments. We commit
small CSV/Stata copies for reproducibility; the original sources are
authoritative.

## IHDP (`ihdp_1.csv` … `ihdp_10.csv`)

Hill's semi-synthetic Infant Health and Development Program benchmark
with simulated outcomes from response-surface B (10 replications).

- Source: <https://www.fredjo.com/> (Hill 2011 release, redistributed
  by Johansson et al. via the CEVAE / CFR codebases)
- Citation: Hill, J. L. (2011). *Bayesian nonparametric modeling for
  causal inference.* Journal of Computational and Graphical
  Statistics, 20(1), 217–240.
- Used by: `benchmarks/run_round*` scripts; reported in §5 (IHDP) of
  the paper.

## Hillstrom email-marketing RCT (`hillstrom.csv`, ~4 MB)

Anonymised customer-level RCT from the MineThatData challenge:
N = 64,000 (filtered to 42,613 customers with valid covariates),
binary treatment (email vs no email), continuous spend outcome with
heavy positive tail.

- Source: <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv>
- Citation: Hillstrom, K. (2008). *The MineThatData e-mail analytics
  and data mining challenge.*
- Used by: `benchmarks/run_hillstrom.py`,
  `benchmarks/run_hillstrom_true_holdout.py`; reported in §5.9 of the
  paper.

## Lalonde NSW (`nsw_dw.dta`)

Dehejia–Wahba subsample of LaLonde's National Supported Work
demonstration (N = 445), binary treatment (job training), 1978
earnings outcome (heavy-tailed, dollar-scale).

- Source: <https://users.nber.org/~rdehejia/data/nsw_dw.dta>
- Citations: LaLonde, R. (1986). *Evaluating the econometric
  evaluations of training programs with experimental data.* American
  Economic Review, 76(4), 604–620; Dehejia, R. H. & Wahba, S. (2002).
  *Propensity score-matching methods for nonexperimental causal
  studies.* Review of Economics and Statistics, 84(1), 151–161.
- Used by: `benchmarks/run_round10_experiments.py`,
  `benchmarks/run_lalonde_madnorm.py`; reported in §5.9 of the paper.

## Re-downloading

The Hillstrom and Lalonde files are auto-downloaded by the relevant
benchmark scripts if absent (`urllib.request.urlretrieve`). If you
delete the local copies, the next run will fetch them.
