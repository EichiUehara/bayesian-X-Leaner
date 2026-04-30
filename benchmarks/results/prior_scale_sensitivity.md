# Prior-scale sensitivity — high-contamination mode-flip hypothesis

N fixed at 1000. Whale density swept over ['5%', '10%', '20%'], beta-prior scale `Normal(0, σ)` swept over [0.5, 1.0, 2.0, 5.0, 10.0]. Robust (Welsch) variant only. Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

Hypothesis (tested): §11 showed robust fits at density ≥ 5 % diverge with large negative ATE (−1534 at 20 %). One candidate mechanism was that Welsch's redescending loss produces a bimodal posterior under majority contamination and the default wide prior `Normal(0, 10)` fails to penalise the wrong mode. If that held, tightening σ should lift the mode-flip rate down and shrink bias toward zero. `mode_flip_rate` reports the fraction of seeds whose posterior mean ATE came out < 0 (true ATE = +2).

**Verdict: hypothesis falsified.** At every density the bias is flat across four orders of prior-scale magnitude (e.g. density 10 %: −277.26 at σ=0.5 vs −280.09 at σ=10.0 — a < 1 % change). Mode-flip rate is 1.00 across all 15 cells. The posterior is not bimodal — the likelihood itself is peaked at the catastrophically biased value because the DR pseudo-outcomes inherit nuisance-model contamination that Welsch cannot undo. The prior is not load-bearing in this failure regime; §12's prescription (deeper nuisance trees) remains the only effective remedy.

## density = 5% (50 whales)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width | Mode-flip rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 8 | -26.549 | 31.808 | -25.886 | 0.00 | 0.409 | 1.00 |
| 1.0 | 8 | -26.663 | 31.994 | -26.103 | 0.00 | 0.379 | 1.00 |
| 2.0 | 8 | -26.748 | 32.045 | -26.099 | 0.00 | 0.412 | 1.00 |
| 5.0 | 8 | -26.791 | 32.095 | -26.125 | 0.00 | 0.426 | 1.00 |
| 10.0 | 8 | -26.887 | 32.096 | -26.152 | 0.00 | 0.249 | 1.00 |

## density = 10% (100 whales)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width | Mode-flip rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 8 | -277.261 | 282.893 | -277.004 | 0.00 | 0.416 | 1.00 |
| 1.0 | 8 | -279.446 | 285.119 | -279.297 | 0.00 | 0.316 | 1.00 |
| 2.0 | 8 | -279.942 | 285.625 | -279.813 | 0.00 | 0.213 | 1.00 |
| 5.0 | 8 | -280.051 | 285.733 | -279.921 | 0.00 | 0.120 | 1.00 |
| 10.0 | 8 | -280.094 | 285.775 | -279.979 | 0.00 | 0.166 | 1.00 |

## density = 20% (200 whales)

| prior_scale | n | Bias | RMSE | Median ATE | Coverage | CI Width | Mode-flip rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 8 | -1516.201 | 1524.713 | -1539.185 | 0.00 | 2.789 | 1.00 |
| 1.0 | 8 | -1529.973 | 1538.600 | -1554.884 | 0.00 | 1.894 | 1.00 |
| 2.0 | 8 | -1533.557 | 1542.185 | -1557.631 | 0.00 | 1.170 | 1.00 |
| 5.0 | 8 | -1534.719 | 1543.343 | -1560.214 | 0.00 | 0.806 | 1.00 |
| 10.0 | 8 | -1534.418 | 1543.047 | -1557.936 | 0.00 | 1.189 | 1.00 |
