# Automated severity selection from tail-index diagnostic

Hill estimator on residuals of a quick S-Learner (top 10\%) gives α̂.
Map: α̂>5 → none; 3<α≤5 → mild; 2<α≤3 → moderate; α≤2 → severe.
3 seeds × 3 DGP regimes (clean, whale 5\%, whale 20\%).

| setup | n | mean α̂ | mode severity | bias | 95% coverage |
|---|---:|---:|---|---:|---:|
| clean | 3 | 4.79 | none | +0.051 | 1.00 |
| whale_20% | 3 | 6.07 | none | +6.487 | 1.00 |
| whale_5% | 3 | 1.23 | severe | +0.130 | 0.00 |