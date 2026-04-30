# Hillstrom email marketing — real heavy-tailed RCT

Dataset: Kevin Hillstrom MineThatData (2008). Two arms: No E-Mail (control) vs Mens E-Mail (treatment). N=42613, treated=21307.
Outcome: 2-week spend ($). Heavy right tail; mean = 1.0377, 99th pct = 0.00.
Ground-truth ATE is identifiable via random assignment but no per-unit τ(x).

| Estimator | ATE | 95% CI | width | runtime |
|---|---:|---|---:|---:|
| Naive (difference of means) | +0.7698 | [+0.4851, +1.0545] | 0.5694 | 0.0s |
| Huber-DR (bootstrap CI) | +0.0204 | [+0.0069, +0.0459] | 0.0390 | 1556.8s |
| RX-Welsch (severity=none) | -0.0026 | [-0.0292, +0.0201] | 0.0493 | 2.5s |
| RX-Welsch (severity=severe) | -0.0001 | [-0.0092, +0.0092] | 0.0184 | 2.7s |