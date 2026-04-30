# Extended IHDP — cheap baselines on 25 replications

5 reps was the previous benchmark cadence (CEVAE preprocessing convention).
BART/BCF rows omitted (each rep takes ~10 min); other baselines run on 10 reps.

| Estimator | n | √PEHE | std(√PEHE) | ε_ATE |
|---|---:|---:|---:|---:|
| T-Learner | 10 | 1.373 | 1.634 | 0.110 |
| RX-Learner (robust) | 10 | 1.951 | 3.584 | 0.252 |
| Huber-DR (point) | 10 | 1.972 | 3.647 | 0.362 |
| S-Learner | 10 | 2.117 | 3.801 | 0.189 |
| X-Learner (std) | 10 | 2.127 | 3.231 | 0.207 |
| EconML Forest | 10 | 3.060 | 5.240 | 0.758 |

## Welch's t-test vs RX-Learner (robust)

| Estimator | mean diff | t | p-value |
|---|---:|---:|---:|
| S-Learner | +0.166 | +0.10 | 0.921 |
| T-Learner | -0.579 | -0.46 | 0.650 |
| X-Learner (std) | +0.176 | +0.12 | 0.909 |
| EconML Forest | +1.109 | +0.55 | 0.588 |
| Huber-DR (point) | +0.020 | +0.01 | 0.990 |