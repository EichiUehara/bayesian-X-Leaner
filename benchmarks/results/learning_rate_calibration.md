# Generalised-Bayes learning-rate calibration for the Welsch posterior

Bootstrap-calibrated η-selector (Lyddon-Holmes-Walker style).
Whale DGP, N = 1000, true ATE = 2.0.
η ∈ [0.5, 0.75, 1.0, 1.5, 2.0]; each candidate refits the posterior with c → c/√η.
LLB target variance from 50 bootstrap replicates of Huber-DR.

| density | n | mean η̂ | ATE | 95% coverage | CI width |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 3 | 0.50 | +2.026 | 1.00 | 0.184 |
| 0.05 | 3 | 2.00 | +4.734 | 1.00 | 7.653 |
| 0.20 | 3 | 1.83 | +4.533 | 1.00 | 38.665 |