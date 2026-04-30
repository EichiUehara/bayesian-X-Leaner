"""Re-run Lalonde NSW with normalize_y_for_nuisance=True.

Verifies (or refutes) the §5.9 claim that MAD pre-standardisation
recovers the canonical positive Lalonde estimate (~$1700) under
severity=severe.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sert_xlearner.targeted_bayesian_xlearner import TargetedBayesianXLearner

import urllib.request
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
path = DATA / "nsw_dw.dta"
if not path.exists():
    urllib.request.urlretrieve(URL, path)
df = pd.read_stata(path)

Y = df["re78"].values.astype(float)
W = df["treat"].values.astype(int)
X = df[["age", "education", "black", "hispanic", "married",
        "nodegree", "re74", "re75"]].values.astype(float)
print(f"Lalonde: N={len(X)}, treated={W.sum()}, "
      f"E[Y]={Y.mean():.0f}, max(Y)={Y.max():.0f}")

rows = []
configs = [
    ("none",   False),
    ("none",   True),
    ("severe", False),
    ("severe", True),
]
for severity, mad_norm in configs:
    kwargs = dict(
        n_splits=2, num_warmup=400, num_samples=800, num_chains=2,
        c_whale=1.34, mad_rescale=False, random_state=0,
        robust=True, use_student_t=True,
        normalize_y_for_nuisance=mad_norm,
    )
    if severity == "none":
        kwargs["nuisance_method"] = "xgboost"
        kwargs["outcome_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
        kwargs["propensity_model_params"] = {"max_depth": 4, "n_estimators": 150, "verbosity": 0}
    else:
        kwargs["contamination_severity"] = "severe"
    model = TargetedBayesianXLearner(**kwargs)
    model.fit(X, Y, W, X_infer=np.ones((len(X), 1)))
    beta = np.asarray(model.mcmc_samples["beta"]).squeeze()
    s = getattr(model, "_y_scale", 1.0)
    beta_dollars = beta * s  # restore dollar scale if normalised
    ate = float(np.mean(beta_dollars))
    lo, hi = np.percentile(beta_dollars, [2.5, 97.5])
    rows.append({"severity": severity, "normalize_y": mad_norm,
                 "ate": ate, "lo": float(lo), "hi": float(hi),
                 "ci_width": float(hi - lo), "y_scale": s})
    print(f"sev={severity:6s} norm={mad_norm!s:5s}  "
          f"ate=${ate:+.1f}  CI=[${lo:+.1f}, ${hi:+.1f}]  y_scale={s:.2f}")

out = Path(__file__).parent / "results" / "lalonde_madnorm_raw.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nWrote {out}")
