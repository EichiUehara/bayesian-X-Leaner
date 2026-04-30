import numpy as np
from sklearn.base import BaseEstimator
from .models.nuisance import NuisanceEstimator, CATBOOST_AVAILABLE
from .core.orthogonalization import impute_and_debias
from .inference.bayesian import BayesianMCMC


_CATBOOST_DEFAULT_OUTCOME = {"depth": 4, "iterations": 150,
                              "loss_function": "Huber:delta=0.5"}
_CATBOOST_DEFAULT_PROPENSITY = {"depth": 4, "iterations": 150}


# Huber (1964) minimax-optimal delta as a function of expected
# contamination rate epsilon. Derived numerically by solving
#     phi(delta)/delta − (1 − Phi(delta)) = epsilon / (2*(1 − epsilon))
# See EXTENSIONS.md §17.1 for the table and references.
_CONTAMINATION_PRESETS = {
    "none":     {"nuisance_method": "xgboost",  "huber_delta": None},
    "mild":     {"nuisance_method": "catboost", "huber_delta": 1.345},
    "moderate": {"nuisance_method": "catboost", "huber_delta": 1.0},
    "severe":   {"nuisance_method": "catboost", "huber_delta": 0.5},
}


def _resolve_contamination_severity(severity):
    """Map a severity string to (nuisance_method, outcome_params).

    severity ∈ {"none", "mild", "moderate", "severe"} maps to Huber's
    1964 minimax-optimal delta for expected contamination rates
    {0 %, ~5 %, ~10 %, ~40 %} respectively. See EXTENSIONS.md §17.1.
    """
    if severity not in _CONTAMINATION_PRESETS:
        raise ValueError(
            f"contamination_severity={severity!r} not recognised. "
            f"Expected one of {list(_CONTAMINATION_PRESETS)}."
        )
    preset = _CONTAMINATION_PRESETS[severity]
    if preset["huber_delta"] is None:
        return "xgboost", None
    outcome_params = {
        "depth": 4,
        "iterations": 150,
        "loss_function": f"Huber:delta={preset['huber_delta']}",
    }
    return preset["nuisance_method"], outcome_params


class TargetedBayesianXLearner(BaseEstimator):
    """Robust Bayesian X-Learner for CATE estimation.

    Defaults (as of the §16 follow-up in benchmarks/results/EXTENSIONS.md):
        nuisance_method='catboost', depth=4, iterations=150,
        loss_function='Huber:delta=0.5' for outcomes. This is
        the configuration with bias −0.018 and coverage 1.00 at 20 %
        whale density. CatBoost is an optional dependency; the
        constructor silently falls back to 'xgboost' with bare
        (empty) params if CatBoost isn't importable. Users who pass
        `outcome_model_params` / `propensity_model_params` explicitly
        override the defaults completely.

    mad_rescale: bool, default True
        When robust=True, rescales the Welsch tuning constant c_whale
        by MAD(pseudo-outcomes)/0.6745 so c tracks the noise scale
        when Y is in engineering units (e.g. dollars). §14 shows this
        branch is catastrophic when paired with a non-robust nuisance
        outcome learner (e.g. nuisance_method='xgboost' with MSE loss)
        on contaminated data: MAD itself becomes contaminated,
        inflating effective c from 1.34 to ~3670 and defeating Welsch
        clipping. Set mad_rescale=False when combining robust=True
        with the xgboost fallback on contaminated data. With the §16
        default (CatBoost + Huber), pseudo-outcomes stay clean, MAD
        reflects the true noise scale, and this flag is a no-op.

    contamination_severity: {"none", "mild", "moderate", "severe"}, optional
        High-level knob that maps to Huber's (1964) minimax-optimal
        delta for an assumed contamination rate. See EXTENSIONS.md
        §17.1 for the theoretical derivation. Maps to:

        =========  =====================  =========================================
        severity   nuisance loss           corresponds to ε (expected contamination)
        =========  =====================  =========================================
        none       XGBoost MSE             0 % (truly clean data)
        mild       CatBoost Huber(δ=1.345) ~5 % (canonical Huber 1964)
        moderate   CatBoost Huber(δ=1.0)   ~10 %
        severe     CatBoost Huber(δ=0.5)   ~40 % (whale DGP; library benchmark target)
        =========  =====================  =========================================

        If outcome_model_params is also passed, it overrides severity.
        Leaving severity as None uses the raw nuisance_method /
        outcome_model_params arguments directly.
    """

    def __init__(self,
                 outcome_model_params=None,
                 propensity_model_params=None,
                 n_splits=2,
                 num_warmup=1000,
                 num_samples=2000,
                 num_chains=2,
                 random_state=42,
                 prior_scale=10.0,
                 robust=True,
                 c_whale=1.34,
                 tail_threshold=None,
                 tail_alpha=None,
                 use_overlap=False,
                 use_student_t=True,
                 nuisance_method='catboost',
                 mad_rescale=True,
                 contamination_severity=None,
                 normalize_y_for_nuisance=False):

        # contamination_severity overrides nuisance_method + outcome_params
        # when the caller hasn't explicitly set them. This is the
        # principled API surface for Huber's minimax-delta prescription —
        # users pick a severity, the library picks the delta.
        if contamination_severity is not None:
            resolved_nuisance, resolved_params = _resolve_contamination_severity(
                contamination_severity)
            if outcome_model_params is None:
                nuisance_method = resolved_nuisance
                if resolved_params is not None:
                    outcome_model_params = resolved_params

        if nuisance_method == 'catboost' and not CATBOOST_AVAILABLE:
            nuisance_method = 'xgboost'

        if nuisance_method == 'catboost':
            self.outcome_model_params = (dict(_CATBOOST_DEFAULT_OUTCOME)
                                          if not outcome_model_params
                                          else outcome_model_params)
            self.propensity_model_params = (dict(_CATBOOST_DEFAULT_PROPENSITY)
                                             if not propensity_model_params
                                             else propensity_model_params)
        else:
            self.outcome_model_params = outcome_model_params or {}
            self.propensity_model_params = propensity_model_params or {}
        self.n_splits = n_splits
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.random_state = random_state
        self.prior_scale = prior_scale
        self.robust = robust
        self.c_whale = c_whale
        self.tail_threshold = tail_threshold
        self.tail_alpha = tail_alpha
        self.use_overlap = use_overlap
        self.use_student_t = use_student_t
        self.nuisance_method = nuisance_method
        self.mad_rescale = mad_rescale
        self.normalize_y_for_nuisance = normalize_y_for_nuisance
        self._y_scale = 1.0  # filled in fit() if normalize_y_for_nuisance

        
        self.nuisance_estimator = NuisanceEstimator(
            self.outcome_model_params,
            self.propensity_model_params,
            n_splits=self.n_splits,
            random_state=self.random_state,
            method=self.nuisance_method,
        )
        self.bayesian_mcmc = BayesianMCMC(
            num_warmup=self.num_warmup, 
            num_samples=self.num_samples, 
            num_chains=self.num_chains,
            random_seed=self.random_state,
            prior_scale=self.prior_scale,
            robust=self.robust,
            c_whale=self.c_whale,
            use_student_t=self.use_student_t
        )

    def fit(self, X, Y, W, X_infer=None):
        """
        Fit the robust Bayesian X-Learner.
        X: Covariates (N, P) for nuisance models.
        Y: Outcomes (N,)
        W: Treatment indicator (N,) {0, 1}
        X_infer: Target low-dimensional covariates (N, P_target) for MCMC (optional).
                 If None, an intercept-only model is used for CATE.

        NOTE ON TAIL PARAMETERS:
        If using robust=True, the parameters `tail_threshold` and `tail_alpha` 
        (the tail index) should ideally be estimated from the observed moderate-frequency 
        data prior to running the RX-Learner. Typically, this is done using a 
        Hill Estimator or a Peaks-Over-Threshold (POT) model to guarantee tail stability.
        """
        # Optional MAD-rescaling of Y so the Huber-delta is in standardised units.
        # The scale is reapplied to the posterior in predict().
        if self.normalize_y_for_nuisance:
            mad_y = np.median(np.abs(Y - np.median(Y)))
            self._y_scale = float(mad_y / 0.6745) if mad_y > 1e-6 else 1.0
            Y = Y / self._y_scale
        else:
            self._y_scale = 1.0

        # PHASE 1: Nuisance Quarantine (Cross-Fitting)
        out_pred_mu0, out_pred_mu1, out_pred_pi = self.nuisance_estimator.fit_predict(X, Y, W)

        # Diagnostic: warn if propensities are near boundary (likely PD failure of I).
        pi_min = float(np.min(out_pred_pi)); pi_max = float(np.max(out_pred_pi))
        if pi_min < 0.05 or pi_max > 0.95:
            import warnings
            warnings.warn(
                f"Estimated propensities span [{pi_min:.3f}, {pi_max:.3f}]; "
                f"posterior calibration may degrade (small Hessian eigenvalues "
                f"in the Welsch information matrix). Consider use_overlap=True "
                f"or pre-trimming.", UserWarning, stacklevel=2)
            # Auto-fallback: if extreme overlap and use_overlap not already on,
            # silently enable it on this fit. The user's explicit choice wins.
            if not self.use_overlap and (pi_min < 0.02 or pi_max > 0.98):
                warnings.warn(
                    "Extreme propensities (<0.02 or >0.98) detected; "
                    "auto-enabling use_overlap=True for this fit. "
                    "Pass use_overlap=True at construction to silence "
                    "this fallback.", UserWarning, stacklevel=2)
                self._auto_use_overlap = True
            else:
                self._auto_use_overlap = False
        else:
            self._auto_use_overlap = False

        # PHASE 2: X-Learner Imputation & Targeted Debiasing
        effective_use_overlap = self.use_overlap or getattr(self, "_auto_use_overlap", False)
        treated_mask, control_mask, D1, D0, W_D1, W_D0 = impute_and_debias(
            Y, W, out_pred_mu0, out_pred_mu1, out_pred_pi,
            robust=self.robust, tail_threshold=self.tail_threshold,
            tail_alpha=self.tail_alpha, use_overlap=effective_use_overlap
        )
        
        if self.robust and self.mad_rescale:
            # Rescale c_whale by MAD(residuals)/0.6745 so Welsch tracks the noise scale.
            # §14: harmful under contamination with a non-robust nuisance learner.
            all_residuals = np.concatenate([D1, D0])
            mad = np.median(np.abs(all_residuals - np.median(all_residuals)))
            mad_scaled = mad / 0.6745 if mad > 1e-6 else 1.0
            self.bayesian_mcmc.c_whale = self.c_whale * mad_scaled
        elif self.robust:
            self.bayesian_mcmc.c_whale = self.c_whale
            
        self.X_infer_was_none = (X_infer is None)
        if self.X_infer_was_none:
            # Default to intercept only model for MCMC to prevent high-dim explosion
            X_infer = np.ones((X.shape[0], 1))

        X_D1 = X_infer[treated_mask]
        X_D0 = X_infer[control_mask]

        # PHASE 3: The Targeted Bayesian Update (MCMC)
        self.mcmc_samples = self.bayesian_mcmc.sample_posterior(
            X_D1, X_D0, D1, D0, W_D1, W_D0
        )
        return self

    def sample_posterior(self):
        """ Returns the posterior MCMC samples. """
        return self.mcmc_samples

    def predict(self, X_new_infer=None):
        """
        Returns posterior predictive CATE distribution (mean and 95% CI).
        """
        if getattr(self, 'X_infer_was_none', False) or X_new_infer is None:
             X_new_infer = np.ones((1, 1))

        beta_samples = self.mcmc_samples["beta"]
        # Expected CATE for each sample
        cate_samples = np.dot(X_new_infer, beta_samples.T)
        
        mean_cate = np.mean(cate_samples, axis=1)
        ci_lower, ci_upper = np.percentile(cate_samples, [2.5, 97.5], axis=1)
        # If Y was MAD-rescaled at fit time, restore the original outcome scale.
        s = getattr(self, "_y_scale", 1.0)
        if s != 1.0:
            mean_cate = mean_cate * s
            ci_lower = ci_lower * s
            ci_upper = ci_upper * s
        return mean_cate, ci_lower, ci_upper
        
    def predict_cate(self, X_new_infer=None):
        return self.predict(X_new_infer)

