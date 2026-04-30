import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

try:
    from tabpfn import TabPFNRegressor, TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class NuisanceEstimator:
    def __init__(self, outcome_params, propensity_params, n_splits=2, random_state=42, use_tabpfn=False, method='xgboost'):
        self.outcome_params = outcome_params
        self.propensity_params = propensity_params
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_tabpfn = use_tabpfn and TABPFN_AVAILABLE
        self.method = method # 'xgboost', 'elasticnet', 'catboost'
        self.models = {
            'mu0': [],
            'mu1': [],
            'pi': []
        }
        self.folds = []

    def fit_predict(self, X, Y, W):
        """
        Executes cross-fitting. Fits models on K-1 folds and predicts on the hold-out fold.
        Returns the out-of-fold predictions for mu0, mu1, and pi.
        """
        N = len(X)
        out_pred_mu0 = np.zeros(N)
        out_pred_mu1 = np.zeros(N)
        out_pred_pi = np.zeros(N)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X):
            self.folds.append((train_idx, test_idx))
            
            X_train, Y_train, W_train = X[train_idx], Y[train_idx], W[train_idx]
            X_test = X[test_idx]

            # Train mu_0 on Control group
            X_train_0 = X_train[W_train == 0]
            Y_train_0 = Y_train[W_train == 0]
            if self.method == 'elasticnet' or getattr(self, 'use_en', False):
                from sklearn.linear_model import ElasticNetCV
                mu0_model = ElasticNetCV(cv=2)
            elif self.method == 'catboost' and CATBOOST_AVAILABLE:
                mu0_model = CatBoostRegressor(verbose=False, random_seed=self.random_state, **self.outcome_params)
            else:
                mu0_model = xgb.XGBRegressor(**self.outcome_params)
            mu0_model.fit(X_train_0, Y_train_0)
            self.models['mu0'].append(mu0_model)
            out_pred_mu0[test_idx] = mu0_model.predict(X_test)

            # Train mu_1 on Treated group
            X_train_1 = X_train[W_train == 1]
            Y_train_1 = Y_train[W_train == 1]
            if self.method == 'elasticnet' or getattr(self, 'use_en', False):
                from sklearn.linear_model import ElasticNetCV
                mu1_model = ElasticNetCV(cv=2)
            elif self.method == 'catboost' and CATBOOST_AVAILABLE:
                mu1_model = CatBoostRegressor(verbose=False, random_seed=self.random_state, **self.outcome_params)
            else:
                mu1_model = xgb.XGBRegressor(**self.outcome_params)
            mu1_model.fit(X_train_1, Y_train_1)
            self.models['mu1'].append(mu1_model)
            out_pred_mu1[test_idx] = mu1_model.predict(X_test)

            # Train pi on All
            if self.method == 'elasticnet' or getattr(self, 'use_en', False):
                from sklearn.linear_model import LogisticRegressionCV
                pi_model = LogisticRegressionCV(cv=2, max_iter=500)
            elif self.method == 'catboost' and CATBOOST_AVAILABLE:
                pi_model = CatBoostClassifier(verbose=False, random_seed=self.random_state, **self.propensity_params)
            else:
                pi_model = xgb.XGBClassifier(**self.propensity_params)
            pi_model.fit(X_train, W_train)
            self.models['pi'].append(pi_model)
            out_pred_pi[test_idx] = pi_model.predict_proba(X_test)[:, 1]

        # Ensure overlap (clip propensity scores)
        out_pred_pi = np.clip(out_pred_pi, 0.01, 0.99)
        
        return out_pred_mu0, out_pred_mu1, out_pred_pi

    def predict(self, X):
        """
        Aggregates predictions from all K models.
        """
        pred_mu0 = np.mean([model.predict(X) for model in self.models['mu0']], axis=0)
        pred_mu1 = np.mean([model.predict(X) for model in self.models['mu1']], axis=0)
        pred_pi = np.mean([model.predict_proba(X)[:, 1] for model in self.models['pi']], axis=0)
        pred_pi = np.clip(pred_pi, 0.01, 0.99)
        return pred_mu0, pred_mu1, pred_pi
