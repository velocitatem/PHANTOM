# sklearn compatible models for agent detection
from sklearn.base import BaseEstimator, ClassifierMixin
from procesing.context import PipelineContext
from typing import Any
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd

TASK = 'classification'
LABELS = ['human', 'agent']

class XGBoostAgentClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost binary classifier for agent detection with class imbalance handling"""

    def __init__(self, context: PipelineContext = None, n_estimators=200, max_depth=6,
                 learning_rate=0.05, early_stopping_rounds=20):
        self.context = context
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X, y, eval_set=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()

        # class imbalance handling via scale_pos_weight
        n_neg = (y_arr == 0).sum()
        n_pos = (y_arr == 1).sum()
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos,
            eval_metric='auc',
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=42,
            tree_method='hist',
            enable_categorical=False
        )

        if eval_set:
            X_val, y_val = eval_set[0]
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            self.model_.fit(X_arr, y_arr, eval_set=[(X_val_arr, y_val_arr)], verbose=False)
        else:
            self.model_.fit(X_arr, y_arr)

        return self

    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict(X_arr)

    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict_proba(X_arr)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_ if self.model_ else None


class LightGBMAgentClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM binary classifier for agent detection with class imbalance handling"""

    def __init__(self, context: PipelineContext = None, n_estimators=200, max_depth=6,
                 learning_rate=0.05, early_stopping_rounds=20):
        self.context = context
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X, y, eval_set=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()

        # class imbalance handling via scale_pos_weight
        n_neg = (y_arr == 0).sum()
        n_pos = (y_arr == 1).sum()
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos,
            metric='auc',
            random_state=42,
            verbosity=-1
        )

        if eval_set:
            X_val, y_val = eval_set[0]
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            self.model_.fit(
                X_arr, y_arr,
                eval_set=[(X_val_arr, y_val_arr)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.model_.fit(X_arr, y_arr)

        return self

    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict(X_arr)

    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict_proba(X_arr)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_ if self.model_ else None
