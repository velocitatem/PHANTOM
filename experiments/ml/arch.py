# sklearn compatible models for agent detection
from sklearn.base import BaseEstimator, ClassifierMixin
from procesing.context import PipelineContext
from typing import Any, Optional, Tuple
from abc import ABC, abstractmethod
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd

TASK = 'classification'
LABELS = ['human', 'agent']


class BaseAgentClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Base class for tree-based agent detection classifiers with common logic"""

    def __init__(self, context: Optional[PipelineContext] = None, n_estimators: int = 200,
                 max_depth: int = 6, learning_rate: float = 0.05,
                 early_stopping_rounds: int = 20):
        self.context = context
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.feature_names_ = None

    def _to_array(self, X):
        """Convert pandas structures to numpy arrays"""
        return X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X

    def _compute_pos_weight(self, y_arr):
        """Calculate scale_pos_weight for class imbalance handling"""
        n_neg, n_pos = (y_arr == 0).sum(), (y_arr == 1).sum()
        return n_neg / n_pos if n_pos > 0 else 1.0

    def _prepare_eval_set(self, eval_set):
        """Convert eval_set to numpy arrays if needed"""
        if not eval_set:
            return None
        X_val, y_val = eval_set[0]
        return [(self._to_array(X_val), self._to_array(y_val))]

    @abstractmethod
    def _build_model(self, scale_pos: float):
        """Build the underlying model instance (must be implemented by subclasses)"""
        pass

    @abstractmethod
    def _fit_with_eval(self, X_arr, y_arr, eval_arr):
        """Fit model with evaluation set (must be implemented by subclasses)"""
        pass

    def fit(self, X, y, eval_set=None):
        X_arr, y_arr = self._to_array(X), self._to_array(y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()

        scale_pos = self._compute_pos_weight(y_arr)
        self.model_ = self._build_model(scale_pos)

        eval_arr = self._prepare_eval_set(eval_set)
        if eval_arr:
            self._fit_with_eval(X_arr, y_arr, eval_arr)
        else:
            self.model_.fit(X_arr, y_arr)

        return self

    def predict(self, X):
        return self.model_.predict(self._to_array(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._to_array(X))

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_ if self.model_ else None


class XGBoostAgentClassifier(BaseAgentClassifier):
    """XGBoost binary classifier for agent detection with class imbalance handling"""

    def _build_model(self, scale_pos: float):
        return xgb.XGBClassifier(
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

    def _fit_with_eval(self, X_arr, y_arr, eval_arr):
        self.model_.fit(X_arr, y_arr, eval_set=eval_arr, verbose=False)


class LightGBMAgentClassifier(BaseAgentClassifier):
    """LightGBM binary classifier for agent detection with class imbalance handling"""

    def _build_model(self, scale_pos: float):
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos,
            metric='auc',
            random_state=42,
            verbosity=-1
        )

    def _fit_with_eval(self, X_arr, y_arr, eval_arr):
        self.model_.fit(
            X_arr, y_arr,
            eval_set=eval_arr,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        )
