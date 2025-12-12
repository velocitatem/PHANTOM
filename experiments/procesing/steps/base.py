from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from procesing.context import PipelineContext
from typing import Any

class BaseContextStep(BaseEstimator, TransformerMixin, ABC):
    """
    Base for all pipeline steps.
    Each step is stateless, context-driven, and performs ONE transformation.
    """

    def __init__(self, context: PipelineContext):
        self.context = context

    def fit(self, X=None, y=None):
        """Most steps don't need training"""
        return self

    @abstractmethod
    def transform(self, X) -> Any:
        """Transform input using context. Must be implemented by subclass."""
        pass

    def get_params(self, deep=True):
        """sklearn compatibility"""
        return {'context': self.context}

    def set_params(self, **params):
        """sklearn compatibility"""
        if 'context' in params:
            self.context = params['context']
        return self
