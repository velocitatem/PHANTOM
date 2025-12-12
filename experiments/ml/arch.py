# this should retrun a model with exposed methods fit and transform method in an sklearn style
from sklearn.base import BaseEstimator, TransformerMixin
from procesing.context import PipelineContext
from typing import Any

TASK = 'classification'
LABELS = ['agent', 'human']

class BaseModel(BaseEstimator, TransformerMixin):
    def __init__(self, context: PipelineContext):
        self.context = context

    def fit(self, X=None, y=None):
        return self

    def transform(self, X) -> Any:
        pass
