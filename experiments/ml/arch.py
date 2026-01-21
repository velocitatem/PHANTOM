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


class WeakClassifier(BaseEstimator, ClassifierMixin, ABC):
    # a simple contrastive machine learning model
    # this model should learn to distinguish between human and agent behavior
    # using a weakly supervised approach and contrastive learning + augmentation
    #
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.kwargs = kwargs
