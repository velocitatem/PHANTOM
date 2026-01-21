# sklearn compatible models for agent detection
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any, Optional, Tuple, Dict, List
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))
from lib.features import (
    transition_histogram as _lib_transition_histogram,
    temporal_signature as _lib_temporal_signature,
    state_coverage as _lib_state_coverage,
    transition_entropy as _lib_transition_entropy,
    featurize_trajectory as _lib_featurize_trajectory,
    parse_timestamp
)
from lib.state import event_to_state, get_event_name, get_timestamp

TASK = 'classification'
LABELS = ['human', 'agent']


class WeakClassifier(BaseEstimator, ClassifierMixin, ABC):
    # a simple contrastive machine learning model learns to distinguish human/agent behavior
    # using weakly supervised contrastive learning + augmentation
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.kwargs = kwargs


class TrajectoryEncoder(nn.Module):
    """Encode variable-length event sequences to fixed-dim embedding via bidirectional LSTM"""
    def __init__(self, input_dim: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.event_embed = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, seq_len, input_dim)
        h = F.relu(self.event_embed(x))
        _, (hn, _) = self.lstm(h)
        hn = torch.cat([hn[-2], hn[-1]], dim=1)  # concat bidirectional hidden states
        return F.normalize(self.proj(hn), dim=1)  # L2 normalized


class ContrastiveWeakClassifier(WeakClassifier):
    """Contrastive learning classifier for human/agent trajectory discrimination"""
    def __init__(self, input_dim: int = 64, embed_dim: int = 32, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.margin = margin
        self.encoder = TrajectoryEncoder(input_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._fitted = False

    def to_device(self):
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.classifier(emb)

    def fit(self, X, y=None):  # sklearn interface - actual training in weak.train.py
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            logits = self.forward(x)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            logits = self.forward(x)
            return F.softmax(logits, dim=1).cpu().numpy()


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Triplet loss using cosine similarity (for L2-normalized embeddings). margin in [0,1] range."""
    pos_sim = F.cosine_similarity(anchor, positive)  # higher = more similar
    neg_sim = F.cosine_similarity(anchor, negative)
    return F.relu(neg_sim - pos_sim + margin).mean()  # want pos_sim > neg_sim + margin


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss (SimCLR style)"""
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2N, embed_dim)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -float('inf'))
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])  # positive pairs
    return F.cross_entropy(sim, labels)


# feature extraction utilities - delegating to lib.features for unified implementation
# these wrappers maintain backwards compatibility for existing imports

def transition_histogram(events: List, state_fn, max_states: int = 50) -> np.ndarray:
    """Compute normalized histogram of state transitions in trajectory"""
    return _lib_transition_histogram(events, state_fn, max_states)


def temporal_signature(events: List, ts_fn) -> np.ndarray:
    """Extract temporal features: mean/std/skew of inter-event times"""
    return _lib_temporal_signature(events, ts_fn)


def state_coverage(events: List, state_fn, mdp_states: set) -> float:
    """Fraction of MDP states visited by trajectory"""
    return _lib_state_coverage(events, state_fn, mdp_states)


def transition_entropy(events: List, state_fn) -> float:
    """Compute entropy of transition distribution (randomness of navigation)"""
    return _lib_transition_entropy(events, state_fn)


def featurize_trajectory(events: List, mdp: Optional[Dict] = None, input_dim: int = 64) -> np.ndarray:
    """Convert trajectory to fixed-dim feature vector - uses lib.features implementation"""
    mdp_states = set(mdp.get('states', [])) if mdp else set()

    def _ts_fn(e):
        return parse_timestamp(get_timestamp(e))

    def _event_name_fn(e):
        return get_event_name(e)

    return _lib_featurize_trajectory(events, event_to_state, _ts_fn, _event_name_fn, mdp_states, input_dim)


# gradient boosting classifiers for comparison baselines
class XGBoostAgentClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier for human/agent detection from session features"""
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           learning_rate=self.learning_rate, **self.kwargs)
            self.model.fit(X, y)
        except ImportError:
            raise ImportError("xgboost required for XGBoostAgentClassifier")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit the model first")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit the model first")
        return self.model.predict_proba(X)


class LightGBMAgentClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM classifier for human/agent detection from session features"""
    def __init__(self, n_estimators: int = 100, max_depth: int = -1, learning_rate: float = 0.1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                            learning_rate=self.learning_rate, verbose=-1, **self.kwargs)
            self.model.fit(X, y)
        except ImportError:
            raise ImportError("lightgbm required for LightGBMAgentClassifier")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit the model first")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit the model first")
        return self.model.predict_proba(X)
