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


# feature extraction utilities for trajectory -> feature vector
def transition_histogram(events: List, state_fn, max_states: int = 50) -> np.ndarray:
    """Compute normalized histogram of state transitions in trajectory"""
    if len(events) < 2:
        return np.zeros(max_states)
    states = [state_fn(e) for e in events]
    trans_counts = defaultdict(int)
    for s, s_next in zip(states, states[1:]):
        trans_counts[(s, s_next)] += 1
    total = sum(trans_counts.values())
    hist = np.array(list(trans_counts.values())[:max_states], dtype=np.float32)
    hist = np.pad(hist, (0, max(0, max_states - len(hist))))
    return hist / (total + 1e-10)


def temporal_signature(events: List, ts_fn) -> np.ndarray:
    """Extract temporal features: mean/std/skew of inter-event times"""
    if len(events) < 2:
        return np.zeros(4, dtype=np.float32)
    times = sorted([ts_fn(e) for e in events])
    diffs = np.diff(times).astype(np.float32)
    if len(diffs) == 0:
        return np.zeros(4, dtype=np.float32)
    mean_dt, std_dt = np.mean(diffs), np.std(diffs) + 1e-10
    skew = np.mean(((diffs - mean_dt) / std_dt) ** 3) if std_dt > 1e-8 else 0.0
    return np.array([mean_dt, std_dt, skew, len(diffs)], dtype=np.float32)


def state_coverage(events: List, state_fn, mdp_states: set) -> float:
    """Fraction of MDP states visited by trajectory"""
    if not mdp_states:
        return 0.0
    visited = set(state_fn(e) for e in events)
    return len(visited & mdp_states) / len(mdp_states)


def transition_entropy(events: List, state_fn) -> float:
    """Compute entropy of transition distribution (randomness of navigation)"""
    if len(events) < 2:
        return 0.0
    states = [state_fn(e) for e in events]
    trans_counts = defaultdict(int)
    for s, s_next in zip(states, states[1:]):
        trans_counts[(s, s_next)] += 1
    total = sum(trans_counts.values())
    probs = [c / total for c in trans_counts.values()]
    return -sum(p * np.log(p + 1e-10) for p in probs)


def featurize_trajectory(events: List, mdp: Optional[Dict] = None, input_dim: int = 64) -> np.ndarray:
    """Convert trajectory to fixed-dim feature vector"""
    def _state_repr(e):
        return f"{getattr(e, 'page', None) or 'unk'}|{getattr(e, 'productId', None) or 'none'}|{e.eventName}"

    def _ts_fn(e):
        ts = getattr(e, 'ts', None)
        if isinstance(ts, str):
            from datetime import datetime
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except:
                return 0.0
        return float(ts) if ts else 0.0

    feats = []
    feats.extend(transition_histogram(events, _state_repr, max_states=40))  # 40 dims
    feats.extend(temporal_signature(events, _ts_fn))  # 4 dims
    mdp_states = set(mdp.get('states', [])) if mdp else set()
    feats.append(state_coverage(events, _state_repr, mdp_states))  # 1 dim
    feats.append(transition_entropy(events, _state_repr))  # 1 dim
    feats.append(len(events))  # trajectory length
    feats.append(len(set(_state_repr(e) for e in events)))  # unique states

    # event type distribution (page_view, hover, cart, purchase indicators)
    event_names = [e.eventName for e in events]
    feats.append(sum(1 for n in event_names if 'page' in n.lower()) / (len(events) + 1))
    feats.append(sum(1 for n in event_names if 'hover' in n.lower()) / (len(events) + 1))
    feats.append(sum(1 for n in event_names if 'cart' in n.lower()) / (len(events) + 1))
    feats.append(sum(1 for n in event_names if 'purchase' in n.lower() or 'checkout' in n.lower()) / (len(events) + 1))

    # pad/truncate to input_dim
    feats = np.array(feats[:input_dim], dtype=np.float32)
    if len(feats) < input_dim:
        feats = np.pad(feats, (0, input_dim - len(feats)))
    return feats


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
