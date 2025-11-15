import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def build_transition_prob_matrix(df: pd.DataFrame):
    df = df.dropna(subset=['eventName'])
    events = df['eventName'].tolist()
    labels = pd.Index(events).unique().tolist()
    idx = {e:i for i,e in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=float)
    for a, b in zip(events, events[1:]):
        M[idx[a], idx[b]] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(M, row_sums, where=row_sums>0)  # row-normalized
    return P, labels

# https://medium.com/data-science/time-series-data-markov-transition-matrices-7060771e362b
from graphviz import Digraph
import numpy as np
import pandas as pd

def _as_prob_df(matrix, labels=None):
    """Return a square DataFrame with index=columns=labels."""
    if isinstance(matrix, pd.DataFrame):
        # Ensure square and aligned
        assert (matrix.index == matrix.columns).all(), "Index/columns must match."
        return matrix
    matrix = np.asarray(matrix, dtype=float)
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
    if labels is None:
        raise ValueError("labels are required when matrix is not a DataFrame")
    assert len(labels) == matrix.shape[0], "labels length must match matrix size."
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))

def _df_to_edgelist(P: pd.DataFrame, threshold=0.0, round_digits=2):
    """Build weighted edges > threshold."""
    edges = []
    for src in P.index:
        for dst in P.columns:
            w = float(P.loc[src, dst])
            if w > threshold:
                edges.append((str(src), str(dst), f"{w:.{round_digits}f}"))
    return edges

def render_graph(fname, matrix, ls_index=None, threshold=0.0, fmt="svg", view=False):
    """
    fname: output file stem (no extension)
    matrix: NumPy array or pandas DataFrame of transition PROBABILITIES
    ls_index: ordered labels (required if matrix is not a DataFrame)
    threshold: hide edges with weight <= threshold
    fmt: 'svg'|'png'|'pdf' etc.
    view: open after rendering
    """
    P = _as_prob_df(matrix, labels=ls_index)
    edges = _df_to_edgelist(P, threshold=threshold)

    g = Digraph(format=fmt)
    g.attr(rankdir="LR", size="30")
    g.attr("node", shape="circle")

    # ensure isolated nodes appear
    for node in P.index:
        g.node(str(node), width="1", height="1")

    for src, dst, label in edges:
        g.edge(src, dst, label=label)

    g.render(fname, view=view, cleanup=True)
    return g


class TransitionProbMatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.P_ = None
        self.labels_ = None

    def fit(self, X: pd.DataFrame, y=None):
        P, labels = build_transition_prob_matrix(X)
        self.P_ = P
        self.labels_ = labels
        return self

    def transform(self, X: pd.DataFrame = None):
        return self.P_, self.labels_

    def render(self, fname: str, fmt="svg", view=False):
        if self.P_ is None or self.labels_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return render_graph(
            fname,
            self.P_,
            ls_index=self.labels_,
            threshold=self.threshold,
            fmt=fmt,
            view=view
        )


class SessionTransitionProbMatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0, session_col='sessionId'):
        self.threshold = threshold
        self.session_col = session_col
        self.session_matrices_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.session_col not in X.columns:
            raise ValueError(f"Column '{self.session_col}' not found in DataFrame")

        session_matrices = {}
        for session_id, grp in X.groupby(self.session_col):
            if len(grp) > 1:  # need at least 2 events for transitions
                P, labels = build_transition_prob_matrix(grp)
                session_matrices[session_id] = {'matrix': P, 'labels': labels}

        self.session_matrices_ = session_matrices
        return self

    def transform(self, X: pd.DataFrame = None):
        if self.session_matrices_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return pd.Series(self.session_matrices_)

    def render_session(self, session_id: str, fname: str, fmt="svg", view=False):
        if self.session_matrices_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        if session_id not in self.session_matrices_:
            raise ValueError(f"Session '{session_id}' not found in fitted data.")

        sess_data = self.session_matrices_[session_id]
        return render_graph(
            fname,
            sess_data['matrix'],
            ls_index=sess_data['labels'],
            threshold=self.threshold,
            fmt=fmt,
            view=view
        )
if __name__ == "__main__":
    # Example usage
    data = {
        'eventName': [
            'A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'C',
            'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'
        ]
    }
    df = pd.DataFrame(data)

    transformer = TransitionProbMatrixTransformer(threshold=0.1)
    transformer.fit(df)
    P, labels = transformer.transform(None)

    print("Transition Probability Matrix:")
    print(pd.DataFrame(P, index=labels, columns=labels))

    # Render the graph
    transformer.render("transition_graph", fmt="svg", view=False)
