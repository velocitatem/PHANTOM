from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from logging import getLogger
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from ml.evals import evaluate, log_feature_importance
from ml.arch import XGBoostAgentClassifier, LightGBMAgentClassifier, LABELS

logger = getLogger(__name__)

FEATURE_COLS_EXCLUDE = ['sessionId', 'experimentId', 'is_agent', 'xp_human_only', 'xp_market_mode', 'browser_family']
RUNS_DIR = Path('ml/runs')
CHECKPOINTS_DIR = Path('ml/checkpoints')


def prepare_data(df):
    """
    Prepare feature matrix and labels from raw dataframe
    Handles missing labels, feature selection, and categorical encoding
    Returns: (X, y, feature_cols)
    """
    # drop rows with missing labels
    n_before = len(df)
    df = df[df['is_agent'].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} sessions with missing labels")

    if len(df) == 0:
        logger.error("No labeled data available")
        return None, None, None

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]

    # handle categorical browser_family via one-hot encoding
    if 'browser_family' in df.columns:
        browser_dummies = pd.get_dummies(df['browser_family'], prefix='browser', drop_first=True)
        df = pd.concat([df, browser_dummies], axis=1)
        feature_cols.extend(browser_dummies.columns.tolist())

    X = df[feature_cols].fillna(0)
    y = df['is_agent'].astype(int)

    return X, y, feature_cols


def train(data_path=None, model_type='xgboost', test_size=0.2, random_state=42,
          n_estimators=200, max_depth=6, learning_rate=0.05):
    """
    Train agent detection classifier
    Args:
        data_path: path to labeled feature matrix CSV or parquet
        model_type: 'xgboost' or 'lightgbm'
        test_size: fraction for test split
        random_state: seed for reproducibility
    """
    RUNS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=RUNS_DIR / run_name)
    logger.info(f"Starting training run: {run_name}")

    # load data
    if data_path is None:
        logger.error("data_path required")
        return
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} sessions from {data_path}")

    # prepare features and labels
    if 'is_agent' not in df.columns:
        logger.error("Missing is_agent column")
        return

    X, y, feature_cols = prepare_data(df)
    if X is None:
        return

    # class distribution
    n_agents = y.sum()
    n_humans = (y == 0).sum()
    logger.info(f"Class distribution: {n_humans} humans, {n_agents} agents" + (f" (ratio {n_humans / n_agents:.2f})" if n_agents > 0 else ""))

    # train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # init model
    if model_type == 'xgboost':
        model = XGBoostAgentClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
    elif model_type == 'lightgbm':
        model = LightGBMAgentClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    # train with eval set for early stopping
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    logger.info("Training complete")

    # evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluate(y_pred, y_prob, y_test, writer, epoch=0)

    # log feature importance
    log_feature_importance(writer, model, X.columns.tolist(), epoch=0)

    # save model
    model_path = CHECKPOINTS_DIR / f"{run_name}.pkl"
    joblib.dump({'model': model, 'feature_cols': X.columns.tolist(), 'run_name': run_name}, model_path)
    logger.info(f"Model saved to {model_path}")

    writer.close()
    return model, X.columns.tolist()


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'xgboost'
    train(data_path, model_type=model_type)
