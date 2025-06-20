"""Walk-forward backtesting with nested hyperparameter optimisation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .ensemble import EnsembleModel
from .training import csv_training_thread
from .backtest import robust_backtest


# ---------------------------------------------------------------------------
# Configuration constants (example values)
# ---------------------------------------------------------------------------
TRAIN_SIZE = 24 * 30 * 6  # 6 months of hourly bars
TEST_SIZE = 24 * 30  # 1 month
STEP = 24 * 30  # slide by 1 month
INNER_FOLDS = 3
MAX_EPOCHS = 5

# Example hyperparameter grid
param_grid = {
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-5],
}


class EnsembleEstimator(BaseEstimator):
    """sklearn wrapper for :class:`EnsembleModel`."""

    def __init__(self, n_features: int, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.n_features = n_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_: EnsembleModel | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "EnsembleEstimator":
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        self.model_ = EnsembleModel(
            n_features=self.n_features,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_models=1,
        )
        stop = threading.Event()
        csv_training_thread(
            self.model_,
            df.values.tolist(),
            stop,
            {"ADAPT_TO_LIVE": False},
            use_prev_weights=False,
            max_epochs=MAX_EPOCHS,
            update_globals=False,
        )
        return self

    def score(self, X: pd.DataFrame, y: pd.Series | None = None) -> float:
        if self.model_ is None:
            raise RuntimeError("Estimator not fitted")
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        metrics = robust_backtest(self.model_, df.values.tolist())
        return float(metrics.get("net_pct", 0.0))


def walk_forward_opt(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run walk-forward backtests with nested hyperparameter search."""

    results: List[Dict[str, Any]] = []
    feature_cols = [c for c in data.columns if c != "y"]
    tscv = TimeSeriesSplit(n_splits=INNER_FOLDS)

    for start in range(0, len(data) - TRAIN_SIZE - TEST_SIZE + 1, STEP):
        train_df = data.iloc[start : start + TRAIN_SIZE].copy()
        test_df = data.iloc[start + TRAIN_SIZE : start + TRAIN_SIZE + TEST_SIZE].copy()
        est = EnsembleEstimator(n_features=len(feature_cols))
        grid = GridSearchCV(est, param_grid=param_grid, cv=tscv)
        grid.fit(train_df[feature_cols], train_df["y"])

        best_est: EnsembleEstimator = grid.best_estimator_
        # Final short train pass on the full training window
        stop = threading.Event()
        csv_training_thread(
            best_est.model_,
            train_df.values.tolist(),
            stop,
            {"ADAPT_TO_LIVE": False},
            use_prev_weights=False,
            max_epochs=MAX_EPOCHS,
            update_globals=False,
        )

        metrics = robust_backtest(best_est.model_, test_df.values.tolist())
        logging.info(
            "Window %d-%d  params=%s  net_pct=%.2f",
            start,
            start + TRAIN_SIZE,
            grid.best_params_,
            metrics.get("net_pct", 0.0),
        )
        results.append({"start": start, "params": grid.best_params_, **metrics})

    return results


def _example_usage() -> None:
    df = pd.read_csv("Gemini_BTCUSD_1h.csv")[
        ["unix", "open", "high", "low", "close"]
    ].copy()
    df["y"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    summary = walk_forward_opt(df)
    print(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _example_usage()
