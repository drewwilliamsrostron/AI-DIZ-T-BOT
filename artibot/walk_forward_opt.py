"""Walk-forward backtesting with nested hyperparameter optimisation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

import pandas as pd
from sklearn.base import BaseEstimator

from .ensemble import EnsembleModel
from .hyperparams import IndicatorHyperparams, WARMUP_STEPS
from .training import csv_training_thread
from .backtest import robust_backtest
from .optuna_opt import run_bohb
import artibot.globals as G


# ---------------------------------------------------------------------------
# Configuration constants (example values)
# ---------------------------------------------------------------------------
TRAIN_SIZE = 24 * 30 * 6  # 6 months of hourly bars
TEST_SIZE = 24 * 30  # 1 month
STEP = 24 * 30  # slide by 1 month
INNER_FOLDS = 3
MAX_EPOCHS = 5


class EnsembleEstimator(BaseEstimator):
    """sklearn wrapper for :class:`EnsembleModel`."""

    def __init__(
        self,
        n_features: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        indicator_hp: IndicatorHyperparams | None = None,
    ) -> None:
        self.n_features = n_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.indicator_hp = indicator_hp or IndicatorHyperparams()
        self.model_: EnsembleModel | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "EnsembleEstimator":
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        self.model_ = EnsembleModel(
            n_features=self.n_features,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_models=1,
            warmup_steps=WARMUP_STEPS,
        )
        self.model_.indicator_hparams = self.indicator_hp
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
        if metrics.get("trades", 0) == 0:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
        else:
            G.push_backtest_metrics(metrics)
        return float(metrics.get("net_pct", 0.0))


def walk_forward_opt(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run walk-forward backtests with nested hyperparameter search."""

    results: List[Dict[str, Any]] = []
    feature_cols = [c for c in data.columns if c != "y"]

    for start in range(0, len(data) - TRAIN_SIZE - TEST_SIZE + 1, STEP):
        train_df = data.iloc[start : start + TRAIN_SIZE].copy()
        test_df = data.iloc[start + TRAIN_SIZE : start + TRAIN_SIZE + TEST_SIZE].copy()
        hp, params = run_bohb(n_trials=20)
        est = EnsembleEstimator(
            n_features=len(feature_cols),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
            indicator_hp=hp,
        )
        est.fit(train_df[feature_cols], train_df["y"])

        stop = threading.Event()
        csv_training_thread(
            est.model_,
            train_df.values.tolist(),
            stop,
            {"ADAPT_TO_LIVE": False},
            use_prev_weights=False,
            max_epochs=MAX_EPOCHS,
            update_globals=False,
        )

        metrics = robust_backtest(est.model_, test_df.values.tolist())
        if metrics.get("trades", 0) == 0:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
        else:
            G.push_backtest_metrics(metrics)
        logging.info(
            "Window %d-%d  params=%s  net_pct=%.2f",
            start,
            start + TRAIN_SIZE,
            params,
            metrics.get("net_pct", 0.0),
        )
        results.append({"start": start, "params": params, **metrics})

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
