"""Walk-forward backtesting with nested hyperparameter optimisation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

import pandas as pd

from .ensemble import EnsembleModel
from .training import csv_training_thread
from .backtest import robust_backtest
from .optuna_opt import run_search
import artibot.globals as G


# ---------------------------------------------------------------------------
# Configuration constants (example values)
# ---------------------------------------------------------------------------
TRAIN_SIZE = 24 * 30 * 6  # 6 months of hourly bars
TEST_SIZE = 24 * 30  # 1 month
STEP = 24 * 30  # slide by 1 month
MAX_EPOCHS = 5


def walk_forward_opt(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run walk-forward backtests with nested hyperparameter search."""

    results: List[Dict[str, Any]] = []
    feature_cols = [c for c in data.columns if c != "y"]
    for start in range(0, len(data) - TRAIN_SIZE - TEST_SIZE + 1, STEP):
        train_df = data.iloc[start : start + TRAIN_SIZE].copy()
        test_df = data.iloc[start + TRAIN_SIZE : start + TRAIN_SIZE + TEST_SIZE].copy()
        hp, lr, wd = run_search(train_df.values.tolist(), n_trials=20)
        model = EnsembleModel(
            n_features=len(feature_cols), lr=lr, weight_decay=wd, n_models=1
        )
        model.indicator_hparams = hp
        stop = threading.Event()
        csv_training_thread(
            model,
            train_df.values.tolist(),
            stop,
            {"ADAPT_TO_LIVE": False},
            use_prev_weights=False,
            max_epochs=MAX_EPOCHS,
            update_globals=False,
        )

        metrics = robust_backtest(model, test_df.values.tolist())
        if metrics.get("trades", 0) == 0:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
        else:
            G.push_backtest_metrics(metrics)
        logging.info(
            "Window %d-%d  params=%s  net_pct=%.2f",
            start,
            start + TRAIN_SIZE,
            {
                "lr": lr,
                "weight_decay": wd,
                **hp.__dict__,
            },
            metrics.get("net_pct", 0.0),
        )
        results.append({"start": start, "params": hp.__dict__, **metrics})

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
