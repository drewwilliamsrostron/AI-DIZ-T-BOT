from __future__ import annotations

from typing import Any

import config
from data_loader import load_backtest_data


def run_backtest(strategy: Any, data_path: str):
    """Run a strategy on CSV data using engineered features.

    Parameters
    ----------
    strategy:
        Object exposing a ``run`` method accepting ``features`` and
        ``timestamps`` arrays.
    data_path:
        Path to the CSV file containing at least a ``timestamp`` column.
    """

    df = load_backtest_data(data_path)

    timestamps = df["timestamp"].values
    feature_cols = getattr(
        config, "FEATURE_COLUMNS", config.FEATURE_CONFIG["feature_columns"]
    )
    feature_matrix = df[feature_cols].values

    if feature_matrix.shape[1] != len(feature_cols):
        raise ValueError(
            f"Backtest requires {len(feature_cols)} features, "
            f"got {feature_matrix.shape[1]}. Check data pipeline!"
        )

    print(
        f"[BACKTEST] Loaded {len(feature_matrix)} candles with {feature_matrix.shape[1]} validated features"
    )

    results = strategy.run(feature_matrix, timestamps)
    return results
