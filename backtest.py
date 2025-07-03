from __future__ import annotations

from typing import Any

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

    # Load and process data with feature engineering
    df = load_backtest_data(data_path)

    # Extract features and validate dimensions
    feature_cols = [col for col in df.columns if col != "timestamp"]
    features = df[feature_cols].values

    # Critical dimension check
    expected_features = 16  # From config
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Backtest requires {expected_features} features, "
            f"got {features.shape[1]}. Check data pipeline!"
        )

    # Debug log for verification
    print(f"[BACKTEST] Running with {features.shape[1]} validated features")

    # Run strategy
    return strategy.run(features, df["timestamp"].values)
