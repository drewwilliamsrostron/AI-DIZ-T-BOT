from __future__ import annotations

from typing import Any


def run_backtest(strategy: Any, data_path: str):
    """Run ``strategy`` on processed features from ``data_path``."""

    from data_loader import load_backtest_data

    df = load_backtest_data(data_path)

    # Validate feature count
    feature_cols = [c for c in df.columns if c != "timestamp"]
    features = df[feature_cols].values
    if features.shape[1] != 16:
        print(
            f"\N{POLICE CARS REVOLVING LIGHT} Expected 16 features, got {features.shape[1]}"
        )
        print("Columns:", feature_cols)
        raise ValueError("Feature dimension mismatch")

    print(f"[BACKTEST] Running with {features.shape[1]} validated features")
    return strategy.run(features, df["timestamp"].values)
