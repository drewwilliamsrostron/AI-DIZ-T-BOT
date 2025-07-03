from artibot.constants import FEATURE_DIMENSION
import numpy as np
from artibot.feature_manager import sanitize_features
from artibot.utils import (
    enforce_feature_dim,
    validate_features,
    feature_mask_for,
)
from artibot.hyperparams import IndicatorHyperparams
from feature_engineering import calculate_technical_indicators
import config
import os
import joblib
import pandas as pd


def load_backtest_data(path: str) -> pd.DataFrame:
    """Load raw CSV data for backtesting with a debug summary.

    The ``timestamp`` column is preserved so sequencing remains valid after
    feature engineering.
    """

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise KeyError("CSV must contain 'timestamp' column")

    timestamps = df["timestamp"].copy()

    if not set(config.FEATURE_CONFIG["feature_columns"]).issubset(df.columns):
        print("🚨 Backtest data missing features - calculating indicators...")
        df = calculate_technical_indicators(df)

    df["timestamp"] = timestamps

    print(f"[DEBUG] Raw CSV shape: {df.shape}, Columns: {df.columns.tolist()}")
    return df


def load_and_clean_data(
    path: str, cache_path: str = "cached_features.pkl"
) -> np.ndarray:
    """Return cleaned feature array loaded from ``path``.

    When ``cache_path`` exists the cached array is returned instead of reading
    the CSV.  The resulting data is validated and padded to
    :data:`~artibot.constants.FEATURE_DIMENSION` columns.
    """

    from artibot.dataset import load_csv_hourly

    if os.path.isfile(cache_path):
        try:
            cached = joblib.load(cache_path)
            print(f"[DEBUG] Loaded cached features: {cached.shape}")
            return np.asarray(cached, dtype=float)
        except Exception:
            pass

    data = np.array(load_csv_hourly(path), dtype=float)
    if data.size == 0:
        return data

    print(f"[DEBUG] Raw CSV shape: {data.shape}")

    if data.shape[1] != FEATURE_DIMENSION:
        data = enforce_feature_dim(data, FEATURE_DIMENSION)

    data = sanitize_features(data)
    mask = feature_mask_for(IndicatorHyperparams())
    validate_features(data, enabled_mask=mask)
    assert not np.isnan(data).any(), "NaN detected in features"
    assert not np.isinf(data).any(), "Inf detected in features"

    try:
        joblib.dump(data, cache_path)
    except Exception:
        pass

    return data


def get_backtest_data() -> np.ndarray:
    """Load, process and return data for debugging the pipeline."""

    data = load_backtest_data("path.csv")
    print(f"[PRE-FEATURE] Data shape: {data.shape}")
    from artibot.features import calculate_features

    processed = calculate_features(data)
    print(f"[POST-FEATURE] Data shape: {processed.shape}")
    return processed
