from artibot.constants import FEATURE_DIMENSION
import numpy as np
from artibot.feature_manager import sanitize_features
from artibot.utils import validate_features, feature_mask_for
from artibot.hyperparams import IndicatorHyperparams
from feature_engineering import calculate_technical_indicators
import os
import joblib
import pandas as pd


def load_backtest_data(csv_path: str) -> pd.DataFrame:
    """Load and engineer features for backtesting with timestamp handling."""
    print(f"[LOADER] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[LOADER] Raw columns: {df.columns.tolist()}")

    # Handle missing timestamps
    if "timestamp" not in df.columns:
        print("⚠️ Generating synthetic timestamps")
        df["timestamp"] = pd.date_range(
            start="2020-01-01", periods=len(df), freq="1min"
        )

    timestamps = df["timestamp"].copy()

    # Calculate missing features
    if "sma_10" not in df.columns:
        print("\ud83d\udd27 Calculating technical indicators...")
        df = calculate_technical_indicators(df)

    # Restore timestamps after feature calculation
    df["timestamp"] = timestamps
    print(f"[LOADER] Processed columns: {df.columns.tolist()}")
    return df


def load_and_clean_data(
    path: str, cache_path: str = "cached_features.pkl"
) -> np.ndarray:
    """Return cleaned feature array with caching."""
    if os.path.isfile(cache_path):
        try:
            cached = joblib.load(cache_path)
            print(f"[DEBUG] Loaded cached features: {cached.shape}")
            return np.asarray(cached, dtype=float)
        except Exception:
            pass

    # Use fixed loader instead of load_csv_hourly
    df = load_backtest_data(path)
    data = df.drop(columns=["timestamp"]).values.astype(float)

    print(f"[DEBUG] Processed CSV shape: {data.shape}")

    # Strict dimension validation
    if data.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected {FEATURE_DIMENSION} features, got {data.shape[1]}. "
            "Check indicator calculations!"
        )

    data = sanitize_features(data)
    mask = feature_mask_for(IndicatorHyperparams())
    validate_features(data, enabled_mask=mask)

    # Save to cache
    try:
        joblib.dump(data, cache_path)
    except Exception:
        pass

    return data


def get_backtest_data() -> np.ndarray:
    """Load and process data with debug output."""
    df = load_backtest_data("path.csv")
    print(f"[PRE-FEATURE] Data shape: {df.shape}")

    # Extract features (exclude timestamp)
    features = df.drop(columns=["timestamp"]).values
    print(f"[POST-FEATURE] Feature matrix: {features.shape}")

    return features
