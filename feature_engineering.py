from __future__ import annotations

import numpy as np
import pandas as pd

import config
from artibot.dataset import generate_fixed_features
from artibot.hyperparams import IndicatorHyperparams


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of engineered features for ``df``."""

    data = df.copy()
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

    volume_col = None
    for col in ("volume_btc", "volume", "volume_usd"):
        if col in data.columns:
            volume_col = col
            break
    if volume_col is None:
        volume_col = "volume"
        data[volume_col] = 0.0

    if "unix" not in data.columns:
        data["unix"] = np.arange(len(data))

    arr = data[["unix", "open", "high", "low", "close", volume_col]].to_numpy(
        dtype=float
    )

    feats = generate_fixed_features(arr, IndicatorHyperparams())
    print(
        f"✅ Calculated {feats.shape[1]} features from raw {df.shape[1]}-column input"
    )

    feature_cols = config.FEATURE_CONFIG.get("feature_columns", [])
    return pd.DataFrame(feats, columns=feature_cols)
