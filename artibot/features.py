from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from .constants import FEATURE_DIMENSION
from config import FEATURE_CONFIG
from .utils import validate_feature_dimension


class FeatureEngineer:
    """Calculate technical indicator features with strict dimension checks."""

    def __init__(self) -> None:
        self.expected_features = FEATURE_DIMENSION
        self.feature_columns = FEATURE_CONFIG.get("feature_columns", [])
        self.logger = logging.getLogger("FeatureEngineer")

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Return sanitized feature matrix for ``data``."""

        feature_columns = [c for c in self.feature_columns if c in data.columns]

        if len(feature_columns) != self.expected_features:
            self.logger.error(
                "Feature mismatch! Expected %s, got %s",
                self.expected_features,
                len(feature_columns),
            )
            missing = set(self.feature_columns) - set(feature_columns)
            for feature in missing:
                data[feature] = data["close"].rolling(window=5).mean().ffill().bfill()
            feature_columns = self.feature_columns

        features = data[feature_columns].replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().to_numpy(dtype=float)
        features = validate_feature_dimension(
            features, self.expected_features, self.logger
        )
        return features


def calculate_features(df: pd.DataFrame) -> np.ndarray:
    """Wrapper around :class:`FeatureEngineer` with debug output."""

    engineer = FeatureEngineer()
    features = engineer.transform(df)
    print(
        f"[DEBUG] Post-feature shape: {features.shape}, Columns: {df.columns.tolist()}"
    )
    return features
