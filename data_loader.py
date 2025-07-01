from config import FEATURE_CONFIG
import numpy as np
from artibot.feature_manager import sanitize_features
from artibot.utils import enforce_feature_dim, validate_features


def load_and_clean_data(path):
    """Load CSV data and ensure correct feature dimensions."""
    from artibot.dataset import load_csv_hourly

    data = np.array(load_csv_hourly(path), dtype=float)
    if data.size == 0:
        return data

    if data.shape[1] != FEATURE_CONFIG["expected_features"]:
        data = enforce_feature_dim(data, FEATURE_CONFIG["expected_features"])

    data = sanitize_features(data)
    validate_features(data)
    assert not np.isnan(data).any(), "NaN detected in features"
    assert not np.isinf(data).any(), "Inf detected in features"
    assert data.any(axis=0).all(), "Zero-feature detected"

    return data
