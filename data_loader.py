from artibot.constants import FEATURE_DIMENSION
import numpy as np
from artibot.feature_manager import sanitize_features
from artibot.utils import enforce_feature_dim, validate_features, feature_mask_for
from artibot.hyperparams import IndicatorHyperparams


def load_and_clean_data(path):
    """Load CSV data and ensure correct feature dimensions."""
    from artibot.dataset import load_csv_hourly

    data = np.array(load_csv_hourly(path), dtype=float)
    if data.size == 0:
        return data

    if data.shape[1] != FEATURE_DIMENSION:
        data = enforce_feature_dim(data, FEATURE_DIMENSION)

    data = sanitize_features(data)
    mask = feature_mask_for(IndicatorHyperparams())
    validate_features(data, enabled_mask=mask)
    assert not np.isnan(data).any(), "NaN detected in features"
    assert not np.isinf(data).any(), "Inf detected in features"

    return data
