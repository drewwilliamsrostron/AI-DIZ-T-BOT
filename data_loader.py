from artibot.constants import FEATURE_DIMENSION
import numpy as np
from artibot.feature_manager import sanitize_features
from artibot.utils import enforce_feature_dim, validate_features, feature_mask_for
from artibot.hyperparams import IndicatorHyperparams
import os
import joblib


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
