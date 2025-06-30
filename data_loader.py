from config import FEATURE_CONFIG
import numpy as np
import torch
from core.feature_manager import sanitize_features, validate_and_align_features


def load_and_clean_data(path):
    """Load CSV data and ensure correct feature dimensions."""
    from artibot.dataset import load_csv_hourly

    data = np.array(load_csv_hourly(path), dtype=float)
    if data.size == 0:
        return data

    if data.shape[1] != FEATURE_CONFIG["expected_features"]:
        if data.shape[1] > FEATURE_CONFIG["expected_features"]:
            data = data[:, : FEATURE_CONFIG["expected_features"]]
        else:
            padding = np.zeros(
                (data.shape[0], FEATURE_CONFIG["expected_features"] - data.shape[1])
            )
            data = np.hstack([data, padding])

    data = sanitize_features(data)

    return data


@validate_and_align_features
def load_batch(batch_data: np.ndarray) -> torch.Tensor:
    """Return ``batch_data`` as a float tensor."""

    batch_data = sanitize_features(batch_data)
    return torch.tensor(batch_data, dtype=torch.float32)
