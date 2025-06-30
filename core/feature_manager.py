import torch
import numpy as np
from config import FEATURE_CONFIG


class FeatureDimensionError(Exception):
    """Custom exception for feature dimension issues."""


def sanitize_features(x):
    """Aggressive NaN/Inf handling with type preservation."""
    if torch.is_tensor(x):
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=torch.finfo(x.dtype).max,
            neginf=torch.finfo(x.dtype).min,
        )
    else:
        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=np.finfo(x.dtype).max,
            neginf=np.finfo(x.dtype).min,
        )
    return x


def validate_and_align_features(fn):
    """Decorator to automatically validate and align feature dimensions."""

    def wrapper(*args, **kwargs):
        x = kwargs.get("x", args[1] if len(args) > 1 else None)
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            return fn(*args, **kwargs)

        current_features = x.shape[-1]
        expected = FEATURE_CONFIG["expected_features"]

        if current_features != expected:
            if current_features > expected:
                x = x[..., :expected]
            else:
                pad_shape = (*x.shape[:-1], expected - current_features)
                if torch.is_tensor(x):
                    pad = torch.zeros(pad_shape, device=x.device)
                    x = torch.cat([x, pad], dim=-1)
                else:
                    pad = np.zeros(pad_shape, dtype=x.dtype)
                    x = np.concatenate([x, pad], axis=-1)

        x = sanitize_features(x)

        return fn(args[0], x, *args[2:], **kwargs)

    return wrapper

