"""Utility helpers for feature dimension management."""

from __future__ import annotations

import numpy as np
import torch

from config import FEATURE_CONFIG


class FeatureDimensionError(Exception):
    """Raised when feature dimension mismatches cannot be resolved."""


def sanitize_features(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return ``x`` with NaN and Inf values replaced."""
    if torch.is_tensor(x):
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=torch.finfo(x.dtype).max,
            neginf=torch.finfo(x.dtype).min,
        )
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=np.finfo(x.dtype).max,
        neginf=np.finfo(x.dtype).min,
    )


def validate_and_align_features(fn):
    """Decorator validating feature dimension and cleaning values."""

    def wrapper(*args, **kwargs):
        x = kwargs.get("x", args[1] if len(args) > 1 else None)
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            return fn(*args, **kwargs)

        expected = FEATURE_CONFIG["expected_features"]
        current = x.shape[-1]
        if current != expected:
            if current > expected:
                x = x[..., :expected]
            else:
                pad_shape = (*x.shape[:-1], expected - current)
                pad = (
                    torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
                    if torch.is_tensor(x)
                    else np.zeros(pad_shape, dtype=x.dtype)
                )
                if torch.is_tensor(x):
                    x = torch.cat([x, pad], dim=-1)
                else:
                    x = np.concatenate([x, pad], axis=-1)

        x = sanitize_features(x)
        return fn(args[0], x, *args[2:], **kwargs)

    return wrapper


__all__ = ["FeatureDimensionError", "sanitize_features", "validate_and_align_features"]
