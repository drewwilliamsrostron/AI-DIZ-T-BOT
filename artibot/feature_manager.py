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


def align_features(x: np.ndarray, expected: int) -> np.ndarray:
    """Ensure ``x`` has ``expected`` feature columns."""

    current = x.shape[1]
    if current != expected:
        raise FeatureDimensionError(f"Expected {expected} features, got {current}")
    return x


def validate_and_align_features(fn):
    """Decorator validating feature dimension and cleaning values."""

    def wrapper(*args, **kwargs):
        x = kwargs.get("x", args[1] if len(args) > 1 else None)
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            return fn(*args, **kwargs)

        expected = FEATURE_CONFIG["expected_features"]
        current = x.shape[-1]
        if current != expected:
            raise FeatureDimensionError(f"Expected {expected} features, got {current}")

        x = sanitize_features(x)
        return fn(args[0], x, *args[2:], **kwargs)

    return wrapper


__all__ = [
    "FeatureDimensionError",
    "sanitize_features",
    "align_features",
    "validate_and_align_features",
]
