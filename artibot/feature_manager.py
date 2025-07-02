"""Utility helpers for feature dimension management."""

from __future__ import annotations

import numpy as np
import torch

from .constants import FEATURE_DIMENSION


class FeatureDimensionError(Exception):
    """Raised when feature dimension mismatches cannot be resolved."""


def sanitize_features(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return ``x`` with NaN and Inf values replaced."""
    if torch.is_tensor(x):
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def align_features(x: np.ndarray, expected: int) -> np.ndarray:
    """Ensure ``x`` has ``expected`` feature columns."""

    current = x.shape[1]
    if current != expected:
        raise FeatureDimensionError(f"Expected {expected} features, got {current}")
    return x


def enforce_feature_dim(x: np.ndarray, expected: int = FEATURE_DIMENSION) -> np.ndarray:
    """Return ``x`` padded with zeros to ``expected`` columns.

    If ``x`` already has ``expected`` or more columns, it is returned
    unchanged.  Arrays with fewer than ``expected`` columns are extended
    with ``0.0`` columns.  This function is written with NumPy 2.x
    compatibility in mind and avoids deprecated ``np.pad`` behaviour.
    """

    cols = x.shape[1]
    if cols >= expected:
        return x

    pad = np.zeros((x.shape[0], expected - cols), dtype=x.dtype)
    return np.concatenate((x, pad), axis=1)


def validate_and_align_features(fn):
    """Decorator validating feature dimension and cleaning values."""

    def wrapper(*args, **kwargs):
        x = kwargs.get("x", args[1] if len(args) > 1 else None)
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            return fn(*args, **kwargs)

        expected = FEATURE_DIMENSION
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
    "enforce_feature_dim",
    "validate_and_align_features",
]
