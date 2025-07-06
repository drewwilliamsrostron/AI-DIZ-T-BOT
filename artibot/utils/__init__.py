"""Utility helpers for device selection and logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
import json

import math
import torch
from ..core.device import get_device as _core_get_device
import pandas as pd
import numpy as np
import hashlib

from ..hyperparams import IndicatorHyperparams
from ..constants import FEATURE_DIMENSION


# [FIX]#
def clean_features(features: np.ndarray, replace_value: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf values with specified replacement."""

    features = np.nan_to_num(
        features, nan=replace_value, posinf=replace_value, neginf=replace_value
    )
    return features


CONTEXT_FLAGS = ("use_sentiment", "use_macro", "use_rvol")


def get_device() -> torch.device:
    """Return detected hardware device."""
    return _core_get_device()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "msg": record.getMessage(),
            "epoch": getattr(record, "epoch", None),
            "sharpe": getattr(record, "sharpe", None),
            "max_dd": getattr(record, "max_dd", None),
            "attn_entropy": getattr(record, "attn_entropy", None),
            "lr": getattr(record, "lr", None),
            "profit_factor": getattr(record, "profit_factor", None),
            "loss": getattr(record, "loss", None),
            "val": getattr(record, "val", None),
        }
        clean = {k: v for k, v in base.items() if v is not None}
        return json.dumps(clean)


def setup_logging() -> None:
    """Configure root logger to emit JSON-formatted messages.

    The console stream handler now mirrors the file handlers so that
    ``logging.info`` messages are visible both in ``bot.log`` and on
    ``stdout``.
    """
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())

    file_handler = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(JsonFormatter())

    debug_handler = RotatingFileHandler(
        "training_debug.log", maxBytes=5_000_000, backupCount=3
    )
    debug_handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, (logging.StreamHandler, RotatingFileHandler)):
            root.removeHandler(h)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    root.addHandler(debug_handler)
    root.setLevel(logging.INFO)


def attention_entropy(tensor: torch.Tensor) -> float:
    """Return mean entropy of attention probabilities or logits."""
    if tensor.max() <= 1 and tensor.min() >= 0:
        p = tensor
    else:
        tensor = tensor - tensor.max(dim=-1, keepdim=True).values
        p = tensor.softmax(dim=-1)
    p = torch.nan_to_num(p)
    ent = (-p * (p + 1e-9).log()).sum(-1).mean().item()
    if not math.isfinite(ent):
        ent = 0.0
    max_ent = math.log(p.size(-1))
    return min(ent, max_ent)


def rolling_zscore(
    arr: np.ndarray, window: int = 50, mask: np.ndarray | None = None
) -> np.ndarray:
    """Return rolling z-score normalised ``arr``.

    Parameters
    ----------
    arr:
        2D array of features to be standardised.
    window:
        Rolling window size used for mean and standard deviation.
    """

    df = pd.DataFrame(arr, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        cols = [i for i, m in enumerate(mask) if m]
        roll_mean = df.iloc[:, cols].rolling(window=window, min_periods=1).mean()
        roll_std = (
            df.iloc[:, cols]
            .rolling(window=window, min_periods=1)
            .std()
            .replace(0, 1e-8)
        )
        df.iloc[:, cols] = (df.iloc[:, cols] - roll_mean) / roll_std
    else:
        roll_mean = df.rolling(window=window, min_periods=1).mean()
        roll_std = df.rolling(window=window, min_periods=1).std().replace(0, 1e-8)
        df = (df - roll_mean) / roll_std
    scaled = df.to_numpy(dtype=float)
    scaled = np.clip(scaled, -50.0, 50.0)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def feature_dim_for(indicators: "IndicatorHyperparams") -> int:
    """Return the expected feature count for a *specific* flag set.

    This lets the dataloader, the ensemble and the model builder agree
    on an identical dimension *without* relying on global state.
    """

    flags = [
        indicators.use_atr,
        indicators.use_vortex,
        indicators.use_cmf,
        indicators.use_rsi,
        indicators.use_sma,
        indicators.use_macd,
        indicators.use_ema,
        indicators.use_donchian,
        indicators.use_kijun,
        indicators.use_tenkan,
        indicators.use_displacement,
        getattr(indicators, "use_ichimoku", False),
        # ✏️  add any new flags here
    ]
    return 4 + sum(flags)  # 4 core OHLC bars + extras


def active_feature_dim(hp: IndicatorHyperparams, *, use_ichimoku: bool = False) -> int:
    """Return feature column count for given indicator flags."""

    dim = 8  # ohlcv + sentiment + macro + realised vol
    if hp.use_sma and not use_ichimoku:
        dim += 1
    if hp.use_rsi:
        dim += 1
    if hp.use_macd:
        dim += 1
    if hp.use_ema and not use_ichimoku:
        dim += 1
    if hp.use_atr and not use_ichimoku:
        dim += 1
    if hp.use_vortex:
        dim += 2
    if hp.use_cmf:
        dim += 1
    if getattr(hp, "use_donchian", False):
        dim += 3
    if getattr(hp, "use_kijun", False):
        dim += 1
    if getattr(hp, "use_tenkan", False):
        dim += 1
    if getattr(hp, "use_displacement", False):
        dim += 1
    if use_ichimoku:
        dim += 4
    return dim


def feature_mask_for(
    hp: IndicatorHyperparams, *, use_ichimoku: bool = False
) -> np.ndarray:
    """Return boolean mask for enabled feature columns."""

    dim = FEATURE_DIMENSION
    mask = np.ones(dim, dtype=bool)

    idx_map = {
        5: hp.use_sma,
        6: hp.use_rsi,
        7: hp.use_macd,
        8: hp.use_ema,
        10: hp.use_atr,
        11: hp.use_vortex,
        12: hp.use_vortex,
        13: hp.use_cmf,
        14: hp.use_donchian,
        15: use_ichimoku,
    }
    for idx, enabled in idx_map.items():
        if idx < dim:
            mask[idx] = bool(enabled)
    return mask


def feature_dim_for(hp: "IndicatorHyperparams") -> int:  # noqa: F811
    """Return the feature dimension implied by *hp*.

    • 5 OHLCV bars are always present
    • +1 for each context flag that is True
    • +1 for every technical-indicator flag that is True
    """

    base = 5  # OHLCV
    for flag in CONTEXT_FLAGS:
        if getattr(hp, flag):
            base += 1

    extras = sum(
        [
            hp.use_atr,
            hp.use_vortex,
            hp.use_cmf,
            hp.use_rsi,
            hp.use_sma,
            hp.use_macd,
            hp.use_ema,
            hp.use_donchian,
            hp.use_kijun,
            hp.use_tenkan,
            hp.use_displacement,
            getattr(hp, "use_ichimoku", False),
        ]
    )
    return base + extras


def feature_version_hash(arr: np.ndarray) -> str:
    """Return md5 hash of the given feature array."""

    return hashlib.md5(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    ).hexdigest()


# [FIX]#
class DimensionError(Exception):
    """Raised when feature matrices do not match the expected shape."""


def validate_features(feat: np.ndarray, enabled_mask: np.ndarray | None = None) -> None:
    """Validate ``feat`` using ``enabled_mask``.

    Raises :class:`DimensionError` when:
      * ``feat`` is not a 2‑D array
      * ``feat.shape[1]`` does not match ``len(enabled_mask)``
      * any active column contains NaN/Inf or has zero variance
    """

    if feat.ndim != 2:
        raise DimensionError("Features must be 2-D")

    if enabled_mask is None:
        mask = np.ones(feat.shape[1], dtype=bool)
    else:
        mask = np.asarray(enabled_mask, dtype=bool)
        if feat.shape[1] != mask.size:
            raise DimensionError("Feature mask size mismatch")
        if not mask.any():
            raise DimensionError("No active features after mask")

    active = feat[:, mask]
    if not np.isfinite(active).all():
        raise DimensionError("NaN or Inf detected in features")

    ranges = np.ptp(active, axis=0)
    has_var = ranges > 0
    if (~has_var).any():
        for idx, var in enumerate(has_var):
            if not var and not np.allclose(active[:, idx], 0.0):
                raise DimensionError("Zero variance feature detected")
    if has_var.any():
        return
    raise DimensionError("All active features have zero variance")


def validate_feature_dimension(
    features: np.ndarray, expected: int, logger: logging.Logger
) -> np.ndarray:
    """Log a feature-dimension mismatch without modifying ``features``."""

    current = features.shape[1]
    if current != expected:
        logger.error("Feature mismatch! Expected %s, got %s", expected, current)
    return features


def enforce_feature_dim(
    features: np.ndarray, expected: int = FEATURE_DIMENSION
) -> np.ndarray:
    """Return ``features`` padded with zeros when dimension mismatches."""

    if features.shape[1] != expected:
        corrected = np.zeros((features.shape[0], expected), dtype=features.dtype)
        cols = min(features.shape[1], expected)
        corrected[:, :cols] = features[:, :cols]
        return corrected
    return features


def zero_disabled(
    features: np.ndarray | torch.Tensor, enabled_mask
) -> np.ndarray | torch.Tensor:
    """Return ``features`` with disabled columns zeroed."""

    if torch.is_tensor(features):
        if not torch.is_tensor(enabled_mask):
            mask = torch.as_tensor(
                enabled_mask, device=features.device, dtype=torch.bool
            )
        else:
            mask = enabled_mask.to(features.device, dtype=torch.bool)

        while mask.dim() < features.dim():
            mask = mask.unsqueeze(0)

        out = features.clone()
        out.masked_fill_(~mask, 0)
        return out

    mask = np.asarray(enabled_mask, dtype=bool)
    reshape = [1] * features.ndim
    reshape[-1] = mask.size
    mask = mask.reshape(reshape)

    return np.where(mask, features, 0.0)


__all__ = [
    "rolling_zscore",
    "feature_mask_for",
    "feature_dim_for",
    "feature_version_hash",
    "clean_features",
    "validate_features",
    "validate_feature_dimension",
    "enforce_feature_dim",
    "DimensionError",
    "zero_disabled",
]
