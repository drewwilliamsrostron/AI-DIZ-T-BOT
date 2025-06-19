"""Utility helpers for device selection and logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
import json

import math
import torch
import pandas as pd
import numpy as np

from ..hyperparams import IndicatorHyperparams

CONTEXT_FLAGS = ("use_sentiment", "use_macro", "use_rvol")


def get_device() -> torch.device:
    """Return a CUDA device when available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return json.dumps(base)


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


def rolling_zscore(arr: np.ndarray, window: int = 50) -> np.ndarray:
    """Return rolling z-score normalised ``arr``.

    Parameters
    ----------
    arr:
        2D array of features to be standardised.
    window:
        Rolling window size used for mean and standard deviation.
    """

    df = pd.DataFrame(arr, dtype=float)
    roll_mean = df.rolling(window=window, min_periods=1).mean()
    roll_std = df.rolling(window=window, min_periods=1).std().replace(0, 1e-8)
    scaled = ((df - roll_mean) / roll_std).to_numpy(dtype=float)
    scaled = np.clip(scaled, -50.0, 50.0)
    return np.nan_to_num(scaled).astype(np.float32)


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


__all__ = ["rolling_zscore", "feature_dim_for"]
