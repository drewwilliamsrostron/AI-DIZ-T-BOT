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

from .torch_threads import set_threads


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
        }
        return json.dumps(base)


def setup_logging() -> None:
    """Configure root logger to emit JSON-formatted messages."""
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


__all__ = [
    "get_device",
    "setup_logging",
    "attention_entropy",
    "rolling_zscore",
    "set_threads",
]
