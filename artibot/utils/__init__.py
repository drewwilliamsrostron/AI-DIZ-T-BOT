"""Utility helpers for device selection and logging."""

from __future__ import annotations

import logging
import sys
import json

import torch


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
        }
        return json.dumps(base)


def setup_logging() -> None:
    """Configure root logger to emit JSON-formatted messages."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def attention_entropy(weights: torch.Tensor) -> float:
    """Return mean entropy of attention probabilities or logits."""
    if weights.min() < 0 or weights.max() > 1:
        probs = torch.softmax(weights, dim=-1)
    else:
        probs = weights
    probs = torch.nan_to_num(probs)
    ent = (-probs * torch.log(probs + 1e-9)).sum(-1).mean().item()
    return float(ent)
