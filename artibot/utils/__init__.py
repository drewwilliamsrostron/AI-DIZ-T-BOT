"""Utility helpers for device selection and logging."""

from __future__ import annotations

import logging
import sys
import json

import math
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
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            root.removeHandler(h)
    root.addHandler(handler)


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
