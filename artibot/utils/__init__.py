"""Utility helpers for device selection and logging."""

from __future__ import annotations

import logging
import sys

import torch


def get_device() -> torch.device:
    """Return a CUDA device when available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging() -> None:
    """Configure root logger to emit JSON-formatted messages."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
