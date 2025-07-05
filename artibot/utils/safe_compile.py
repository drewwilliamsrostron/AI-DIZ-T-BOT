"""Utility to safely call ``torch.compile`` where supported."""

import platform
import logging
import sys
import torch


def safe_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Return ``torch.compile(model)`` except on unsupported platforms."""

    if (
        hasattr(torch, "compile")
        and sys.version_info < (3, 12)
        and platform.system() != "Windows"
    ):
        try:
            return torch.compile(model)  # type: ignore[arg-type]
        except RuntimeError as e:  # pragma: no cover - compile not always available
            logging.warning("torch.compile disabled: %s", e)
    return model
