"""Lightweight training state loader and saver."""

from __future__ import annotations

import json
import os
from typing import Any

from . import globals as G


def load(path: str | os.PathLike = "checkpoint.json") -> dict[str, Any]:
    """Load checkpoint ``path`` and update globals.

    The file is optional; missing files return an empty dict.
    ``G.global_best_composite_reward`` defaults to ``-inf`` when the key
    is absent.
    """
    try:
        with open(path, "r") as fh:
            state = json.load(fh)
    except OSError:
        return {}

    G.global_best_composite_reward = state.get("best_reward", float("-inf"))
    return state


def save(state: dict[str, Any], path: str | os.PathLike = "checkpoint.json") -> None:
    """Persist ``state`` to ``path``."""
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2)
