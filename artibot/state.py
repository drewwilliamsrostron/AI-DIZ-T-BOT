# artibot/state.py
"""Light-weight training-state loader/saver.

Features
--------
* Always initialises ``G.global_best_composite_reward`` to -∞ when the value is
  missing from a checkpoint, so comparisons in ``ensemble.train_one_epoch`` can
  never raise a ``TypeError``.
* Optional `save()` helper for symmetry with `load()`.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any

import artibot.globals as G


def load(path: str | os.PathLike = "checkpoint.json") -> dict[str, Any]:
    """Load checkpoint *path* and update globals.

    If the file is absent or unreadable an **empty dict** is returned.
    The global best reward is forced to ``-inf`` when the key is missing.
    """
    try:
        with open(path, "r") as fh:
            state: dict[str, Any] = json.load(fh)
    except OSError:
        return {}

    G.global_best_composite_reward = state.get("best_reward", float("-inf"))
    if not state.get("global_best_full_data", False):
        logging.info(
            "Checkpoint best came from partial data; will be replaced once a full-data run beats it."
        )
    return state


def save(state: dict[str, Any], path: str | os.PathLike = "checkpoint.json") -> None:
    """Persist *state* to *path* as prettified JSON."""
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2)
