"""Simplified public API for the :mod:`artibot` package.

Importing :mod:`artibot` now ensures that required third-party packages are
available.  This mirrors the behavior of the legacy single-file script where
the installer ran once on first launch.
"""

# ruff: noqa: E402
from .environment import ensure_dependencies

# Make sure third‑party packages like ``torch`` are present before importing
# modules that rely on them.  This mirrors the behavior of the original
# single‑file script where packages were installed on first launch.
ensure_dependencies()  # run the installer once at import time

try:
    from .core.device import check_cuda
except Exception:  # pragma: no cover - optional torch missing

    def check_cuda() -> None:
        pass


check_cuda()
from config import FEATURE_CONFIG
from .constants import FEATURE_DIMENSION

import logging

_LOG = logging.getLogger(__name__)

_LOG.debug("[INIT] System configured for %s features", FEATURE_DIMENSION)
_LOG.debug("[INIT] Feature columns: %s", ", ".join(FEATURE_CONFIG["feature_columns"]))

try:
    from screeninfo import get_monitors
except Exception:  # pragma: no cover - optional dep missing

    def get_monitors() -> list:
        """Fallback when :mod:`screeninfo` is unavailable."""

        return []


try:
    from . import globals as G
except Exception:  # pragma: no cover - optional dep missing

    class _Dummy:
        UI_SCALE = 1.0

    G = _Dummy()


def _detect_scale(baseline: float = 96.0) -> float:
    try:
        mon = get_monitors()[0]
        dpi = mon.width * 25.4 / mon.width_mm
    except Exception:
        dpi = baseline
    return max(0.9, min(2.0, dpi / baseline))


G.UI_SCALE = _detect_scale()


from .environment import *  # noqa: F401,F403 - re-export environment helpers


def run_bot(*args, **kwargs):
    """Lazy import :func:`bot_app.run_bot` to ensure deps are loaded."""
    from .bot_app import run_bot as _run_bot

    return _run_bot(*args, **kwargs)


__all__ = ["run_bot"]
