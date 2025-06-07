"""Simplified public API for the :mod:`artibot` package.

Importing :mod:`artibot` now ensures that required third-party packages are
available.  This mirrors the behavior of the legacy single-file script where
the installer ran once on first launch.
"""

from __future__ import annotations

from .environment import ensure_dependencies

ensure_dependencies()  # run the installer once at import time

from .environment import *  # noqa: F401, F403 - re-export environment helpers


def run_bot(*args, **kwargs):
    """Lazy import :func:`bot_app.run_bot` to ensure deps are loaded."""
    from .bot_app import run_bot as _run_bot

    return _run_bot(*args, **kwargs)


__all__ = ["run_bot"]
