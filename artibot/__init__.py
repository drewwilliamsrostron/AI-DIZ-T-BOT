"""Simplified public API for the :mod:`artibot` package."""

from .environment import *  # noqa: F401,F403 - re-export environment helpers

def run_bot(*args, **kwargs):
    """Lazy import :func:`bot_app.run_bot` to ensure deps are loaded."""
    from .bot_app import run_bot as _run_bot
    return _run_bot(*args, **kwargs)

__all__ = ["run_bot"]
