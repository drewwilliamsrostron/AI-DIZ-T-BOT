"""Simplified public API for the :mod:`artibot` package."""

from .environment import *  # noqa: F401,F403 - re-export environment helpers
from .bot_app import run_bot

__all__ = ["run_bot"]
