"""Compatibility shim exposing the new GUI implementation.

This module remains for backward compatibility and will be removed in a
future release. Import :mod:`artibot.gui_v2` directly instead.
"""

from .gui_v2 import *  # noqa: F401,F403
