"""Simple risk filter wrappers."""

from __future__ import annotations

import config


def apply_risk_filter(signals):
    """Return ``signals`` unchanged when the filter is disabled."""
    if not getattr(config, "RISK_FILTER", True):
        return signals
    # Placeholder for real logic
    return signals
