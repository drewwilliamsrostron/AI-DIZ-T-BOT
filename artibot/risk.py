"""Simple risk utilities for sizing positions."""

from __future__ import annotations


def position_size(
    balance: float,
    risk_fraction: float,
    stop_distance: float,
    price: float,
    leverage: float,
) -> float:
    """Return max position size given risk parameters."""
    risk_cap = balance * risk_fraction
    pos_size_risk = risk_cap / (stop_distance + 1e-8)
    max_sz = (balance * leverage) / price
    return min(pos_size_risk, max_sz)
