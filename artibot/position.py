"""Utilities for tracking and executing trading positions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .training import PhemexConnector


@dataclass
class Position:
    """Represents an open trade."""

    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_time: Optional[float] = None


def open_position(
    connector: "PhemexConnector",
    side: str,
    amount: float,
    price: float,
    stop_loss: float,
    take_profit: float,
) -> Position:
    """Place an order via ``connector`` and return a ``Position``."""

    connector.create_order(side, amount, price)
    return Position(
        side=side,
        size=amount if side == "long" else -amount,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_time=time.time(),
    )


def close_position(connector: "PhemexConnector", pos: Position, price: float) -> None:
    """Submit an exit order and reset ``pos`` in place."""

    exit_side = "sell" if pos.side == "long" else "buy"
    connector.create_order(exit_side, abs(pos.size), price)
    pos.side = None
    pos.size = 0.0
