"""Helpers for submitting orders with random delay and slippage."""

from __future__ import annotations

import random
import time
from typing import Any, Callable


def submit_order(
    func: Callable[..., Any],
    side: str,
    amount: float,
    price: float,
    *,
    delay: float | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``func`` with a jittered price after a short delay."""
    if delay is None:
        delay = max(0.0, random.normalvariate(0.1, 0.05))
    time.sleep(delay)
    adj_price = price + random.uniform(-0.0005, 0.0005)
    return func(side=side, amount=amount, price=adj_price, **kwargs)
