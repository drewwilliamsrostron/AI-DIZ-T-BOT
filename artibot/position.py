"""Trade leg and hedging helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import artibot.globals as G

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .training import PhemexConnector


@dataclass
class TradeLeg:
    """Represents one side of a hedged position."""

    side: str
    size: float
    entry_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_time: float | None = None


@dataclass
class Position(TradeLeg):
    """Backward compatible position alias."""


@dataclass
class HedgeBook:
    """Tracks independent long and short legs."""

    long_leg: TradeLeg | None = None
    short_leg: TradeLeg | None = None

    @staticmethod
    def _usd_to_contracts(usd: float, price: float) -> int:
        """Convert USD exposure to contract quantity."""
        if price <= 0:
            return 0
        return int(usd / price)

    def open_long(self, connector: "PhemexConnector", price: float, hp) -> None:
        if hp.long_frac == 0:
            self.close_long(connector, price)
            return
        usd = hp.long_frac * G.live_equity
        contracts = self._usd_to_contracts(usd, price)
        if contracts > 0:
            connector.create_order("buy", contracts, price)
        self.long_leg = TradeLeg(
            side="long",
            size=contracts,
            entry_price=price,
            entry_time=time.time(),
        )

    def open_short(self, connector: "PhemexConnector", price: float, hp) -> None:
        if hp.short_frac == 0:
            self.close_short(connector, price)
            return
        usd = hp.short_frac * G.live_equity
        contracts = self._usd_to_contracts(usd, price)
        if contracts > 0:
            connector.create_order("sell", contracts, price)
        self.short_leg = TradeLeg(
            side="short",
            size=contracts,
            entry_price=price,
            entry_time=time.time(),
        )

    def close_long(self, connector: "PhemexConnector", price: float) -> None:
        if not self.long_leg:
            return
        connector.create_order("sell", self.long_leg.size, price)
        self.long_leg = None

    def close_short(self, connector: "PhemexConnector", price: float) -> None:
        if not self.short_leg:
            return
        connector.create_order("buy", self.short_leg.size, price)
        self.short_leg = None


# ---------------------------------------------------------------------------
# Backwards compatibility wrappers matching the previous API
# ---------------------------------------------------------------------------


def open_position(
    connector: "PhemexConnector",
    side: str,
    amount: float,
    price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> TradeLeg:
    if stop_loss is None or take_profit is None:
        sl_m = G.global_SL_multiplier
        tp_m = G.global_TP_multiplier
        if side == "long":
            stop_loss = price - sl_m
            take_profit = price + tp_m
        else:
            stop_loss = price + sl_m
            take_profit = price - tp_m

    connector.create_order(
        side,
        amount,
        price,
        order_type="market",
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    with G.state_lock:
        G.live_trade_count += 1
    return Position(
        side=side,
        size=amount,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_time=time.time(),
    )


def close_position(connector: "PhemexConnector", leg: TradeLeg, price: float) -> None:
    exit_side = "sell" if leg.side == "long" else "buy"
    connector.create_order(exit_side, leg.size, price, order_type="market")
