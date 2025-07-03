"""Live trading risk checks and auto-pause helpers."""

from __future__ import annotations

import time

from artibot.metrics import DAY_SEC

import artibot.globals as G


def update_auto_pause(sharpe: float, drawdown: float, ts: float | None = None) -> bool:
    """Record the latest live metrics and update ``trading_paused`` state."""

    if ts is None:
        ts = time.time()

    G.live_sharpe_history.append((ts, float(sharpe)))
    G.live_drawdown_history.append((ts, float(drawdown)))

    seven_days_ago = ts - 7 * DAY_SEC
    day_ago = ts - DAY_SEC

    sharpes = [s for t, s in G.live_sharpe_history if t >= seven_days_ago]
    dds = [d for t, d in G.live_drawdown_history if t >= day_ago]

    pause = False
    if (
        not hasattr(G, "daily_equity_start")
        or ts - getattr(G, "daily_equity_start", 0) > DAY_SEC
    ):
        G.daily_equity_start = ts
        G.daily_equity_reference = G.live_equity
    if sharpes and all(s < 1.0 for s in sharpes):
        pause = True
    if any(d < -0.10 for d in dds):
        pause = True
    if G.live_equity < G.start_equity * 0.8:
        pause = True
    if G.live_equity < getattr(G, "daily_equity_reference", G.live_equity) * 0.98:
        pause = True

    G.trading_paused = pause
    return pause
