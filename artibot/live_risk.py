"""Live trading risk checks and auto-pause helpers."""

from __future__ import annotations

import time

import artibot.globals as G


def update_auto_pause(sharpe: float, drawdown: float, ts: float | None = None) -> bool:
    """Record the latest live metrics and update ``trading_paused`` state."""

    if ts is None:
        ts = time.time()

    G.live_sharpe_history.append((ts, float(sharpe)))
    G.live_drawdown_history.append((ts, float(drawdown)))

    seven_days_ago = ts - 7 * 86400
    day_ago = ts - 86400

    sharpes = [s for t, s in G.live_sharpe_history if t >= seven_days_ago]
    dds = [d for t, d in G.live_drawdown_history if t >= day_ago]

    pause = False
    if sharpes and all(s < 1.0 for s in sharpes):
        pause = True
    if any(d < -0.10 for d in dds):
        pause = True

    G.trading_paused = pause
    return pause
