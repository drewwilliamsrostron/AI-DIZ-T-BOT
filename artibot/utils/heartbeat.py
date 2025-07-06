"""Thread-safe liveness heartbeat."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

try:  # optional dependency
    import psutil
except Exception:  # pragma: no cover - optional dep not installed
    psutil = None

_LOCK = threading.Lock()

LATEST_STATS: dict[str, Any] = {}


def update(**kwargs: Any) -> None:
    """Store the latest training stats for the heartbeat."""

    with _LOCK:
        LATEST_STATS.update(kwargs)


def start(interval: int = 120) -> None:
    """Start a background heartbeat logger."""

    def _beat() -> None:
        while True:
            with _LOCK:
                stats = {
                    "event": "HEARTBEAT",
                    "ts": time.ctime(),
                    "epoch": LATEST_STATS.get("epoch"),
                    "candidate": LATEST_STATS.get("candidate"),
                    "best_sharpe": LATEST_STATS.get("best_sharpe"),
                    "cpu": f"{psutil.cpu_percent()}%" if psutil else "0%",
                    "mem": f"{psutil.virtual_memory().percent}%" if psutil else "0%",
                }
            logging.info(json.dumps(stats))
            time.sleep(interval)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
