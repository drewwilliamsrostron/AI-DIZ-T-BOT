"""Thread-safe liveness heartbeat."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import tomllib
import torch
import psutil
from nvidia import nvml

_LOCK = threading.Lock()

LATEST_STATS: dict[str, Any] = {}


def update(**kwargs: Any) -> None:
    """Store the latest training stats for the heartbeat."""

    with _LOCK:
        LATEST_STATS.update(kwargs)


def sample() -> dict[str, Any]:
    """Return CPU, memory and optional GPU utilisation."""

    cpu = f"{psutil.cpu_percent():.1f}%"
    mem = f"{psutil.virtual_memory().percent:.1f}%"
    gpu, vram = None, None
    if torch.cuda.is_available():
        nvml.nvmlInit()
        h = nvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        util = nvml.nvmlDeviceGetUtilizationRates(h)
        meminfo = nvml.nvmlDeviceGetMemoryInfo(h)
        gpu = f"{util.gpu}%"
        vram = f"{meminfo.used / meminfo.total * 100:.1f}%"
        nvml.nvmlShutdown()
    return dict(cpu=cpu, mem=mem, gpu=gpu, vram=vram)


def start(interval: int | None = None) -> None:
    """Start a background heartbeat logger."""

    if interval is None:
        try:
            with open("config/default.toml", "rb") as fh:
                cfg = tomllib.load(fh)
                interval = int(cfg.get("logging", {}).get("heartbeat_interval", 120))
        except Exception:
            interval = 120

    def _beat() -> None:
        while True:
            with _LOCK:
                stats = {
                    "event": "HEARTBEAT",
                    "ts": time.ctime(),
                    **sample(),
                    **LATEST_STATS,
                }
            logging.info(json.dumps(stats))
            time.sleep(interval)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
