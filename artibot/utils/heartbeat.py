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

try:
    from nvidia import nvml

    GPU_METRICS = True
except ImportError:  # pragma: no cover - optional dep
    nvml = None
    GPU_METRICS = False

_HANDLE = None

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
    gpu_util = None
    vram_used_mb = None
    if GPU_METRICS and _HANDLE is not None:
        util = nvml.nvmlDeviceGetUtilizationRates(_HANDLE)
        meminfo = nvml.nvmlDeviceGetMemoryInfo(_HANDLE)
        gpu_util = util.gpu
        vram_used_mb = int(meminfo.used / 1024**2)
    return dict(
        cpu=cpu,
        mem=mem,
        gpu_util=gpu_util,
        vram_used_mb=vram_used_mb,
        gpu=gpu_util,  # backward compat
        vram=vram_used_mb,
    )


def start(interval: int | None = None) -> None:
    """Start a background heartbeat logger."""

    global _HANDLE
    if interval is None:
        try:
            with open("config/default.toml", "rb") as fh:
                cfg = tomllib.load(fh)
                interval = int(cfg.get("logging", {}).get("heartbeat_interval", 120))
        except Exception:
            interval = 120

    if GPU_METRICS and torch.cuda.is_available():
        try:
            nvml.nvmlInit()
            _HANDLE = nvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception as exc:  # pragma: no cover - nvml errors
            logging.warning("NVML init failed: %s", exc)
            _HANDLE = None

    def _beat() -> None:
        while True:
            with _LOCK:
                stats = {
                    "event": "HEARTBEAT",
                    "ts": time.ctime(),
                }
                stats.update(sample())
                stats.update(LATEST_STATS)
                stats = {k: v for k, v in stats.items() if v is not None}
            logging.info(json.dumps(stats))
            time.sleep(interval)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
