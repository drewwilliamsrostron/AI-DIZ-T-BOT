import time
import sys
import types

# ruff: noqa: E402

sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda *a, **k: types.SimpleNamespace(type="cpu"),
    ),
)
sys.modules.setdefault(
    "matplotlib",
    types.SimpleNamespace(use=lambda *a, **k: None),
)
sys.modules.setdefault(
    "nvidia",
    types.SimpleNamespace(
        nvml=types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda idx: object(),
            nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=0),
            nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=0, total=1),
            nvmlShutdown=lambda: None,
        )
    ),
)

from artibot.utils import heartbeat


def test_heartbeat_logging(monkeypatch):
    logged = {}

    def fake_info(msg, *a, **k):
        logged["beat"] = True

    monkeypatch.setattr("logging.info", fake_info)
    heartbeat.start(interval=0.1)
    time.sleep(0.25)
    assert logged.get("beat")
