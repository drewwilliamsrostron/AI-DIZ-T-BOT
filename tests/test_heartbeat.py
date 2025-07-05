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

from artibot.core.device import DEVICE
from artibot.utils import heartbeat


def test_device_constant():
    assert getattr(DEVICE, "type", str(DEVICE)) in {"cuda", "cpu"}


def test_heartbeat_logging(monkeypatch):
    logged = {}

    def fake_info(msg, *a, **k):
        logged["beat"] = True

    monkeypatch.setattr("logging.info", fake_info)
    heartbeat.start(interval=0.1)
    time.sleep(0.25)
    assert logged.get("beat")
