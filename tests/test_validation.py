import threading
import sys
import types
from importlib.machinery import ModuleSpec

# ruff: noqa: E402

for name in ["torch", "pandas"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

import artibot.globals as G
from artibot.validation import gate_nuclear_key, schedule_monthly_validation


def test_gate_nuclear_key():
    G.nuclear_key_enabled = True
    gate_nuclear_key([-1.0] * 10, threshold=0.0)
    assert not G.nuclear_key_enabled
    gate_nuclear_key([1.0] * 10, threshold=0.0)
    assert G.nuclear_key_enabled


def test_schedule_monthly_validation():
    timer = schedule_monthly_validation("x", {}, interval=1.0)
    assert isinstance(timer, threading.Timer)
    assert timer.interval == 1.0
    timer.cancel()
