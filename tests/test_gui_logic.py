# ruff: noqa: E402
import types
import sys
from importlib.machinery import ModuleSpec

# stub heavy modules before import
for name in ["openai", "ccxt", "tkinter", "tkinter.ttk"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules.setdefault("matplotlib", matplotlib)
pyplot = types.ModuleType("pyplot")
pyplot.__spec__ = ModuleSpec("pyplot", loader=None)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = object
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules.setdefault("matplotlib.backends.backend_tkagg", backend)

from artibot import gui, globals as G  # noqa: E402
import artibot.bot_app as bot_app


def _patch_config(min_pnl, min_trades):
    bot_app.CONFIG["NK_MIN_PNL"] = min_pnl
    bot_app.CONFIG["NK_MIN_TRADES"] = min_trades


def test_should_enable_live_trading():
    _patch_config(10.0, 2)
    G.global_holdout_sharpe = 1.2
    G.global_holdout_max_drawdown = -0.1
    G.start_equity = 100.0
    G.live_equity = 115.0
    G.live_trade_count = 3
    assert gui.should_enable_live_trading() is True
    G.live_equity = 105.0
    assert gui.should_enable_live_trading() is False


def test_manual_validation_button(monkeypatch):
    bot_app.CONFIG["CSV_PATH"] = "data.csv"
    called = {}

    def fake_validate(path, cfg):
        called["path"] = path
        called["cfg"] = cfg

    monkeypatch.setattr("artibot.validation.validate_and_gate", fake_validate)

    class DummyThread:
        def __init__(self, target, daemon=None):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(gui.threading, "Thread", DummyThread)

    ui = gui.TradingGUI.__new__(gui.TradingGUI)
    ui.root = types.SimpleNamespace(after=lambda *a, **k: None)
    ui.validation_label = types.SimpleNamespace(
        config=lambda **kw: called.setdefault("label", kw)
    )
    ui.validate_button = types.SimpleNamespace(
        config=lambda **kw: called.setdefault("btn", kw)
    )

    ui.manual_validate()

    assert called["path"].endswith("data.csv")
    assert called["btn"]["state"] == "disabled"
