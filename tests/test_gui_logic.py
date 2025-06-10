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


def test_should_enable_live_trading():
    G.global_holdout_sharpe = 1.2
    G.global_holdout_max_drawdown = -0.1
    assert gui.should_enable_live_trading() is True
    G.global_holdout_sharpe = 0.5
    assert gui.should_enable_live_trading() is False
