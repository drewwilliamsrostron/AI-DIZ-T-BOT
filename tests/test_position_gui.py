# ruff: noqa: E402
import types
import sys
from importlib.machinery import ModuleSpec

# stub heavy modules
for name in ["openai", "ccxt"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

class DummyWidget:
    def __init__(self, *a, **k):
        self.attrs = {"state": "normal"}
    def pack(self, *a, **k):
        pass
    def grid(self, *a, **k):
        pass
    def config(self, **kw):
        self.attrs.update(kw)
    def __setitem__(self, key, val):
        self.attrs[key] = val
    def __getitem__(self, key):
        return self.attrs.get(key)

class DummyTk(DummyWidget):
    def after(self, *a, **k):
        pass
    def title(self, t):
        pass

# tkinter stubs
TkMod = types.ModuleType("tkinter")
TkMod.Tk = DummyTk
TkMod.Label = DummyWidget
TkMod.Frame = DummyWidget
TkMod.LabelFrame = DummyWidget
TkMod.Text = DummyWidget
TkMod.Toplevel = DummyTk
TkMod.StringVar = lambda *a, **k: None
TkMod.DoubleVar = lambda *a, **k: None
sys.modules["tkinter"] = TkMod

TtkMod = types.ModuleType("tkinter.ttk")
for n in ["Frame", "Label", "LabelFrame", "Button", "Progressbar", "Notebook", "Scrollbar", "Treeview"]:
    setattr(TtkMod, n, DummyWidget)
sys.modules["tkinter.ttk"] = TtkMod

# matplotlib stubs
matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules["matplotlib"] = matplotlib
plt = types.ModuleType("matplotlib.pyplot")
plt.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
plt.subplots = lambda *a, **k: (object(), (object(), object()))
sys.modules["matplotlib.pyplot"] = plt
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = lambda *a, **k: DummyWidget()
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules["matplotlib.backends.backend_tkagg"] = backend

from artibot.gui import TradingGUI

ens = types.SimpleNamespace(optimizers=[types.SimpleNamespace(param_groups=[{"lr": 0.001}])])
ui = TradingGUI.__new__(TradingGUI)
ui.btn_buy = DummyWidget()
ui.btn_sell = DummyWidget()
ui.btn_close = DummyWidget()
ui.label_side = DummyWidget()
ui.label_size = DummyWidget()
ui.label_entry = DummyWidget()


def test_update_position_buttons():
    ui.update_position("LONG", 2, 100.0)

    assert ui.label_side["text"] == "LONG"
    assert ui.label_size["text"] == "2"
    assert ui.label_entry["text"] == "100.00"
    assert ui.btn_buy["state"] == "disabled"
    assert ui.btn_sell["state"] == "disabled"
    assert ui.btn_close["state"] == "normal"

