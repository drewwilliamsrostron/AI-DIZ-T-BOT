# ruff: noqa: E402
import types
import sys
from importlib.machinery import ModuleSpec

# stub heavy modules
for name in ["openai", "ccxt"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

import artibot.globals as G


class DummyWidget:
    def __init__(self, master=None, *a, **k):
        self.master = master
        self.root = getattr(master, "root", master)
        self.attrs = {"state": "normal"}

    def pack(self, *a, **k):
        pass

    def grid(self, *a, **k):
        pass

    def config(self, **kw):
        self.attrs.update(kw)

    def create_window(self, *a, **k):
        pass

    def configure(self, **kw):
        pass

    def bbox(self, *a):
        return (0, 0, 100, 100)

    def yview(self, *a, **k):
        pass

    def yview_moveto(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass

    def add(self, *a, **k):
        pass

    def heading(self, *a, **k):
        pass

    def column(self, *a, **k):
        pass

    def insert(self, *a, **k):
        pass

    def delete(self, *a, **k):
        pass

    def get_children(self):
        return []

    def bind(self, *a, **k):
        pass

    def columnconfigure(self, *a, **k):
        pass

    def rowconfigure(self, *a, **k):
        pass

    def winfo_width(self):
        return int(getattr(self.root, "width", 100) * 0.5 * G.UI_SCALE)

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
TkMod.Canvas = DummyWidget
TkMod.Listbox = DummyWidget
TkMod.StringVar = lambda *a, **k: None
TkMod.DoubleVar = lambda *a, **k: None


class DummyVar:
    def __init__(self, value=False):
        self.val = value

    def get(self):
        return self.val

    def set(self, v):
        self.val = v


TkMod.BooleanVar = DummyVar
sys.modules["tkinter"] = TkMod

TtkMod = types.ModuleType("tkinter.ttk")
for n in [
    "Frame",
    "Label",
    "LabelFrame",
    "Button",
    "Progressbar",
    "Notebook",
    "Scrollbar",
    "Treeview",
    "Checkbutton",
]:
    setattr(TtkMod, n, DummyWidget)
sys.modules["tkinter.ttk"] = TtkMod

# matplotlib stubs
matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules["matplotlib"] = matplotlib
plt = types.ModuleType("matplotlib.pyplot")
plt.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
plt.figure = lambda *a, **k: types.SimpleNamespace(
    add_subplot=lambda *a, **k: object(), tight_layout=lambda *a, **k: None
)
plt.subplots = lambda *a, **k: (
    types.SimpleNamespace(add_subplot=lambda *a, **k: object()),
    (object(), object()),
)
plt.ioff = lambda: None
sys.modules["matplotlib.pyplot"] = plt
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = lambda *a, master=None, **k: types.SimpleNamespace(
    get_tk_widget=lambda: DummyWidget(master)
)
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules["matplotlib.backends.backend_tkagg"] = backend

from artibot.gui import TradingGUI

ens = types.SimpleNamespace(
    optimizers=[types.SimpleNamespace(param_groups=[{"lr": 0.001}])]
)
ui = TradingGUI.__new__(TradingGUI)
ui.btn_buy = DummyWidget()
ui.btn_sell = DummyWidget()
ui.btn_close = DummyWidget()
ui.label_side = DummyWidget()
ui.label_size = DummyWidget()
ui.label_entry = DummyWidget()
ui.root = DummyTk()


def test_update_position_buttons():
    ui.update_position("LONG", 2, 100.0)

    assert ui.label_side["text"] == "LONG"
    assert ui.label_size["text"] == "2"
    assert ui.label_entry["text"] == "100.00"
    assert ui.btn_buy["state"] == "disabled"
    assert ui.btn_sell["state"] == "disabled"
    assert ui.btn_close["state"] == "normal"


def test_fetch_position(monkeypatch):
    from artibot.gui import _fetch_position

    ex = types.SimpleNamespace()

    def risk(symbol):
        return {"size": -2.0, "entryPrice": 42.0}

    def allpos():
        return [
            {
                "symbol": "BTCUSD",
                "info": {"type": "swap"},
                "contracts": 3.0,
                "entryPrice": 99.0,
            }
        ]

    monkeypatch.setattr(ex, "fetch_position_risk", risk, raising=False)
    monkeypatch.setattr(ex, "fetch_positions", allpos, raising=False)

    side, sz, entry = _fetch_position(ex)
    assert side == "SHORT" and sz == 2.0 and entry == 42.0

    monkeypatch.delattr(ex, "fetch_position_risk")
    side, sz, entry = _fetch_position(ex)
    assert side == "LONG" and sz == 3.0 and entry == 99.0


def test_on_test_trade_places_order(monkeypatch):
    called = []

    def fake_order(side, amount, price):
        called.append({"side": side, "amount": amount, "price": price})
        return {"side": side, "amount": amount, "price": price}

    ui.connector = types.SimpleNamespace()
    monkeypatch.setattr(ui.connector, "create_order", fake_order, raising=False)
    monkeypatch.setattr(
        ui.connector,
        "fetch_latest_bars",
        lambda limit=1: [[1, 2, 3, 4, 99]],
        raising=False,
    )
    ui.log_trade = lambda msg: None
    after_args = {}

    def fake_after(ms, func):
        after_args["delay"] = ms
        after_args["func"] = func

    ui.root.after = fake_after
    ui.on_test_trade("buy")

    assert called[0] == {"side": "buy", "amount": 1, "price": 99}
    assert after_args["delay"] == 10000

    monkeypatch.setattr(
        ui.connector,
        "fetch_latest_bars",
        lambda limit=1: [[1, 2, 3, 4, 101]],
        raising=False,
    )
    after_args["func"]()
    assert called[1]["side"] == "sell"


def test_set_exposure_stats_updates_label():
    stats = {
        "flips": 3,
        "avg_exposure": 4.5,
        "max_long": 10,
        "max_short": -5,
        "time_in_market_pct": 75,
    }
    ui.exposure_label = DummyWidget()
    ui.set_exposure_stats(stats)
    assert "Flips:3" in ui.exposure_label["text"] or "Flips:3" in str(
        ui.exposure_label["text"]
    )
