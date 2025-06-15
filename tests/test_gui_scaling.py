# ruff: noqa: E402
import sys
import types
from importlib.machinery import ModuleSpec

for name in ["openai", "ccxt"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)


# tkinter stubs with simple geometry handling
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


class DummyTk(DummyWidget):
    def __init__(self, *a, **k):
        super().__init__(self)
        self.width = 1280
        self.height = 720
        self.tk = types.SimpleNamespace(call=lambda *a, **k: None)
        self._cb = None
        self.root = self

    def after(self, *a, **k):
        pass

    def after_idle(self, cb, *a, **k):
        pass

    def title(self, t):
        pass

    def bind(self, ev, cb):
        if ev == "<Configure>":
            self._cb = cb

    def geometry(self, spec):
        w, h = spec.split("x")
        self.width = int(w)
        self.height = int(h)

    def update_idletasks(self):
        if self._cb:
            e = types.SimpleNamespace(width=self.width, height=self.height)
            self._cb(e)

    def winfo_fpixels(self, val):
        return 96.0


TkMod = types.ModuleType("tkinter")
TkMod.Tk = DummyTk
TkMod.Frame = DummyWidget
TkMod.Label = DummyWidget
TkMod.LabelFrame = DummyWidget
TkMod.Text = DummyWidget
TkMod.Toplevel = DummyTk
TkMod.Canvas = DummyWidget
TkMod.Listbox = DummyWidget
TkMod.StringVar = lambda *a, **k: None
TkMod.DoubleVar = lambda *a, **k: DummyVar()


class DummyVar:
    def __init__(self, value=False):
        self.val = value

    def get(self):
        return self.val

    def set(self, v):
        self.val = v


TkMod.BooleanVar = DummyVar
TkMod.BOTH = "both"
TkMod.LEFT = "left"
TkMod.RIGHT = "right"
TkMod.Y = "y"
TkMod.X = "x"
TkMod.END = "end"
TkMod.W = "w"
TkMod.CENTER = "center"
TkMod.DISABLED = "disabled"
TkMod.NORMAL = "normal"
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


class DummyStyle:
    def theme_use(self, *a, **k):
        pass

    def configure(self, *a, **k):
        pass


TtkMod.Style = DummyStyle
sys.modules["tkinter.ttk"] = TtkMod

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules["matplotlib"] = matplotlib
plt = types.ModuleType("matplotlib.pyplot")
plt.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
plt.figure = lambda *a, **k: types.SimpleNamespace(
    add_subplot=lambda *a, **k: object(),
    tight_layout=lambda *a, **k: None,
)
plt.subplots = lambda *a, **k: (
    types.SimpleNamespace(add_subplot=lambda *a, **k: object()),
    object(),
)
plt.ioff = lambda: None
sys.modules["matplotlib.pyplot"] = plt
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = lambda *a, master=None, **k: types.SimpleNamespace(
    get_tk_widget=lambda: DummyWidget(master)
)
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules["matplotlib.backends.backend_tkagg"] = backend

import artibot.globals as G
from artibot.gui import TradingGUI
import artibot.gui as gui_module

gui_module.redraw_everything = lambda: None


def test_gui_resizes_smaller():
    root = DummyTk()
    ens = types.SimpleNamespace(
        optimizers=[types.SimpleNamespace(param_groups=[{"lr": 0.001}])]
    )
    gui = TradingGUI(root, ens)
    root.update_idletasks()
    scale = G.UI_SCALE
    w0 = gui.canvas_train.get_tk_widget().winfo_width()
    root.geometry("800x600")
    root.update_idletasks()
    assert G.UI_SCALE == scale
    assert gui.canvas_train.get_tk_widget().winfo_width() < w0
