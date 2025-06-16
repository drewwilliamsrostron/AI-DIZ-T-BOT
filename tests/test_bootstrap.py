import sys
import types
from importlib.machinery import ModuleSpec
from queue import Queue


# tkinter / matplotlib stubs
class DummyWidget:
    def __init__(self, master=None, **_kw):
        self.master = master
        self.state = "normal"
        self.attrs = {}
        self.root = self
        self._after_cbs = []
        self.destroyed = False

    def pack(self, *a, **k):
        pass

    def config(self, **kw):
        self.attrs.update(kw)

    def __setitem__(self, k, v):
        self.attrs[k] = v

    def after(self, ms, cb):
        self._after_cbs.append(cb)

    def withdraw(self):
        self.state = "withdrawn"

    def deiconify(self):
        self.state = "normal"

    def update_idletasks(self):
        pending = self._after_cbs[:]
        self._after_cbs.clear()
        for cb in pending:
            cb()

    def destroy(self):
        self.destroyed = True

    def title(self, _):
        pass

    def grab_set(self):
        pass


class DummyTk(DummyWidget):
    pass


TkMod = types.ModuleType("tkinter")
TkMod.Tk = DummyTk
TkMod.Toplevel = DummyTk
TkMod.StringVar = lambda value="": types.SimpleNamespace(
    set=lambda x: None, get=lambda: value
)
sys.modules["tkinter"] = TkMod
TtkMod = types.ModuleType("tkinter.ttk")
TtkMod.Label = DummyWidget
TtkMod.Progressbar = DummyWidget
sys.modules["tkinter.ttk"] = TtkMod

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
plt = types.ModuleType("matplotlib.pyplot")
plt.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
plt.figure = lambda *a, **k: types.SimpleNamespace(add_subplot=lambda *a, **k: object())
sys.modules["matplotlib"] = matplotlib
sys.modules["matplotlib.pyplot"] = plt
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = lambda *a, **k: DummyWidget()
sys.modules["matplotlib.backends.backend_tkagg"] = backend

import run_artibot  # noqa: E402


def test_splash_handoff():
    root = TkMod.Tk()
    q: Queue[tuple[float, str] | tuple[str, str]] = Queue()
    win, msg, pb = run_artibot._launch_loading(root, q)

    sys.modules.setdefault(
        "requests",
        types.SimpleNamespace(
            RequestException=Exception,
            get=lambda *a, **k: types.SimpleNamespace(json=lambda: {}),
        ),
    )
    sys.modules.setdefault(
        "yfinance", types.SimpleNamespace(download=lambda *a, **k: [])
    )
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x))

    def _bg():
        q.put((0.0, "start"))
        try:
            from tools import backfill_gdelt as _bf

            _bf.fetch_docs("btc")  # patched to raise
        finally:
            q.put(("DONE", ""))

    import threading

    import tools.backfill_gdelt as _bf

    _orig = _bf.fetch_docs
    _bf.fetch_docs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    threading.Thread(target=_bg, daemon=True).start()

    for _ in range(3):
        root.update_idletasks()

    _bf.fetch_docs = _orig

    assert root.state == "normal"
    assert win.destroyed
