import artibot.globals as G
from artibot.gui import select_weight_file, ask_use_prev_weights
from artibot.ensemble import nuclear_key_gate
import sys
import types


def test_set_status_and_full():
    G.set_status("Working", "")
    G.epoch_count = 3
    assert G.get_status_full() == ("Working", "epoch 3")


def test_weight_dialog_default(monkeypatch):
    tk_stub = types.SimpleNamespace(
        messagebox=types.SimpleNamespace(askyesno=lambda *a, **k: True),
        filedialog=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk_stub.messagebox)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", tk_stub.filedialog)
    path = select_weight_file()
    assert path == "best_model_weights.pth"


def test_weight_dialog_custom(monkeypatch):
    tk_stub = types.SimpleNamespace(
        messagebox=types.SimpleNamespace(askyesno=lambda *a, **k: False),
        filedialog=types.SimpleNamespace(askopenfilename=lambda *a, **k: "foo.pth"),
    )
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk_stub.messagebox)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", tk_stub.filedialog)
    path = select_weight_file()
    assert path == "foo.pth"


class DummyRoot:
    def withdraw(self):
        pass

    def destroy(self):
        pass


def test_startup_dialog_yes(monkeypatch):
    tk_stub = types.SimpleNamespace(
        Tk=lambda: DummyRoot(),
        messagebox=types.SimpleNamespace(askyesno=lambda *a, **k: True),
    )
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk_stub.messagebox)
    assert ask_use_prev_weights(tk_module=tk_stub) is True


def test_startup_dialog_no(monkeypatch):
    tk_stub = types.SimpleNamespace(
        Tk=lambda: DummyRoot(),
        messagebox=types.SimpleNamespace(askyesno=lambda *a, **k: False),
    )
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk_stub.messagebox)
    assert ask_use_prev_weights(tk_module=tk_stub) is False


def test_nuclear_gate():
    G.set_nuclear_key(False)
    assert nuclear_key_gate(0.2, -0.5, 0.5, 2.0) is True
    G.set_nuclear_key(True)
    assert nuclear_key_gate(0.8, -0.4, 0.5, 0.8) is False
    assert nuclear_key_gate(1.2, -0.1, 1.2, 1.5) is True
