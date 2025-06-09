import artibot.globals as G
from artibot.gui import select_weight_file
from artibot.ensemble import nuclear_key_gate
import sys
import types


def test_set_status_and_full():
    G.set_status("Working")
    G.epoch_count = 3
    assert G.get_status_full() == "Working | epoch 3"


def test_weight_dialog_default(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tkinter.messagebox",
        types.SimpleNamespace(askyesno=lambda *a, **k: True),
    )
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", types.SimpleNamespace())
    path = select_weight_file()
    assert path == "best_model_weights.pth"


def test_weight_dialog_custom(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tkinter.messagebox",
        types.SimpleNamespace(askyesno=lambda *a, **k: False),
    )
    monkeypatch.setitem(
        sys.modules,
        "tkinter.filedialog",
        types.SimpleNamespace(askopenfilename=lambda *a, **k: "foo.pth"),
    )
    path = select_weight_file()
    assert path == "foo.pth"


def test_nuclear_gate():
    G.set_nuclear_key(False)
    assert nuclear_key_gate(0.2, -0.5, 0.5) is True
    G.set_nuclear_key(True)
    assert nuclear_key_gate(0.8, -0.4, 0.5) is False
    assert nuclear_key_gate(1.2, -0.1, 1.2) is True
