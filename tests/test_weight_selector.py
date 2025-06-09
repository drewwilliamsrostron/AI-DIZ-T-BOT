import types
from pathlib import Path

from artibot.bot_app import weight_selector_dialog


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyWidget:
    def __init__(self, *a, **k):
        pass

    def pack(self, *a, **k):
        pass


class DummyButton(DummyWidget):
    def __init__(self, master, text, command):
        master.command = command


class DummyTk:
    def __init__(self):
        self.command = None

    def title(self, t):
        pass

    def mainloop(self):
        if self.command:
            self.command()

    def quit(self):
        pass

    def destroy(self):
        pass


def make_stub():
    mod = types.SimpleNamespace(
        Tk=DummyTk,
        BooleanVar=DummyVar,
        StringVar=DummyVar,
        Checkbutton=DummyWidget,
        OptionMenu=DummyWidget,
        Button=DummyButton,
    )
    return mod


def test_weight_selector_dialog(tmp_path):
    (tmp_path / "a.pth").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "c.pth").write_text("")
    tk_stub = make_stub()
    conf = {"WEIGHTS_DIR": str(tmp_path), "USE_PREV_WEIGHTS": False}
    use_prev, path = weight_selector_dialog(conf, tk_module=tk_stub)
    assert use_prev is False
    assert Path(path).name == "a.pth"
