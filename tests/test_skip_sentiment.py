import sys
import types
import os


class DummyRoot:
    def withdraw(self):
        pass

    def destroy(self):
        pass


def test_skip_sentiment_yes(monkeypatch, capsys):
    tk_stub = types.SimpleNamespace(
        Tk=lambda: DummyRoot(),
        messagebox=types.SimpleNamespace(askyesno=lambda *a, **k: True),
    )
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk_stub.messagebox)
    os.environ["NO_HEAVY"] = "1"
    import importlib
    import run_artibot

    importlib.reload(run_artibot)
    run_artibot.SKIP_SENTIMENT = False
    run_artibot.ask_skip_sentiment(tk_module=tk_stub)
    assert run_artibot.SKIP_SENTIMENT is True
    assert sys.modules["os"].environ.get("NO_HEAVY") == "1"
    out = capsys.readouterr().out
    assert "Skipping sentiment pull" in out


def test_skip_sentiment_no(monkeypatch):
    tk_stub = types.SimpleNamespace(
        Tk=lambda: DummyRoot(),
        messagebox=types.SimpleNamespace(askyesno=lambda *a, **k: False),
    )
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk_stub.messagebox)
    os.environ["NO_HEAVY"] = "1"
    import importlib
    import run_artibot

    importlib.reload(run_artibot)
    run_artibot.SKIP_SENTIMENT = False
    run_artibot.ask_skip_sentiment(tk_module=tk_stub)
    assert run_artibot.SKIP_SENTIMENT is False
    assert "NO_HEAVY" not in sys.modules["os"].environ
