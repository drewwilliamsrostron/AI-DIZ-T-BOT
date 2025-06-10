import time
import artibot.execution as ex


def test_submit_order_jitter(monkeypatch):
    calls = {}

    def dummy(side, amount, price):
        calls["price"] = price
        calls["side"] = side
        calls["amount"] = amount
        return "ok"

    monkeypatch.setattr(time, "sleep", lambda s: calls.setdefault("delay", s))
    monkeypatch.setattr(ex.random, "normalvariate", lambda a, b: 0.05)
    monkeypatch.setattr(ex.random, "uniform", lambda a, b: 0.0004)
    result = ex.submit_order(dummy, "buy", 1.0, 100.0)
    assert result == "ok"
    assert abs(calls["delay"] - 0.05) < 1e-9
    assert calls["price"] == 100.0004
