import random
from artibot.execution import submit_order


def test_submit_order_adjusts_price(monkeypatch):
    prices = []

    def fake(side: str, amount: float, price: float, **kw):
        prices.append(price)
        return price

    import artibot.execution as exe

    monkeypatch.setattr(exe.time, "sleep", lambda s: None)
    random.seed(0)
    submit_order(fake, "buy", 1.0, 100.0)
    assert 99.9995 <= prices[0] <= 100.0005
