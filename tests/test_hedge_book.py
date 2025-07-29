import types
import artibot.globals as G
from artibot.position import HedgeBook


class DummyConn:
    def __init__(self) -> None:
        self.orders = []

    def create_order(self, side, amount, price):
        self.orders.append((side, amount, price))
        return {"ok": True}


def test_open_long_and_short():
    hb = HedgeBook()
    conn = DummyConn()
    G.live_equity = 1000.0
    hp = types.SimpleNamespace(long_frac=0.05, short_frac=0.03)
    hb.open_long(conn, 100.0, hp)
    hb.open_short(conn, 100.0, hp)
    assert hb.long_leg is not None
    assert hb.short_leg is not None


def test_skip_trades_in_high_vol_regime():
    hb = HedgeBook()
    conn = DummyConn()
    G.live_equity = 1000.0
    G.current_regime = 1
    hp = types.SimpleNamespace(long_frac=0.05, short_frac=0.05)
    hb.open_long(conn, 100.0, hp)
    hb.open_short(conn, 100.0, hp)
    assert hb.long_leg is None
    assert hb.short_leg is None
    assert conn.orders == []
    G.current_regime = None
