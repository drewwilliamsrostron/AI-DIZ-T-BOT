import types
from artibot.position import open_position, Position

class DummyConn:
    def __init__(self):
        self.calls = []
    def create_order(self, side, amount, price, order_type="market", *, stop_loss=None, take_profit=None):
        self.calls.append({
            "side": side,
            "amount": amount,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "order_type": order_type,
        })
        return {"ok": True}


def test_open_position_attaches_tp_sl():
    conn = DummyConn()
    pos = open_position(conn, "long", 1.0, 100.0, 95.0, 105.0)
    assert isinstance(pos, Position)
    assert conn.calls[0]["stop_loss"] == 95.0
    assert conn.calls[0]["take_profit"] == 105.0
