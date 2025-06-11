import types
import logging
import sys
from artibot.training import PhemexConnector


class DummyEx:
    def __init__(self):
        self.markets = {"BTC/USD": {}}
        self.created = []
        self.sandbox = False

    def load_markets(self):
        pass

    def set_sandbox_mode(self, mode):
        self.sandbox = mode

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[1, 2, 3, 4, 5, 6]]

    def create_order(self, symbol, order_type, side, amount, price, params=None):
        self.created.append((symbol, order_type, side, amount, price, params))
        return {"id": "1"}

    def fetch_balance(self):
        return {"total": {"BTC": 1.0}}


def test_phemex_connector(monkeypatch):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: DummyEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    conf = {"symbol": "BTC/USD", "API": {"LIVE_TRADING": False}}
    conn = PhemexConnector(conf)
    bars = conn.fetch_latest_bars(limit=1)
    assert bars == [[1, 2, 3, 4, 5, 6]]
    order = conn.create_order("buy", 1.0, 100.0)
    assert order == {"id": "1"}
    assert conn.exchange.created[0][2] == "buy"
    assert conn.exchange.created[0][-1] == {"type": "swap"}

    assert conn.exchange.sandbox is True

    stats = conn.get_account_stats()
    assert stats == {"total": {"BTC": 1.0}}


class ErrorEx(DummyEx):
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        raise RuntimeError("oops")


def test_phemex_connector_error(monkeypatch, caplog):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: ErrorEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    conf = {"symbol": "BTC/USD", "API": {"LIVE_TRADING": False}}
    conn = PhemexConnector(conf)
    caplog.set_level(logging.ERROR)
    bars = conn.fetch_latest_bars(limit=3)
    assert bars == []
    assert any(
        "fetch_ohlcv failed for BTC/USD tf=1h limit=3: oops" in r.message
        for r in caplog.records
    )
