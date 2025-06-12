import types
import logging
import sys
import time
from artibot.training import PhemexConnector


class DummyEx:
    def __init__(self):
        self.markets = {"BTCUSD": {}}
        self.created = []
        self.sandbox = False
        self.last_since = None

    def load_markets(self):
        pass

    def set_sandbox_mode(self, mode):
        self.sandbox = mode

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.last_since = since
        return [[1, 2, 3, 4, 5, 6]]

    def create_order(self, symbol, order_type, side, amount, price, params=None):
        self.created.append((symbol, order_type, side, amount, price, params))
        return {"id": "1"}

    def fetch_balance(self):
        return {"total": {"BTC": 1.0}}


def test_phemex_connector(monkeypatch):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: DummyEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr("artibot.utils.account.get_account_equity", lambda ex: 100.0)
    monkeypatch.setattr(time, "time", lambda: 3600 * 2 + 10)
    conf = {"API": {"LIVE_TRADING": False}}
    conn = PhemexConnector(conf)
    assert conn.exchange.last_since is None
    bars = conn.fetch_latest_bars(limit=1)
    assert bars == [[1, 2, 3, 4, 5, 6]]
    assert conn.exchange.last_since is None
    order = conn.create_order("buy", 1.0, 100.0)
    assert order == {"id": "1"}
    assert conn.exchange.created[0][2] == "buy"
    assert conn.exchange.created[0][-1] == {"type": "swap"}

    assert conn.exchange.sandbox is True

    stats = conn.get_account_stats()
    assert stats == {"total": {"BTC": 1.0, "USDT": 100.0}}


class ErrorEx(DummyEx):
    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.last_since = since
        raise RuntimeError("oops")


def test_phemex_connector_error(monkeypatch, caplog):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: ErrorEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr(time, "time", lambda: 3600 * 2 + 10)
    conf = {"API": {"LIVE_TRADING": False}}
    conn = PhemexConnector(conf)
    caplog.set_level(logging.ERROR)
    bars = conn.fetch_latest_bars(limit=3)
    assert bars == []
    assert conn.exchange.last_since is None
    assert any(
        "fetch_ohlcv failed for BTCUSD tf=1h limit=3: oops" in r.message
        for r in caplog.records
    )
