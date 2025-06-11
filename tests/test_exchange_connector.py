# ruff: noqa: E402
import logging
import sys
import types
import time

sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("pandas", types.SimpleNamespace())

from exchanges import ExchangeConnector


class DummyEx:
    def __init__(self):
        self.markets = {"BTC/USD": {}, "BTCUSD": {}}
        self.sandbox = False
        self.last_params = None
        self.last_since = None

    def load_markets(self):
        pass

    def set_sandbox_mode(self, mode):
        self.sandbox = mode

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100, params=None):
        self.last_params = params
        self.last_since = since
        return [[1, 2, 3, 4, 5, 6]]


def test_exchange_connector_fetch(monkeypatch):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: DummyEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr(time, "time", lambda: 3600 * 2 + 10)
    conf = {"symbol": "BTCUSD", "API": {"LIVE_TRADING": False}}
    conn = ExchangeConnector(conf)

    assert conn.symbol in conn.exchange.markets

    bars = conn.fetch_latest_bars(limit=1)
    assert bars == [[1, 2, 3, 4, 5, 6]]
    assert conn.exchange.sandbox is True
    assert conn.exchange.last_params == {"type": "swap"}
    assert conn.exchange.last_since == 3600 * 1000


class ErrorEx(DummyEx):
    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100, params=None):
        self.last_params = params
        self.last_since = since
        raise RuntimeError("boom")


def test_exchange_connector_error(monkeypatch, caplog):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: ErrorEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr(time, "time", lambda: 3600 * 2 + 10)
    conf = {"symbol": "BTCUSD", "API": {"LIVE_TRADING": False}}
    conn = ExchangeConnector(conf)
    caplog.set_level(logging.ERROR)
    bars = conn.fetch_latest_bars(limit=2)
    assert bars == []
    assert conn.exchange.last_params == {"type": "swap"}
    assert conn.exchange.last_since == 0
    expected = f"fetch_ohlcv failed for {conn.symbol} tf=1h limit=2: boom"
    assert any(expected in r.message for r in caplog.records)
