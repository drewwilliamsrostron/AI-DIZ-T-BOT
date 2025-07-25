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
        self.markets = {"BTCUSD": {}}
        self.sandbox = False
        self.last_params = None
        self.last_since = None
        self.created = []

    def load_markets(self):
        pass

    def set_sandbox_mode(self, mode):
        self.sandbox = mode

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100, params=None):
        self.last_params = params
        self.last_since = since
        return [[1, 2, 3, 4, 5, 6]]

    def create_order(self, symbol, order_type, side, amount, price, params=None):
        self.created.append((symbol, order_type, side, amount, price, params))
        return {"id": "1"}


def test_exchange_connector_fetch(monkeypatch):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: DummyEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr(time, "time", lambda: 3600 * 2 + 10)
    conf = {"API": {"LIVE_TRADING": False}}
    conn = ExchangeConnector(conf)

    assert conn.symbol in conn.exchange.markets

    bars = conn.fetch_latest_bars(limit=1)
    assert bars == [[1, 2, 3, 4, 5, 6]]
    assert conn.exchange.sandbox is True
    assert conn.exchange.last_params is None
    assert conn.exchange.last_since is None

    order = conn.create_order("buy", 1.0, 100.0)
    assert order == {"id": "1"}
    assert conn.exchange.created[0][2] == "buy"
    assert conn.exchange.created[0][-1] == {"type": "swap"}


class ErrorEx(DummyEx):
    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100, params=None):
        self.last_params = params
        self.last_since = since
        raise RuntimeError("boom")


def test_exchange_connector_error(monkeypatch, caplog):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: ErrorEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr(time, "time", lambda: 3600 * 2 + 10)
    conf = {"API": {"LIVE_TRADING": False}}
    conn = ExchangeConnector(conf)
    caplog.set_level(logging.ERROR)
    bars = conn.fetch_latest_bars(limit=2)
    assert bars == []
    assert conn.exchange.last_params is None
    assert conn.exchange.last_since is None
    expected = f"fetch_ohlcv failed for {conn.symbol} tf=1h limit=2: boom"
    assert any(expected in r.message for r in caplog.records)


def test_connector_urls_sandbox(monkeypatch):
    captured = {}

    def factory(params):
        captured["params"] = params
        return DummyEx()

    mod = types.SimpleNamespace(phemex=factory)
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    conf = {
        "API": {
            "LIVE_TRADING": False,
            "API_URL_LIVE": "https://live",
            "API_URL_TEST": "https://test",
        },
    }
    conn = ExchangeConnector(conf)
    urls = captured["params"]["urls"]["api"]
    assert urls == {"live": "https://live", "test": "https://test"}
    assert conn.exchange.sandbox is True


def test_connector_urls_live(monkeypatch):
    captured = {}

    def factory(params):
        captured["params"] = params
        return DummyEx()

    mod = types.SimpleNamespace(phemex=factory)
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    conf = {
        "API": {
            "LIVE_TRADING": True,
            "API_URL_LIVE": "https://live",
            "API_URL_TEST": "https://test",
        },
    }
    conn = ExchangeConnector(conf)
    urls = captured["params"]["urls"]["api"]
    assert urls == {"live": "https://live", "test": "https://test"}
    assert conn.exchange.sandbox is False
