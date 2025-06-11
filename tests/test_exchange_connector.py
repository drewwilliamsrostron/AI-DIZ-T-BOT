import types
import sys

from exchanges import ExchangeConnector


class DummyEx:
    def __init__(self):
        self.markets = {"BTC/USD": {}, "BTCUSD": {}}
        self.sandbox = False

    def load_markets(self):
        pass

    def set_sandbox_mode(self, mode):
        self.sandbox = mode

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[1, 2, 3, 4, 5, 6]]


def test_exchange_connector_fetch(monkeypatch):
    mod = types.SimpleNamespace(phemex=lambda *a, **k: DummyEx())
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    conf = {"symbol": "BTCUSD", "API": {"LIVE_TRADING": False}}
    conn = ExchangeConnector(conf)

    assert conn.symbol in conn.exchange.markets

    bars = conn.fetch_latest_bars(limit=1)
    assert bars == [[1, 2, 3, 4, 5, 6]]
    assert conn.exchange.sandbox is True
