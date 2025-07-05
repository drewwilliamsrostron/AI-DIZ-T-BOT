import pandas as pd
import sys
import types

# ruff: noqa: E402

sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda *a, **k: types.SimpleNamespace(type="cpu"),
    ),
)
sys.modules.setdefault(
    "matplotlib",
    types.SimpleNamespace(use=lambda *a, **k: None),
)
sys.modules.setdefault(
    "duckdb",
    types.SimpleNamespace(
        connect=lambda *a, **k: None,
        InvalidInputException=Exception,
        BinderException=Exception,
    ),
)

from artibot.feature_ingest import _fetch_btc_usdt_hourly


class DummyEx:
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=168):
        return [[0, 1, 1, 1, 1, 1]] * 2


def test_fetch_hourly(monkeypatch):
    monkeypatch.setattr("ccxt.phemex", lambda: DummyEx())
    import artibot.feature_store as fs

    monkeypatch.setattr(fs, "upsert_rvol", lambda *a, **k: None)
    df = _fetch_btc_usdt_hourly()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
