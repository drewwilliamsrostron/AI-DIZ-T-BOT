import importlib.util
import os
import sys
import types
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(ROOT, "artibot", "metrics.py")


def load_metrics():
    """Load artibot.metrics without heavy dependencies."""
    if "artibot.metrics" in sys.modules:
        return sys.modules["artibot.metrics"]

    # Stub package and globals to avoid side-effect imports
    pkg = types.ModuleType("artibot")
    pkg.__path__ = [os.path.join(ROOT, "artibot")]
    sys.modules.setdefault("artibot", pkg)

    globals_stub = types.ModuleType("artibot.globals")
    globals_stub.pd = pd
    globals_stub.np = np
    sys.modules["artibot.globals"] = globals_stub

    spec = importlib.util.spec_from_file_location("artibot.metrics", METRICS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["artibot.metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


metrics = load_metrics()


def test_inactivity_exponential_penalty():
    month = 30 * 24 * 3600
    assert metrics.inactivity_exponential_penalty(0) == 0
    assert metrics.inactivity_exponential_penalty(month) == 0.01
    assert metrics.inactivity_exponential_penalty(month * 2) == 0.03
    assert metrics.inactivity_exponential_penalty(month * 3) == 0.07
    # large gap should cap at max_penalty
    assert metrics.inactivity_exponential_penalty(month * 15) == 100.0


def test_compute_days_in_profit():
    d = 86_400
    init = 100
    always_above = [(0, 110), (d, 120), (2 * d, 130)]
    always_below = [(0, 90), (d, 95), (2 * d, 98)]
    cross_up = [(0, 90), (d, 110), (2 * d, 120)]
    cross_down = [(0, 110), (d, 90), (2 * d, 80)]

    assert metrics.compute_days_in_profit(always_above, init) == 2
    assert metrics.compute_days_in_profit(always_below, init) == 0
    assert metrics.compute_days_in_profit(cross_up, init) == 1.5
    assert metrics.compute_days_in_profit(cross_down, init) == 0.5


def test_compute_yearly_stats_basic():
    def ts(s: str) -> int:
        """Shortcut to convert date string to epoch seconds."""
        return int(pd.Timestamp(s).timestamp())

    eq = [
        (ts("2023-01-01"), 100),
        (ts("2023-12-31"), 110),
        (ts("2024-01-01"), 110),
        (ts("2024-12-31"), 121),
    ]
    trades = [
        {"entry_time": ts("2023-02-01")},
        {"entry_time": ts("2023-04-01")},
        {"entry_time": ts("2024-03-01")},
    ]

    df, table = metrics.compute_yearly_stats(eq, trades)

    assert list(df.columns) == ["NetPct", "Sharpe", "MaxDD", "Trades"]
    assert list(df.index) == [2023, 2024]
    assert df.loc[2023, "NetPct"] == 10.0
    assert df.loc[2023, "Trades"] == 2
    assert df.loc[2024, "Trades"] == 1
    assert isinstance(table, str)
    assert "2023" in table and "2024" in table
