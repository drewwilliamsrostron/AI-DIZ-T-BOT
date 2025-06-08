# ruff: noqa: E402
import types
import numpy as np
import sys
import os
from importlib.machinery import ModuleSpec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for name in ["openai", "ccxt", "tkinter", "tkinter.ttk"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

talib = types.ModuleType("talib")
talib.RSI = lambda arr, timeperiod=14: np.zeros_like(arr, dtype=np.float64)
talib.MACD = lambda arr, fastperiod=12, slowperiod=26, signalperiod=9: (
    np.zeros_like(arr, dtype=np.float64),
    np.zeros_like(arr, dtype=np.float64),
    np.zeros_like(arr, dtype=np.float64),
)
talib.__spec__ = ModuleSpec("talib", loader=None)
sys.modules.setdefault("talib", talib)

from artibot.dataset import load_csv_hourly


def test_adaptive_scaler(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "https://x\n"
        "unix,date,symbol,open,high,low,close,Volume BTC,Volume USD\n"
        "1000,2024-01-01 00:00:00,BTC/USD,2000000000,2100000000,1900000000,2050000000,1,10\n"
        "2000,2024-01-01 01:00:00,BTC/USD,2100000000,2200000000,2000000000,2150000000,1,10\n"
    )
    rows = load_csv_hourly(str(csv_file))
    assert 1000 < rows[0][1] < 200000
    assert 1000 < rows[1][1] < 200000
