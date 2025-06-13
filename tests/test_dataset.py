import os
import random
import sys
import types
from importlib.machinery import ModuleSpec

# ruff: noqa: E402
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Stub heavy optional modules before importing dataset
for name in ["openai", "ccxt", "tkinter", "tkinter.ttk"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules.setdefault("matplotlib", matplotlib)
pyplot = types.ModuleType("pyplot")
pyplot.__spec__ = ModuleSpec("pyplot", loader=None)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = object
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules.setdefault("matplotlib.backends.backend_tkagg", backend)

talib = types.ModuleType("talib")
talib.RSI = lambda arr, timeperiod=14: np.zeros_like(arr, dtype=np.float64)
talib.MACD = lambda arr, fastperiod=12, slowperiod=26, signalperiod=9: (
    np.zeros_like(arr, dtype=np.float64),
    np.zeros_like(arr, dtype=np.float64),
    np.zeros_like(arr, dtype=np.float64),
)
talib.__spec__ = ModuleSpec("talib", loader=None)
sys.modules.setdefault("talib", talib)


# Minimal StandardScaler stub
class SimpleScaler:
    def fit_transform(self, arr):
        arr = np.asarray(arr, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-8
        return (arr - mean) / std


preproc = types.ModuleType("sklearn.preprocessing")
preproc.StandardScaler = SimpleScaler
preproc.__spec__ = ModuleSpec("sklearn.preprocessing", loader=None)
sklearn = types.ModuleType("sklearn")
sklearn.preprocessing = preproc
sklearn.__spec__ = ModuleSpec("sklearn", loader=None)
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.preprocessing", preproc)

from artibot import dataset


def test_load_csv_hourly_missing(tmp_path):
    missing = tmp_path / "nope.csv"
    assert dataset.load_csv_hourly(str(missing)) == []


def test_load_csv_hourly_parse(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "https://example.com\n"
        "unix,date,symbol,open,high,low,close,Volume BTC,Volume USD\n"
        "2000,2024-01-01 01:00:00,BTC/USD,200000,210000,190000,205000,1,10\n"
        "1000,2024-01-01 00:00:00,BTC/USD,100000,110000,90000,105000,2,20\n"
        "3000,2024-01-01 02:00:00,BTC/USD,300000,310000,290000,305000,3,30\n"
    )
    result = dataset.load_csv_hourly(str(csv_file))
    assert result == [
        [1000, 100.0, 110.0, 90.0, 105.0, 2.0],
        [2000, 200.0, 210.0, 190.0, 205.0, 1.0],
        [3000, 300.0, 310.0, 290.0, 305.0, 3.0],
    ]


def test_hourlydataset_basic(monkeypatch):
    data = [
        [0, 1.0, 1.01, 0.99, 1.005, 0.0],
        [1, 1.005, 1.015, 1.0, 1.01, 0.0],
        [2, 1.01, 1.02, 1.005, 1.015, 0.0],
        [3, 1.015, 1.025, 1.01, 1.02, 0.0],
        [4, 1.02, 1.03, 1.015, 1.025, 0.0],
    ]

    ds = dataset.HourlyDataset(
        data,
        seq_len=3,
        indicator_hparams=dataset.IndicatorHyperparams(sma_period=1),
    )
    assert len(ds) == 2

    random.seed(0)
    sample, label = ds[0]
    assert sample.shape == (3, 8)
    assert label.shape == ()
    assert label.item() == 2
