import sys
import types
from importlib.machinery import ModuleSpec
import json

# ruff: noqa: E402

import torch
from torch.utils.data import DataLoader, TensorDataset

# stub heavy deps
for name in ["openai", "ccxt", "tkinter", "tkinter.ttk"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules.setdefault("matplotlib", matplotlib)
plt = types.ModuleType("matplotlib.pyplot")
plt.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
sys.modules.setdefault("matplotlib.pyplot", plt)
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = lambda *a, **k: types.SimpleNamespace(
    get_tk_widget=lambda: None
)
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules.setdefault("matplotlib.backends.backend_tkagg", backend)
if "duckdb" not in sys.modules:
    db = types.ModuleType("duckdb")
    db.connect = lambda *a, **k: types.SimpleNamespace(
        execute=lambda *a, **k: types.SimpleNamespace(fetchone=lambda: (0,))
    )
    db.DuckDBPyConnection = object
    db.InvalidInputException = Exception
    db.BinderException = Exception
    sys.modules["duckdb"] = db

from artibot.ensemble import EnsembleModel


def test_persist_and_reload(tmp_path, monkeypatch):
    rewards = [10.0, 5.0]

    def fake_backtest(*a, **k):
        r = rewards.pop(0)
        return {
            "equity_curve": [[0, 1.0], [1, 2.0]],
            "effective_net_pct": r,
            "inactivity_penalty": 0.0,
            "composite_reward": r,
            "days_without_trading": 0,
            "trade_details": [{"entry_time": 0}] * 5,
            "days_in_profit": 0.0,
            "sharpe": r,
            "max_drawdown": -0.1,
            "net_pct": r,
            "trades": 5,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "avg_trade_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    monkeypatch.setattr("artibot.ensemble.robust_backtest", fake_backtest)
    monkeypatch.setattr(
        "artibot.ensemble.compute_yearly_stats", lambda *a, **k: (None, "")
    )
    monkeypatch.setattr(
        "artibot.ensemble.compute_monthly_stats", lambda *a, **k: (None, "")
    )

    ens = EnsembleModel(
        device=torch.device("cpu"), n_models=1, weights_path=str(tmp_path / "best.pth")
    )
    ens.save_dir = str(tmp_path)
    ens.gui = types.SimpleNamespace(
        update_equity_curve=lambda *a, **k: None, update_best_stats=lambda *a, **k: None
    )

    ds = TensorDataset(
        torch.zeros(1, 24, ens.models[0].input_dim), torch.zeros(1, dtype=torch.long)
    )
    dl = DataLoader(ds, batch_size=1)

    ens.train_one_epoch(dl, dl, [])

    best_model = tmp_path / "best_model.pt"
    metrics_file = tmp_path / "best_metrics.json"
    assert best_model.exists() and metrics_file.exists()

    metrics = json.loads(metrics_file.read_text())
    assert metrics["best_reward"] == 10.0
    assert metrics["best_epoch"] == ens.best_epoch

    mtime = metrics_file.stat().st_mtime
    ens.train_one_epoch(dl, dl, [])
    assert metrics_file.stat().st_mtime == mtime

    new_ens = EnsembleModel(
        device=torch.device("cpu"), n_models=1, weights_path=str(tmp_path / "best2.pth")
    )
    new_ens.save_dir = str(tmp_path)
    new_ens.load_persisted_best()
    assert new_ens.best_reward == 10.0
    assert new_ens.best_epoch == 0
