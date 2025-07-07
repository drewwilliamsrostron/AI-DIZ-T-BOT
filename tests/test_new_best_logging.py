import logging
import sys
import types
from types import SimpleNamespace
from importlib.machinery import ModuleSpec

# ruff: noqa: E402

import torch
from torch.utils.data import DataLoader, TensorDataset

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

import artibot.globals as G
from artibot.ensemble import EnsembleModel
from artibot.utils import get_device


def test_new_best_logging(monkeypatch, caplog):
    device = get_device()

    def dummy_backtest(ensemble, data_full, indicators=None):
        return {
            "equity_curve": [],
            "effective_net_pct": 1.0,
            "inactivity_penalty": 0.0,
            "composite_reward": 10.0,
            "days_without_trading": 0,
            "trade_details": [],
            "days_in_profit": 0.0,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "net_pct": 1.0,
            "trades": 5,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "avg_trade_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    def dummy_stats(ec, trades, initial_balance=100.0):
        return None, ""

    monkeypatch.setattr("artibot.ensemble.robust_backtest", dummy_backtest)
    monkeypatch.setattr("artibot.ensemble.compute_yearly_stats", dummy_stats)

    ens = EnsembleModel(device=device, n_models=1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(8, 3))

        def forward(self, x):
            batch = x.size(0)
            logits = x.mean(dim=1) @ self.w
            return logits, SimpleNamespace(), torch.zeros(batch)

    ens.models = [DummyModel().to(device)]
    ens.optimizers = [torch.optim.AdamW(ens.models[0].parameters(), lr=1e-3)]

    ds = TensorDataset(torch.zeros(1, 24, 8), torch.zeros(1, dtype=torch.long))
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    G.global_attention_entropy_history = [1.2]
    G.global_sharpe = 1.2
    G.global_max_drawdown = -0.1

    caplog.set_level(logging.INFO)
    ens.train_one_epoch(dl, dl, [])

    assert any("NEW_BEST" in r.message for r in caplog.records)
