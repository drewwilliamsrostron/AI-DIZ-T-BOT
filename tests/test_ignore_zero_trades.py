import logging
import types
import torch
from torch.utils.data import DataLoader, TensorDataset

import artibot.globals as G
from artibot.ensemble import EnsembleModel
from artibot.utils import get_device


def test_ignore_zero_trade_backtest(monkeypatch, caplog):
    device = get_device()

    def zero_trade_backtest(ensemble, data_full, indicators=None):
        return {
            "equity_curve": [],
            "effective_net_pct": 0.0,
            "inactivity_penalty": 0.0,
            "composite_reward": 0.0,
            "days_without_trading": 0,
            "trade_details": [],
            "days_in_profit": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "net_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    def dummy_stats(ec, trades, initial_balance=100.0):
        return None, ""

    monkeypatch.setattr("artibot.ensemble.robust_backtest", zero_trade_backtest)
    monkeypatch.setattr("artibot.ensemble.compute_yearly_stats", dummy_stats)
    monkeypatch.setattr(
        "artibot.ensemble.compute_monthly_stats", lambda *a, **k: (None, "")
    )
    import artibot.constants as const
    import artibot.model as model

    monkeypatch.setattr(const, "FEATURE_DIMENSION", 8)
    monkeypatch.setattr(model, "FEATURE_DIMENSION", 8)

    ens = EnsembleModel(device=device, n_models=1, n_features=8)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 3)

        def forward(self, x):
            batch = x.size(0)
            logits = x.mean(dim=1) @ self.fc.weight.T
            return logits, types.SimpleNamespace(), torch.zeros(batch)

    ens.models = [DummyModel().to(device)]
    ens.optimizers = [torch.optim.AdamW(ens.models[0].parameters(), lr=1e-3)]

    ds = TensorDataset(torch.zeros(1, 24, 8), torch.zeros(1, dtype=torch.long))
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    G.global_backtest_profit = []
    caplog.set_level(logging.INFO)
    ens.train_one_epoch(dl, dl, [])

    assert any("IGNORED_EMPTY_BACKTEST" in r.message for r in caplog.records)
    assert not any("NEW_BEST" in r.message for r in caplog.records)
    assert G.global_backtest_profit == []
