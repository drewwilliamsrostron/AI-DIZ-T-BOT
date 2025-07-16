import logging
import types
import torch
from torch.utils.data import DataLoader, TensorDataset

import artibot.globals as G
from artibot.ensemble import EnsembleModel
from artibot.utils import get_device


def test_not_promoted_low_reward(monkeypatch, caplog):
    device = get_device()

    def dummy_backtest(ensemble, data_full, indicators=None):
        return {
            "equity_curve": [],
            "effective_net_pct": 1.0,
            "inactivity_penalty": 0.0,
            "composite_reward": 1.0,
            "days_without_trading": 0,
            "trade_details": [0],
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
    monkeypatch.setattr(
        "artibot.ensemble.compute_monthly_stats", lambda *a, **k: (None, "")
    )

    import artibot.constants as const
    import artibot.model as model

    monkeypatch.setattr(const, "FEATURE_DIMENSION", 8)
    monkeypatch.setattr(model, "FEATURE_DIMENSION", 8)

    ens = EnsembleModel(device=device, n_models=1, n_features=8)
    ens.best_composite_reward = 5.0

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

    G.global_attention_entropy_history = [1.2]
    G.global_sharpe = 1.2
    G.global_max_drawdown = -0.1

    ens.train_steps = 1
    caplog.set_level(logging.INFO)
    ens.train_one_epoch(dl, dl, [])

    assert any("NOT_PROMOTED" in r.message for r in caplog.records)
