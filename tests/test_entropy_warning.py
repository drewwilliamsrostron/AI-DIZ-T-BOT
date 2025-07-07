import logging
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, TensorDataset

import artibot.globals as G
from artibot.ensemble import EnsembleModel
from artibot.utils import get_device


def test_entropy_warning(monkeypatch, caplog):
    device = get_device()

    def dummy_backtest(ensemble, data_full, indicators=None):
        return {
            "equity_curve": [],
            "effective_net_pct": 0.0,
            "inactivity_penalty": 0.0,
            "composite_reward": 0.0,
            "days_without_trading": 0,
            "trade_details": [],
            "days_in_profit": 0.0,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "net_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
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
            self.last_entropy = 0.4
            self.last_max_prob = 1.0
            batch = x.size(0)
            feat = x.mean(dim=1)
            logits = feat @ self.w
            return logits, SimpleNamespace(), torch.zeros(batch)

    ens.models = [DummyModel().to(device)]
    ens.optimizers = [torch.optim.AdamW(ens.models[0].parameters(), lr=1e-3)]

    ds = TensorDataset(torch.zeros(1, 24, 8), torch.zeros(1, dtype=torch.long))
    dl = DataLoader(ds, batch_size=1, pin_memory=True)

    caplog.set_level(logging.WARNING)
    G.global_attention_entropy_history = [0.4] * 120
    ens.train_one_epoch(dl, dl, [])
    assert any("Attention entropy" in r.message for r in caplog.records)
