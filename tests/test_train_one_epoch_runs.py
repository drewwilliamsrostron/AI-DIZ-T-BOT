import types
import torch
from torch.utils.data import DataLoader, TensorDataset

from artibot.ensemble import EnsembleModel
from artibot.utils import get_device


def test_train_one_epoch_runs(monkeypatch):
    device = get_device()

    def dummy_backtest(ensemble, data_full, indicators=None):
        return {
            "equity_curve": [(0, 100.0)],
            "effective_net_pct": 0.0,
            "inactivity_penalty": 0.0,
            "composite_reward": 0.1,
            "days_without_trading": 0,
            "trade_details": [],
            "days_in_profit": 0.0,
            "sharpe": 0.1,
            "max_drawdown": -0.1,
            "net_pct": 0.0,
            "trades": 1,
            "win_rate": 1.0,
            "profit_factor": 1.0,
            "avg_trade_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    def dummy_stats(ec, trades, initial_balance=100.0):
        return None, ""

    monkeypatch.setattr("artibot.ensemble.robust_backtest", dummy_backtest)
    monkeypatch.setattr("artibot.ensemble.compute_yearly_stats", dummy_stats)
    monkeypatch.setattr("artibot.ensemble.compute_monthly_stats", lambda *a, **k: (None, ""))

    ens = EnsembleModel(device=device, n_models=1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(16, 3))

        def forward(self, x):
            batch = x.size(0)
            logits = x.mean(dim=1) @ self.w
            return logits, types.SimpleNamespace(), torch.zeros(batch)

    ens.models = [DummyModel().to(device)]
    ens.optimizers = [torch.optim.AdamW(ens.models[0].parameters(), lr=1e-3)]

    ds = TensorDataset(torch.zeros(2, 24, 16), torch.zeros(2, dtype=torch.long))
    dl = DataLoader(ds, batch_size=1)

    ens.train_one_epoch(dl, dl, [])
