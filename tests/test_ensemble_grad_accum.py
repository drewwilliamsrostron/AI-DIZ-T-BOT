import types
import torch
from torch.utils.data import DataLoader, TensorDataset
from artibot.ensemble import EnsembleModel
import artibot.constants as const
import artibot.model as model


def test_optimizer_steps_with_grad_accum(monkeypatch):
    device = torch.device("cpu")

    def dummy_backtest(*a, **k):
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

    monkeypatch.setattr("artibot.ensemble.robust_backtest", dummy_backtest)
    monkeypatch.setattr("artibot.ensemble.compute_yearly_stats", lambda *a, **k: (None, ""))
    monkeypatch.setattr("artibot.ensemble.compute_monthly_stats", lambda *a, **k: (None, ""))

    monkeypatch.setattr(const, "FEATURE_DIMENSION", 8)
    monkeypatch.setattr(model, "FEATURE_DIMENSION", 8)

    ens = EnsembleModel(device=device, n_models=1, n_features=8, grad_accum_steps=2, delayed_reward_epochs=0)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(8, 3))

        def forward(self, x):
            batch = x.size(0)
            logits = x.mean(dim=1) @ self.w
            return logits, types.SimpleNamespace(), torch.zeros(batch)

    ens.models = [DummyModel().to(device)]
    opt = torch.optim.SGD(ens.models[0].parameters(), lr=0.1)
    ens.optimizers = [opt]

    ds = TensorDataset(torch.zeros(3, 24, 8), torch.zeros(3, dtype=torch.long))
    dl = DataLoader(ds, batch_size=1)

    calls = []
    orig_step = opt.step

    def counting_step(*a, **k):
        calls.append(True)
        return orig_step(*a, **k)

    opt.step = counting_step

    ens.train_one_epoch(dl, None, [], update_globals=False)

    assert len(calls) == 2
