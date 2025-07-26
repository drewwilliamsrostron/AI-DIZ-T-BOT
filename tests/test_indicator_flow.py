
import artibot.training as training
from artibot.hyperparams import IndicatorHyperparams
from artibot.ensemble import EnsembleModel
import artibot.globals as G
import torch


def test_indicator_flow(monkeypatch):
    periods = []

    def fake_quick_fit(*a, **k):
        pass

    def fake_backtest(model, data, indicator_hp=None):
        periods.append(indicator_hp.sma_period)
        return {"trades": 1, "net_pct": 0.0, "composite_reward": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    monkeypatch.setattr(training, "quick_fit", fake_quick_fit)
    monkeypatch.setattr(training, "robust_backtest", fake_backtest)
    monkeypatch.setattr(G, "push_backtest_metrics", lambda *a, **k: None)

    data = [[i, 0, 0, 0, 0, 0] for i in range(80)]
    hp = IndicatorHyperparams(sma_period=29)

    training.walk_forward_backtest(data, 20, 20, indicator_hp=hp, freeze_features=True)
    assert periods[0] == 29
    assert periods[1] == 29
