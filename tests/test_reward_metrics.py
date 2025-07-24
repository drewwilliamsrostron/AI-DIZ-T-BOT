import sys
import torch
import pytest
from artibot.utils.reward_utils import sortino_ratio, omega_ratio, calmar_ratio
import artibot.globals as G
from artibot.backtest import robust_backtest
from artibot.hyperparams import IndicatorHyperparams
import types
import numpy as np


def test_sortino_ratio_basic():
    r = torch.tensor([0.1, -0.05, 0.2])
    val = sortino_ratio(r)
    assert val > 0


def test_omega_ratio_basic():
    r = torch.tensor([0.1, -0.1, 0.2])
    val = omega_ratio(r)
    assert val > 1


def test_calmar_ratio_basic():
    val = calmar_ratio(0.2, -0.1, 365)
    assert val == pytest.approx(2.0, rel=1e-3)


def test_composite_reward_uses_risk_metrics(monkeypatch):
    sys.modules["openai"] = types.SimpleNamespace()
    sys.modules["talib"] = types.SimpleNamespace(
        RSI=lambda arr, timeperiod=14: np.zeros_like(arr),
        MACD=lambda arr, fastperiod=12, slowperiod=26, signalperiod=9: (
            np.zeros_like(arr),
            np.zeros_like(arr),
            np.zeros_like(arr),
        ),
        EMA=lambda arr, timeperiod=20: np.zeros_like(arr),
    )

    G.use_net_term = False
    G.use_drawdown_term = False
    G.use_trade_term = False
    G.use_profit_days_term = False
    G.use_sharpe_term = True
    G.use_sortino_term = True
    G.use_omega_term = True
    G.use_calmar_term = True
    G.beta = 0.5
    G.theta = 0.5
    G.phi = 0.5
    G.chi = 0.5

    class Dummy:
        def __init__(self):
            self.device = torch.device("cpu")
            self.indicator_hparams = IndicatorHyperparams()

        def vectorized_predict(self, w, batch_size=512):
            preds = torch.zeros(len(w), dtype=torch.long)
            avg = {
                "sl_multiplier": torch.tensor(1.0),
                "tp_multiplier": torch.tensor(1.0),
                "risk_fraction": torch.tensor(0.1),
            }
            return preds, None, avg

    rows = [[i * 3600, 100, 101, 99, 100, 0] for i in range(30)]
    res = robust_backtest(Dummy(), rows)
    exp = (
        G.beta * float(np.clip(res["sharpe"], -1.0, 1.0))
        + G.theta * float(np.clip(res["sortino"], -1.0, 1.0))
        + G.phi * float(np.clip(res["omega"], -1.0, 1.0))
        + G.chi * float(np.clip(res["calmar"], -1.0, 1.0))
        - res["inactivity_penalty"]
    )
    assert res["composite_reward"] == pytest.approx(exp, rel=1e-6)
