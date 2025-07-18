import os
import sys
import types

import numpy as np
import pytest

# ruff: noqa: E402
import torch

# Stub external dependencies required by artibot.globals
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

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

from artibot.backtest import robust_backtest
from artibot.backtest import compute_indicators
import artibot.execution as execution
from artibot.hyperparams import IndicatorHyperparams


class DummyEnsemble:
    """Ensemble producing constant predictions."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.indicator_hparams = IndicatorHyperparams(
            rsi_period=14,
            sma_period=10,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
        )

    def vectorized_predict(self, windows_t, batch_size: int = 512):
        preds = torch.zeros(len(windows_t), dtype=torch.long)
        avg = {
            "sl_multiplier": torch.tensor(1.0),
            "tp_multiplier": torch.tensor(1.0),
            "risk_fraction": torch.tensor(0.1),
        }
        return preds, None, avg


def test_robust_backtest_simple(monkeypatch):
    def stub_submit(func, side, amount, price, **kw):
        slip = 0.0002
        adj = price * (1 + slip) if side == "buy" else price * (1 - slip)
        return func(side=side, amount=amount, price=adj, **kw)

    monkeypatch.setattr(execution, "submit_order", stub_submit)
    data = [
        [
            i * 3600,
            100 + i * 0.1,
            100 + i * 0.1 + 0.05,
            100 + i * 0.1 - 0.05,
            100 + i * 0.1,
            0,
        ]
        for i in range(30)
    ]
    result = robust_backtest(DummyEnsemble(), data)
    assert result["trades"] == 1
    assert round(result["net_pct"], 2) == 0.93
    # Composite reward should heavily penalise draw-downs.
    assert result["composite_reward"] == pytest.approx(2.19581, rel=1e-3)


def test_robust_backtest_unbounded_reward(monkeypatch):
    def stub_submit(func, side, amount, price, **kw):
        slip = 0.0002
        adj = price * (1 + slip) if side == "buy" else price * (1 - slip)
        return func(side=side, amount=amount, price=adj, **kw)

    monkeypatch.setattr(execution, "submit_order", stub_submit)
    data = [
        [i * 30 * 24 * 3600, 100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i, 0]
        for i in range(50)
    ]
    result = robust_backtest(DummyEnsemble(), data)
    # Large draw-downs should yield a strongly negative reward.
    assert result["composite_reward"] < 0


def test_backtest_with_precomputed_features(monkeypatch):
    def stub_submit(func, side, amount, price, **kw):
        slip = 0.0002
        adj = price * (1 + slip) if side == "buy" else price * (1 - slip)
        return func(side=side, amount=amount, price=adj, **kw)

    monkeypatch.setattr(execution, "submit_order", stub_submit)
    data = [
        [
            i * 3600,
            100 + i * 0.1,
            100 + i * 0.1 + 0.05,
            100 + i * 0.1 - 0.05,
            100 + i * 0.1,
            0,
        ]
        for i in range(30)
    ]
    ens = DummyEnsemble()
    indicators = compute_indicators(data, ens.indicator_hparams)
    result_pre = robust_backtest(ens, data, indicators=indicators)
    result_std = robust_backtest(ens, data)
    assert result_pre["net_pct"] == pytest.approx(result_std["net_pct"], rel=5e-3)


def test_trade_details_have_size(monkeypatch):
    def stub_submit(func, side, amount, price, **kw):
        return func(side=side, amount=amount, price=price, **kw)

    monkeypatch.setattr(execution, "submit_order", stub_submit)
    data = [[i * 3600, 100.0, 101.0, 99.0, 100.0, 0.0] for i in range(30)]
    result = robust_backtest(DummyEnsemble(), data)
    assert result["trade_details"], "Expected at least one trade"
    trade = result["trade_details"][0]
    assert "size" in trade and abs(trade["size"]) > 0
