import logging
import types
import numpy as np

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import artibot.globals as G
import threading

from artibot.ensemble import EnsembleModel
from artibot.hyperparams import HyperParams
from artibot.rl import ACTION_KEYS, MetaTransformerRL, IndicatorHyperparams
from artibot.training import csv_training_thread
from artibot.utils import get_device, setup_logging


@pytest.fixture(autouse=True)
def reset_handlers():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)


def test_setup_logging_adds_debug_file(tmp_path):
    setup_logging()
    handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    names = sorted([getattr(h, "baseFilename", "") for h in handlers])
    assert any(name.endswith("training_debug.log") for name in names)


def make_dummy_backtest(net_pct=1.0, sharpe=1.1):
    def dummy_backtest(ensemble, data_full, indicators=None):
        return {
            "equity_curve": [],
            "effective_net_pct": net_pct,
            "inactivity_penalty": 0.0,
            "composite_reward": 1.0,
            "days_without_trading": 0,
            "trade_details": [],
            "days_in_profit": 0.0,
            "sharpe": sharpe,
            "max_drawdown": -0.1,
            "net_pct": net_pct,
            "trades": 1,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "avg_trade_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    return dummy_backtest


def dummy_stats(ec, trades, initial_balance=100.0):
    return None, ""


def build_dummy_ensemble(monkeypatch):
    device = get_device()
    ens = EnsembleModel(device=device, n_models=1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(8, 3))

        def forward(self, x):
            batch = x.size(0)
            logits = x.mean(dim=1) @ self.w
            return logits, types.SimpleNamespace(), torch.zeros(batch)

    ens.models = [DummyModel().to(device)]
    ens.optimizers = [torch.optim.Adam(ens.models[0].parameters(), lr=1e-3)]

    monkeypatch.setattr("artibot.ensemble.robust_backtest", make_dummy_backtest())
    monkeypatch.setattr("artibot.ensemble.compute_yearly_stats", dummy_stats)

    ds = TensorDataset(torch.zeros(1, 24, 8), torch.zeros(1, dtype=torch.long))
    dl = DataLoader(ds, batch_size=1)
    return ens, dl


def test_epoch_logging_includes_net_pct_and_lr(monkeypatch, caplog):
    monkeypatch.setattr(
        "torch.utils.data.DataLoader",
        lambda *a, **k: types.SimpleNamespace(_iterator=None),
    )
    monkeypatch.setattr("torch.utils.data.random_split", lambda ds, lens: (ds, ds))
    monkeypatch.setattr(
        "artibot.training.compute_indicators",
        lambda *a, **k: {
            "scaled": np.zeros((0, 16), dtype=np.float32),
            "mask": np.ones(16, dtype=bool),
        },
    )

    class DummyDS:
        def __init__(self, *a, **k):
            self.data = [0] * 10

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.zeros(24, 8), torch.tensor(0)

    monkeypatch.setattr("artibot.training.HourlyDataset", DummyDS)

    ens = EnsembleModel(device=torch.device("cpu"), n_models=1)
    monkeypatch.setattr(ens, "load_best_weights", lambda *a, **k: None)
    monkeypatch.setattr(ens, "optimize_models", lambda *a, **k: None)

    def fake_train_one_epoch(*a, **k):
        G.global_sharpe = 1.1
        G.global_max_drawdown = -0.1
        G.global_net_pct = 2.0
        return 0.1, 0.1

    monkeypatch.setattr(ens, "train_one_epoch", fake_train_one_epoch)

    data = [[i, 0, 0, 0, 0, 0] for i in range(10)]
    stop = threading.Event()
    caplog.set_level(logging.INFO)
    G.global_attention_entropy_history = [1.2]
    csv_training_thread(ens, data, stop, {}, use_prev_weights=False, max_epochs=1)

    record = next(r for r in caplog.records if r.message == "EPOCH")
    assert record.epoch == 1
    assert record.net_pct == 2.0
    assert hasattr(record, "lr")


def test_meta_mutation_logging(monkeypatch, caplog):
    class DummyOpt:
        def __init__(self):
            self.param_groups = [{"lr": 0.01, "weight_decay": 0.001}]

    class DummyEnsemble:
        def __init__(self):
            self.optimizers = [DummyOpt()]
            self.indicator_hparams = IndicatorHyperparams(
                rsi_period=14,
                sma_period=10,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
            )

    ens = DummyEnsemble()
    ens.hp = HyperParams(indicator_hp=ens.indicator_hparams)
    agent = MetaTransformerRL(ens)
    act = {k: 0 for k in ACTION_KEYS}
    act.update({"toggle_atr": 1})

    caplog.set_level(logging.INFO)
    agent.apply_action(ens.hp, ens.indicator_hparams, act)
    assert any("META_MUTATION" in r.message for r in caplog.records)
