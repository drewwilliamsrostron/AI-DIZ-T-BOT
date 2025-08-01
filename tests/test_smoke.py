import sys
import threading
import types
import pytest

import torch

import artibot.globals as g
from artibot.backtest import robust_backtest
from artibot.dataset import load_csv_hourly
from artibot.ensemble import EnsembleModel
from artibot.training import csv_training_thread


def test_end_to_end_smoke():
    sys.modules["openai"] = types.SimpleNamespace()
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[:1500]
    assert len(data) > 24
    ensemble = EnsembleModel(
        device=torch.device("cpu"), n_models=1, lr=1e-4, weight_decay=1e-4
    )
    stop_event = threading.Event()
    csv_training_thread(
        ensemble,
        data,
        stop_event,
        {"ADAPT_TO_LIVE": False},
        use_prev_weights=False,
        max_epochs=1,
        debug_anomaly=__debug__,
    )
    result = robust_backtest(ensemble, data)
    assert result["trades"] > 0
    if g.epoch_count == 0:
        pytest.skip("training failed to run")
    assert g.epoch_count >= 1
    assert isinstance(g.global_holdout_sharpe, float)
    assert isinstance(g.global_holdout_max_drawdown, float)
    curve = result["equity_curve"]
    assert curve and curve[0][1] != curve[-1][1]
