import logging
import json
import pytest
import threading

import torch

import artibot.globals as g
from artibot.dataset import load_csv_hourly
from artibot.ensemble import EnsembleModel
from artibot.training import csv_training_thread


def test_loss_regression(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[:100]
    ens = EnsembleModel(device=torch.device("cpu"), n_models=1, lr=1e-4)
    stop = threading.Event()
    csv_training_thread(
        ens,
        data,
        stop,
        {"ADAPT_TO_LIVE": False, "NUM_WORKERS": 0},
        use_prev_weights=False,
        max_epochs=3,
    )
    msgs = [r.message for r in caplog.records]
    assert not any("ValueError: Tried to step" in m for m in msgs)
    n = min(len(g.global_training_loss), len(g.global_validation_loss))
    tr = g.global_training_loss[:n]
    val = g.global_validation_loss[:n]
    assert tr and val
    assert max(tr) <= max(val) * 1.05


@pytest.fixture
def dummy_checkpoint(tmp_path):
    path = tmp_path / "ckpt.json"
    path.write_text(json.dumps({"foo": "bar"}))
    return str(path)


def test_best_reward_never_none_after_state_load(dummy_checkpoint):
    from artibot import globals as G, state

    state.load(dummy_checkpoint)
    assert G.global_best_composite_reward is not None
