import os
import threading

import torch

from artibot.training import csv_training_thread
from artibot.ensemble import EnsembleModel


def test_csv_thread_uses_config(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    called = {}

    def fake_loader(*args, **kwargs):
        called["workers"] = kwargs.get("num_workers")
        called["persistent"] = kwargs.get("persistent_workers")

        class DL:
            pass

        return DL()

    monkeypatch.setattr("torch.utils.data.DataLoader", fake_loader)
    monkeypatch.setattr("torch.utils.data.random_split", lambda ds, lens: (ds, ds))
    monkeypatch.setattr(
        "artibot.training.compute_indicators",
        lambda *a, **k: {"sma": [], "rsi": [], "macd": [], "scaled": []},
    )

    ens = EnsembleModel(device=torch.device("cpu"), n_models=1)
    monkeypatch.setattr(ens, "load_best_weights", lambda *a, **k: None)
    monkeypatch.setattr(ens, "optimize_models", lambda *a, **k: None)
    monkeypatch.setattr(ens, "train_one_epoch", lambda *a, **k: (0.0, 0.0))

    data = [[i, 0, 0, 0, 0, 0] for i in range(50)]
    stop = threading.Event()
    csv_training_thread(
        ens, data, stop, {"NUM_WORKERS": 3}, use_prev_weights=False, max_epochs=1
    )

    assert called.get("workers") == 3
    assert called.get("persistent") is True


def test_persistent_workers_enabled(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    called = {}

    def fake_loader(*args, **kwargs):
        called["persistent"] = kwargs.get("persistent_workers")

        class DL:
            pass

        return DL()

    monkeypatch.setattr("torch.utils.data.DataLoader", fake_loader)
    monkeypatch.setattr("torch.utils.data.random_split", lambda ds, lens: (ds, ds))
    monkeypatch.setattr(
        "artibot.training.compute_indicators",
        lambda *a, **k: {"sma": [], "rsi": [], "macd": [], "scaled": []},
    )

    ens = EnsembleModel(device=torch.device("cpu"), n_models=1)
    monkeypatch.setattr(ens, "load_best_weights", lambda *a, **k: None)
    monkeypatch.setattr(ens, "optimize_models", lambda *a, **k: None)
    monkeypatch.setattr(ens, "train_one_epoch", lambda *a, **k: (0.0, 0.0))

    data = [[i, 0, 0, 0, 0, 0] for i in range(50)]
    stop = threading.Event()
    csv_training_thread(ens, data, stop, {}, use_prev_weights=False, max_epochs=1)

    assert called.get("persistent") is True
