import threading
import numpy as np
import torch

import artibot.training as training
from artibot.ensemble import EnsembleModel


def test_regime_retrain_trigger(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")
    monkeypatch.setattr(training.G, "sync_globals", lambda *a, **k: None)

    # Minimal dataset + dummy loaders
    class DummyDL:
        def __len__(self):
            return 1

    monkeypatch.setattr(
        "torch.utils.data.DataLoader",
        lambda *a, **k: DummyDL(),
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

    # Simulate regime shift persisting for 3 epochs
    states = iter([0, 1, 1, 1])
    monkeypatch.setattr(
        training, "detect_volatility_regime", lambda prices: next(states)
    )

    retrain_calls = {"n": 0}
    monkeypatch.setattr(
        training,
        "quick_fit",
        lambda *a, **k: retrain_calls.__setitem__("n", retrain_calls["n"] + 1),
    )

    ens = EnsembleModel(device=torch.device("cpu"), n_models=1)
    monkeypatch.setattr(ens, "load_best_weights", lambda *a, **k: None)
    monkeypatch.setattr(ens, "optimize_models", lambda *a, **k: None)
    monkeypatch.setattr(ens, "train_one_epoch", lambda *a, **k: (0.0, 0.0))

    data = [[i, 0, 0, 0, 0, 0] for i in range(50)]
    stop = threading.Event()
    training.csv_training_thread(
        ens,
        data,
        stop,
        {},
        use_prev_weights=False,
        max_epochs=4,
        update_globals=False,
    )

    assert retrain_calls["n"] == 1
