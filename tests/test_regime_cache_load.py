import threading
import numpy as np
import torch
import types

import artibot.training as training


def test_regime_cache_load(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")
    monkeypatch.setattr(training.G, "sync_globals", lambda *a, **k: None)

    class DummyDL:
        def __len__(self):
            return 1

    import torch.utils.data as tud

    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: DummyDL())
    monkeypatch.setattr(tud, "Subset", lambda ds, idx: ds, raising=False)
    monkeypatch.setattr(tud, "random_split", lambda ds, lens: (ds, ds))
    monkeypatch.setattr(torch, "randn", lambda *a, **k: torch.zeros(*a))
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

    states = iter([0, 1, 1, 1])
    monkeypatch.setattr(training, "classify_market_regime", lambda prices: next(states))

    load_calls = {"n": 0}

    def fake_load(regime, ens):
        load_calls["n"] += 1
        return {"sharpe": 1.5, "net_pct": 10.0}

    monkeypatch.setattr("artibot.regime_cache.load_best_for_regime", fake_load)
    monkeypatch.setattr(training, "robust_backtest", lambda *a, **k: {"sharpe": 1.3})

    quick_calls = {"n": 0}
    monkeypatch.setattr(
        training,
        "quick_fit",
        lambda *a, **k: quick_calls.__setitem__("n", quick_calls["n"] + 1),
    )

    class DummyModel:
        input_dim = 8
        input_size = 8

        def to(self, *a, **k):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, *a, **k):
            pass

    class DummyEnsemble:
        def __init__(self) -> None:
            self.models = [DummyModel()]
            self.device = torch.device("cpu")
            self.train_steps = 0
            self.cycle = []
            self.hp = training.hyperparams.HyperParams()
            self.indicator_hparams = training.hyperparams.IndicatorHyperparams()
            self.reward_loss_weight = 0.0
            self.max_reward_loss_weight = 1.0
            self.optimizers = [types.SimpleNamespace(param_groups=[{"lr": 0.0}])]

        def configure_one_cycle(self, *a, **k):
            pass

        def optimize_models(self, *a, **k):
            pass

        def load_best_weights(self, *a, **k):
            pass

        def train_one_epoch(self, *a, **k):
            return 0.0, 0.0

        def save_best_weights(self, *a, **k):
            pass

    ens = DummyEnsemble()

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

    assert load_calls["n"] == 1
    assert quick_calls["n"] == 0
