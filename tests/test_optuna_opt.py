from dataclasses import fields
import random

import sys
import types

optuna_stub = types.ModuleType("optuna")
optuna_stub.pruners = types.ModuleType("optuna.pruners")
optuna_stub.samplers = types.ModuleType("optuna.samplers")
optuna_stub.trial = types.ModuleType("optuna.trial")

def _dummy_study(*a, **k):
    class DummyStudy:
        def __init__(self) -> None:
            self.best_trial = types.SimpleNamespace(params={})

        def optimize(self, func, n_trials=1):
            trial = types.SimpleNamespace(
                suggest_int=lambda *a, **k: 1,
                suggest_float=lambda *a, **k: 0.1,
                suggest_categorical=lambda *a, **k: True,
                report=lambda *a, **k: None,
                should_prune=lambda: False,
            )
            for _ in range(n_trials):
                func(trial)

    return DummyStudy()

optuna_stub.create_study = _dummy_study
sys.modules.setdefault("optuna", optuna_stub)
sys.modules.setdefault("optuna.pruners", optuna_stub.pruners)
sys.modules.setdefault("optuna.samplers", optuna_stub.samplers)
sys.modules.setdefault("optuna.trial", optuna_stub.trial)

from artibot.optuna_opt import run_search
from artibot.hyperparams import IndicatorHyperparams


def test_run_search_ranges(monkeypatch):
    class DummyModel:
        def __init__(self, *a, **k):
            self.indicator_hparams = None

    monkeypatch.setattr("artibot.optuna_opt.EnsembleModel", DummyModel)
    monkeypatch.setattr("artibot.optuna_opt.G.push_backtest_metrics", lambda *_a, **_k: None)
    def dummy_train(*a, **k):
        return None

    def dummy_backtest(*_a, **_k):
        return {"composite_reward": random.random(), "trades": 1}

    monkeypatch.setattr("artibot.optuna_opt.csv_training_thread", dummy_train)
    monkeypatch.setattr("artibot.optuna_opt.robust_backtest", dummy_backtest)

    data = [[0, 0, 0, 0, 0]] * 30
    hp, lr, wd = run_search(data, n_trials=3)

    for f in fields(IndicatorHyperparams):
        if f.name.startswith("use_"):
            continue
        val = getattr(hp, f.name)
        assert 1 <= val <= 200
    assert 1e-5 <= lr <= 1e-3
    assert 0.0 <= wd <= 1e-3
