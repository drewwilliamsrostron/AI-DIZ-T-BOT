from artibot.optuna_opt import run_bohb
from artibot.hyperparams import IndicatorHyperparams
from dataclasses import fields
import optuna
import pytest


def test_bohb_range():
    if not hasattr(optuna, "samplers"):
        pytest.skip("optuna stub without samplers")
    hp, params = run_bohb(n_trials=3)
    for f in fields(IndicatorHyperparams):
        if f.type is int and not f.name.startswith("use_"):
            val = getattr(hp, f.name)
            assert 1 <= val <= 200
    assert 1e-5 <= params["learning_rate"] <= 1e-2
    assert 1e-6 <= params["weight_decay"] <= 1e-2

