import importlib
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import torch
import random
from typing import NamedTuple
from torch.utils.data import Dataset
import random as _random
import pytest


def load_rl_module():
    """Import artibot.rl with stubbed dependencies and no package side effects."""
    stub = types.ModuleType("artibot.globals")
    stub.np = np
    stub.torch = torch
    stub.nn = torch.nn
    stub.optim = torch.optim
    stub.NamedTuple = NamedTuple
    stub.Dataset = Dataset
    stub.random = _random

    def status_sleep(comp: str, msg: str, t: float) -> None:
        pass

    stub.status_sleep = status_sleep

    def set_status(comp: str, msg: str | None = None) -> None:
        stub.global_status_message = f"[{comp}] {msg}" if msg is not None else comp

    def get_status() -> str:
        return stub.global_status_message

    stub.set_status = set_status
    stub.get_status = get_status
    stub.GLOBAL_THRESHOLD = 0.0
    stub.global_ai_adjustments = ""
    stub.global_ai_adjustments_log = ""
    stub.epoch_count = 0
    stub.global_min_hold_seconds = 1800
    stub.global_composite_reward = 0.0
    stub.global_best_composite_reward = 0.0
    stub.global_sharpe = 0.0
    stub.global_max_drawdown = 0.0
    stub.global_num_trades = 0
    stub.global_days_in_profit = 0
    stub.global_status_message = ""
    sys.modules["artibot.globals"] = stub

    base = Path(__file__).resolve().parent.parent / "artibot"
    pkg = types.ModuleType("artibot")
    pkg.__path__ = [str(base)]
    sys.modules["artibot"] = pkg

    for name in ("dataset", "model", "rl"):
        spec = importlib.util.spec_from_file_location(
            f"artibot.{name}", base / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"artibot.{name}"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules["artibot.rl"]


def test_find_nearest_action():
    rl = load_rl_module()
    agent = rl.MetaTransformerRL(ensemble=None)
    agent.action_space = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    idx = agent._find_nearest_action([1.1, 0.9])
    assert idx == 1


def test_pick_action_deterministic(monkeypatch):
    rl = load_rl_module()
    agent = rl.MetaTransformerRL(ensemble=None)
    agent.action_space = [(0,), (1,), (2,)]

    def fake_forward(self, x):
        return torch.zeros((1, len(agent.action_space))), torch.tensor([[0.5]])

    monkeypatch.setattr(
        agent.model, "forward", fake_forward.__get__(agent.model, type(agent.model))
    )

    class DummyDist:
        def __init__(self, logits=None):
            pass

        def sample(self):
            return torch.tensor(2)

        def log_prob(self, value):
            return torch.tensor([0.25])

    monkeypatch.setattr(random, "random", lambda: 1.0)
    monkeypatch.setattr(rl.torch.distributions, "Categorical", DummyDist)

    idx, logp, value = agent.pick_action(np.zeros(agent.model.state_dim))
    assert idx == 2
    assert value == 0.5
    assert torch.isclose(logp, torch.tensor([0.25])).item()


def test_apply_action_custom_space():
    rl = load_rl_module()

    class DummyOpt:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.01, "weight_decay": 0.001}]

    class DummyEnsemble:
        def __init__(self) -> None:
            self.optimizers = [DummyOpt()]
            self.indicator_hparams = rl.IndicatorHyperparams(14, 10, 12, 26, 9)

    ensemble = DummyEnsemble()
    agent = rl.MetaTransformerRL(ensemble)
    agent.action_space = [(0.1, 0.2, 1, 2, 3, 4, 5, 0.05)]
    agent.model.action_space = [(0.0,) * 8]

    result = agent.apply_action(0)

    assert result == pytest.approx(
        (
            0.011,
            0.0012,
            15,
            12,
            15,
            30,
            14,
            0.05,
        ),
        rel=1e-5,
    )

    pg = ensemble.optimizers[0].param_groups[0]
    assert pg["lr"] == pytest.approx(0.011)
    assert pg["weight_decay"] == pytest.approx(0.0012)
    assert ensemble.indicator_hparams == rl.IndicatorHyperparams(15, 12, 15, 30, 14)


def test_no_warnings(recwarn):
    import warnings  # noqa: F401
    import run_artibot  # noqa: F401

    assert not recwarn.list
