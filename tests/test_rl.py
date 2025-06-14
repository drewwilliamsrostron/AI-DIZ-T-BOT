import importlib
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import torch
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

    def status_sleep(primary: str, secondary: str, t: float) -> None:

        pass

    stub.status_sleep = status_sleep

    def set_status(primary: str, secondary: str) -> None:
        stub.global_status_primary = primary
        stub.global_status_secondary = secondary

    def get_status_full() -> tuple[str, str]:
        return stub.global_status_primary, stub.global_status_secondary

    stub.set_status = set_status
    stub.get_status_full = get_status_full
    stub.sync_globals = lambda *a, **k: None
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
    stub.global_status_primary = ""
    stub.global_status_secondary = ""
    stub.timeline_depth = 300
    stub.timeline_index = 0
    stub.timeline_ind_on = np.zeros((stub.timeline_depth, 6), dtype=np.uint8)
    stub.timeline_trades = np.zeros(stub.timeline_depth, dtype=np.uint8)
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


def test_pick_action_returns_dict():
    rl = load_rl_module()
    agent = rl.MetaTransformerRL(ensemble=None)
    act, logp, val = agent.pick_action(np.zeros(agent.model.state_dim))
    assert isinstance(act, dict)
    assert set(act.keys()).issubset(set(rl.ACTION_KEYS))
    assert torch.is_tensor(logp)
    assert torch.is_tensor(val)


def test_pick_action_deterministic_without_exploration():
    rl = load_rl_module()
    agent = rl.MetaTransformerRL(ensemble=None)
    agent.eps_start = 0.0
    agent.eps_end = 0.0
    rl.torch.manual_seed(0)
    rl._random.seed(0)
    state = rl.np.zeros(agent.model.state_dim)
    act1, _, _ = agent.pick_action(state)
    rl.torch.manual_seed(0)
    rl._random.seed(0)
    agent.steps = 0
    act2, _, _ = agent.pick_action(state)
    assert act1 == act2


def test_apply_action_custom_space():
    rl = load_rl_module()

    class DummyOpt:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.01, "weight_decay": 0.001}]

    class DummyEnsemble:
        def __init__(self) -> None:
            self.optimizers = [DummyOpt()]
            self.indicator_hparams = rl.IndicatorHyperparams(
                rsi_period=14,
                sma_period=10,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
            )

    ensemble = DummyEnsemble()
    ensemble.hp = rl.HyperParams(indicator_hp=ensemble.indicator_hparams)
    agent = rl.MetaTransformerRL(ensemble)
    act = {k: 0 for k in rl.ACTION_KEYS}
    act.update({"d_sl": -1, "d_tp": 0.5})

    agent.apply_action(ensemble.hp, ensemble.indicator_hparams, act)

    pg = ensemble.optimizers[0].param_groups[0]
    assert pg["lr"] == pytest.approx(0.01)
    assert pg["weight_decay"] == pytest.approx(0.001)
    assert ensemble.hp.sl == pytest.approx(4.0)
    assert ensemble.hp.tp == pytest.approx(5.5)


def test_update_nan_protection():
    rl = load_rl_module()

    class DummyModel(rl.torch.nn.Module):
        def __init__(self, n_actions) -> None:
            super().__init__()
            self.param = rl.torch.nn.Parameter(rl.torch.zeros(1))
            self.n_actions = n_actions

        def forward(self, x):
            val = rl.torch.tensor([rl.torch.inf])
            return rl.torch.zeros((1, self.n_actions)), val

    agent = rl.MetaTransformerRL(ensemble=None)
    agent.action_space = [(0.0,)]
    agent.model = DummyModel(len(agent.action_space))
    agent.n_actions = 1
    agent.opt = rl.optim.SGD(agent.model.parameters(), lr=0.1)

    agent.update(
        rl.np.zeros(agent.model.state_dim),
        0,
        rl.torch.inf,
        rl.np.zeros(agent.model.state_dim),
        rl.torch.tensor([0.0]),
        0.0,
    )

    for p in agent.model.parameters():
        assert rl.torch.isfinite(p).all()


def test_no_warnings(recwarn):
    import warnings  # noqa: F401
    import run_artibot  # noqa: F401

    assert not recwarn.list
