import numpy as np
import torch
from artibot.rl import MetaTransformerRL
from artibot.hyperparams import HyperParams


class DummyEnsemble:
    def __init__(self):
        self.hp = HyperParams(freeze_features=True)


def test_meta_freeze():
    agent = MetaTransformerRL(DummyEnsemble())
    act, _, _ = agent.pick_action(np.zeros(agent.state_dim, dtype=np.float32))
    assert not any(k.startswith("toggle_") or k.endswith("_period_delta") for k in act)
