
import artibot.rl as rl
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.ensemble import EnsembleModel
import torch


class DummyEnsemble:
    def __init__(self):
        self.hp = HyperParams(indicator_hp=IndicatorHyperparams(), freeze_features=True)
        self.device = torch.device("cpu")
        self.optimizers = []


def test_meta_freeze(monkeypatch):
    ens = DummyEnsemble()
    agent = rl.MetaTransformerRL(ens)
    act, _, _ = agent.pick_action([0.0] * agent.state_dim)
    assert not any(k.startswith("toggle_") or k.endswith("_period_delta") for k in act)
