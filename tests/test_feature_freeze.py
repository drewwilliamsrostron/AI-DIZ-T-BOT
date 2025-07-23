import torch
import artibot.hyperparams as hp
import artibot.globals as G
from artibot.rl import MetaTransformerRL
from artibot.ensemble import EnsembleModel


def test_actions_apply_after_warmup(monkeypatch):
    ens = EnsembleModel(device=torch.device("cpu"), n_models=1)
    agent = MetaTransformerRL(ens)
    hp_inst = hp.HyperParams(indicator_hp=hp.IndicatorHyperparams(sma_period=10))
    ihp = hp_inst.indicator_hp
    sma_flag = ihp.use_sma
    period = ihp.sma_period
    monkeypatch.setattr(G, "get_warmup_step", lambda: hp.WARMUP_STEPS)
    agent.apply_action(hp_inst, ihp, {"toggle_sma": 1, "d_sma_period": 5})
    assert ihp.use_sma != sma_flag
    assert ihp.sma_period == period + 5
