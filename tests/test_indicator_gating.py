import artibot.globals as G
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.rl import MetaTransformerRL


def test_toggle_ema_updates_globals():
    hp = HyperParams()
    ind_hp = IndicatorHyperparams()
    agent = MetaTransformerRL(ensemble=None)
    prev = ind_hp.use_ema
    agent.apply_action(hp, ind_hp, {"toggle_ema": 1})
    assert ind_hp.use_ema != prev
    assert G.global_use_EMA == ind_hp.use_ema
