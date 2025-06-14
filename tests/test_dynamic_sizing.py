import artibot.globals as G
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.rl import MetaTransformerRL


def test_dynamic_sizing_gross_cap():
    hp = HyperParams(long_frac=0.06, short_frac=0.08)
    ind_hp = IndicatorHyperparams()
    agent = MetaTransformerRL(ensemble=None)
    act = {"d_long_frac": 0.04, "d_short_frac": -0.06}
    agent.apply_action(hp, ind_hp, act)
    assert hp.long_frac <= G.MAX_SIDE_EXPOSURE_PCT
    assert hp.long_frac + hp.short_frac <= G.MAX_GROSS_PCT
