import artibot.globals as G
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.rl import ACTION_KEYS, MetaTransformerRL


def test_apply_action_syncs_globals():
    hp = HyperParams()
    ind_hp = IndicatorHyperparams()
    agent = MetaTransformerRL(ensemble=None)

    act = {k: 0 for k in ACTION_KEYS}
    act.update({"toggle_vortex": 1, "d_vortex_period": 2, "d_sl": -1.0, "d_tp": 1.5})

    prev_use = ind_hp.use_vortex
    agent.apply_action(hp, ind_hp, act)

    assert ind_hp.use_vortex == (not prev_use)
    assert ind_hp.vortex_period == 16
    assert hp.sl == 4.0
    assert hp.tp == 6.5
    assert G.global_VORTEX_period == ind_hp.vortex_period
    assert G.global_TP_multiplier == hp.tp
