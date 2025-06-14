import artibot.globals as G
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.rl import ACTION_KEYS, MetaTransformerRL


def test_apply_action_syncs_globals():
    hp = HyperParams()
    ind_hp = IndicatorHyperparams()
    agent = MetaTransformerRL(ensemble=None)

    act = {k: 0 for k in ACTION_KEYS}
    act.update(
        {
            "toggle_vortex": 1,
            "d_vortex_period": 2,
            "d_sl": -1.0,
            "d_tp": 1.5,
            "d_long_frac": 0.02,
            "d_short_frac": -0.01,
        }
    )

    prev_use = ind_hp.use_vortex
    agent.apply_action(hp, ind_hp, act)

    assert ind_hp.use_vortex == (not prev_use)
    assert ind_hp.vortex_period == 16
    assert hp.sl == 4.0
    assert hp.tp == 6.5
    assert G.global_VORTEX_period == ind_hp.vortex_period
    assert G.global_TP_multiplier == hp.tp
    assert G.global_RSI_period == ind_hp.rsi_period
    assert hp.long_frac > 0
    assert hp.short_frac == 0
    assert G.global_long_frac == hp.long_frac
    assert G.global_short_frac == hp.short_frac


def test_toggle_ema_period_updates_globals():
    hp = HyperParams()
    ind_hp = IndicatorHyperparams()
    agent = MetaTransformerRL(ensemble=None)

    act = {k: 0 for k in ACTION_KEYS}
    act.update({"toggle_ema": 1, "d_ema_period": 3})

    prev_use = ind_hp.use_ema
    prev_period = ind_hp.ema_period
    agent.apply_action(hp, ind_hp, act)

    assert ind_hp.use_ema == (not prev_use)
    assert ind_hp.ema_period == prev_period + 3 if prev_period + 3 >= 2 else 2
    assert G.global_EMA_period == ind_hp.ema_period
