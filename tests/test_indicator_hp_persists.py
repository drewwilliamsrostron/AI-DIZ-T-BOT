import artibot.ensemble as ens
from artibot.hyperparams import HyperParams, IndicatorHyperparams


def test_indicator_hp_sync_property():
    model = ens.EnsembleModel.__new__(ens.EnsembleModel)
    model._indicator_hparams = ens.IndicatorHyperparams()
    model._hp = HyperParams(indicator_hp=model._indicator_hparams)
    ihp = IndicatorHyperparams(sma_period=45)
    model.indicator_hparams = ihp
    assert model.hp.indicator_hp is ihp

    new_hp = HyperParams(indicator_hp=IndicatorHyperparams(rsi_period=30))
    model.hp = new_hp
    assert model.indicator_hparams is new_hp.indicator_hp
