import types

import artibot.training as training
from artibot.hyperparams import HyperParams, IndicatorHyperparams


def test_adjust_for_regime_changes(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")
    monkeypatch.setattr(training.G, "sync_globals", lambda *a, **k: None)
    training.G.current_regime = None

    ens = types.SimpleNamespace(
        hp=HyperParams(), indicator_hparams=IndicatorHyperparams()
    )

    training.adjust_for_regime(1, ens)
    assert ens.hp.long_frac == 0.0
    assert ens.hp.short_frac == 0.0
    assert ens.indicator_hparams.use_tenkan is False
    assert ens.indicator_hparams.use_kijun is False
    assert training.G.current_regime == 1

    training.adjust_for_regime(0, ens)
    assert ens.hp.long_frac == 0.1
    assert ens.hp.short_frac == 0.1
    assert ens.indicator_hparams.use_tenkan is True
    assert ens.indicator_hparams.use_kijun is True
    assert training.G.current_regime == 0
