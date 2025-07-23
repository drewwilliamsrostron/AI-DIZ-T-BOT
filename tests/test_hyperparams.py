import importlib

from artibot import hyperparams


def test_hyperparams_defaults():
    importlib.reload(hyperparams)
    hp = hyperparams.HyperParams()
    assert 1e-4 < hp.learning_rate < 1e-3
    assert isinstance(hp.use_sma, bool)


def test_indicator_defaults_none():
    importlib.reload(hyperparams)
    ihp = hyperparams.IndicatorHyperparams()
    for f in hyperparams.fields(hyperparams.IndicatorHyperparams):
        if f.name.startswith("use_"):
            continue
        assert getattr(ihp, f.name) is None
