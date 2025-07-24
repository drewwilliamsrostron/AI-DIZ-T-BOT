import importlib

from artibot import hyperparams


def test_hyperparams_defaults():
    importlib.reload(hyperparams)
    hp = hyperparams.HyperParams()
    assert 1e-4 < hp.learning_rate < 1e-3
    assert isinstance(hp.use_sma, bool)


def test_indicator_defaults_one_and_true():
    importlib.reload(hyperparams)
    ihp = hyperparams.IndicatorHyperparams()
    for f in hyperparams.fields(hyperparams.IndicatorHyperparams):
        val = getattr(ihp, f.name)
        if f.name.startswith("use_"):
            assert val is True
        else:
            assert val == 1
