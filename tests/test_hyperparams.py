import importlib

from artibot import hyperparams


def test_hyperparams_defaults():
    importlib.reload(hyperparams)
    hp = hyperparams.HyperParams()
    assert 1e-4 < hp.learning_rate < 1e-3
    assert isinstance(hp.use_sma, bool)
