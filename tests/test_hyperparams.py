import importlib

from artibot import hyperparams


def test_hyperparams_defaults():
    importlib.reload(hyperparams)
    hp = hyperparams.HyperParams()
    assert hp.learning_rate > 0
    assert isinstance(hp.use_sma, bool)
