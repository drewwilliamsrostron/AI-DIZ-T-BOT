import torch
import artibot.training as training
from artibot.hyperparams import IndicatorHyperparams
from artibot.constants import FEATURE_DIMENSION


def make_data(n: int = 40):
    row = [0.0] * FEATURE_DIMENSION
    return [row[:] for _ in range(n)]


def test_indicator_flow():
    data = make_data(40)
    hp = IndicatorHyperparams(sma_period=29)
    training.walk_forward_backtest(data, 20, 10, indicator_hp=hp)
    assert hp.sma_period == 29
    training.walk_forward_backtest(data, 20, 10, indicator_hp=hp)
    assert hp.sma_period == 29
