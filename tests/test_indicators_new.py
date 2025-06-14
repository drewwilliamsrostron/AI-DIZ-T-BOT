import numpy as np
from artibot import indicators


def test_ema_length():
    closes = np.linspace(1, 10, 10)
    ema = indicators.ema(closes, period=3)
    assert len(ema) == len(closes)


def test_donchian_bounds():
    highs = np.linspace(5, 9, 5)
    lows = highs - 2
    up, lo, mid = indicators.donchian(highs, lows, period=3)
    assert len(up) == len(highs)
    assert np.all(up >= mid)
    assert np.all(mid >= lo)
