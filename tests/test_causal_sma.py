import numpy as np
from artibot.dataset import trailing_sma


def test_trailing_sma_is_causal():
    close = np.arange(1, 101)
    sma = trailing_sma(close, 10)
    for t in range(9, 100):
        left = close[t - 9 : t + 1].mean()
        assert np.isclose(sma[t], left)
