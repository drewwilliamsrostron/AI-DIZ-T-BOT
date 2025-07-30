import numpy as np
import pytest
import artibot.backtest as bt


class DummyEnsemble:
    device = "cpu"

    def vectorized_predict(self, w, batch_size, regime_labels=None):
        return (
            np.zeros(len(w), dtype=int),
            None,
            {
                "sl_multiplier": np.array(1.0),
                "tp_multiplier": np.array(1.0),
                "risk_fraction": np.array(0.1),
            },
        )


def test_raw_ohlc_only():
    dummy = DummyEnsemble()
    raw = np.tile([1_600_000_000, 1, 1, 1, 1, 100], (25, 1))
    bt.robust_backtest(dummy, raw)
    bad = np.random.rand(25, 16)
    with pytest.raises(ValueError):
        bt.robust_backtest(dummy, bad)
