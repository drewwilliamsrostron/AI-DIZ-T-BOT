import os
import sys
import types

import numpy as np

# ruff: noqa: E402
import torch

# Stub external dependencies required by artibot.globals
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

sys.modules["openai"] = types.SimpleNamespace()
sys.modules["talib"] = types.SimpleNamespace(
    RSI=lambda arr, timeperiod=14: np.zeros_like(arr),
    MACD=lambda arr, fastperiod=12, slowperiod=26, signalperiod=9: (
        np.zeros_like(arr),
        np.zeros_like(arr),
        np.zeros_like(arr),
    ),
)

from artibot.backtest import robust_backtest
from artibot.dataset import IndicatorHyperparams


class DummyEnsemble:
    """Ensemble producing constant predictions."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.indicator_hparams = IndicatorHyperparams(14, 10, 12, 26, 9)

    def vectorized_predict(self, windows_t, batch_size: int = 512):
        preds = torch.zeros(len(windows_t), dtype=torch.long)
        avg = {
            "sl_multiplier": torch.tensor(1.0),
            "tp_multiplier": torch.tensor(1.0),
            "risk_fraction": torch.tensor(0.1),
        }
        return preds, None, avg


def test_robust_backtest_simple():
    data = [
        [
            i * 3600,
            100 + i * 0.1,
            100 + i * 0.1 + 0.05,
            100 + i * 0.1 - 0.05,
            100 + i * 0.1,
            0,
        ]
        for i in range(30)
    ]
    result = robust_backtest(DummyEnsemble(), data)
    assert result["trades"] == 1

    assert round(result["net_pct"], 2) == 1.62
    assert -100.0 <= result["composite_reward"] <= 100.0
