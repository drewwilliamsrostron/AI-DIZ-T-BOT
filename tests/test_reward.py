import os, sys, types
import numpy as np
sys.modules['openai'] = types.SimpleNamespace()
sys.modules['talib'] = types.SimpleNamespace(RSI=lambda a, timeperiod=14: np.zeros_like(a), MACD=lambda a, fastperiod=12, slowperiod=26, signalperiod=9: (np.zeros_like(a),np.zeros_like(a),np.zeros_like(a)), EMA=lambda a, timeperiod=20: np.zeros_like(a))
from artibot.backtest import robust_backtest
from artibot.hyperparams import IndicatorHyperparams

class DummyEnsemble:
    def __init__(self):
        self.device = 'cpu'
        self.indicator_hparams = IndicatorHyperparams()
    def vectorized_predict(self, windows_t, batch_size=512):
        preds = np.zeros(len(windows_t), dtype=np.int64)
        import torch
        avg = {
            "sl_multiplier": torch.tensor(1.0),
            "tp_multiplier": torch.tensor(1.0),
            "risk_fraction": torch.tensor(0.1),
        }
        return torch.tensor(preds), None, avg

def test_inactivity_penalty_applied():
    data = [[i*86400,100,101,99,100,0] for i in range(60)]
    result = robust_backtest(DummyEnsemble(), data)
    assert result['inactivity_penalty'] > 0
    assert 0 <= result['days_in_profit'] / 365 <= 1
