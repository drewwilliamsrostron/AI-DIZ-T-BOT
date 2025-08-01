import types
import sys

import numpy as np


def test_detect_volatility_regime(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")

    class DummyHMM:
        def __init__(self, *a, **k):
            pass

        def fit(self, X):
            self.X = X

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    hmm_mod = types.ModuleType("hmmlearn.hmm")
    hmm_mod.GaussianHMM = DummyHMM
    sys.modules.setdefault("hmmlearn", types.ModuleType("hmmlearn"))
    sys.modules["hmmlearn.hmm"] = hmm_mod

    from artibot.regime import detect_volatility_regime

    prices = np.linspace(1, 10, 10)
    regime = detect_volatility_regime(prices)
    assert regime == 0


def test_classify_market_regime(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")

    class DummyEncoder:
        def __init__(self, *a, **k):
            self.n_regimes = k.get("n_regimes", 3)
            self.seq_len = k.get("seq_len", 32)

        def train_unsupervised(self, *a, **k):
            pass

        def encode_sequence(self, prices, *a, **k):
            num = len(prices) - self.seq_len + 1
            if num <= 0:
                return np.empty((0, self.n_regimes))
            probs = np.zeros((num, self.n_regimes), dtype=float)
            probs[:, 1] = 1.0
            return probs

    import importlib
    import artibot.regime as regime

    importlib.reload(regime)
    regime.RegimeEncoder = DummyEncoder
    regime._last_regime_encoder = DummyEncoder()
    classify_market_regime = regime.classify_market_regime

    prices = np.linspace(1.0, 10.0, 200)
    label = classify_market_regime(prices)
    assert label == 1
