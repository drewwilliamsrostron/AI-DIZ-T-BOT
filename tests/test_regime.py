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
