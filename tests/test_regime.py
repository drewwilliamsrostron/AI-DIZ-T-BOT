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

    class DummyKMeans:
        def __init__(self, *a, **k):
            pass

        def fit(self, X):
            self.X = X

        def predict(self, X):
            return np.array([0])

    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = DummyKMeans
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules["sklearn.cluster"] = cluster_mod

    from artibot.regime import detect_volatility_regime

    prices = np.linspace(1, 10, 10)
    regime = detect_volatility_regime(prices)
    assert regime == 0


def test_classify_market_regime(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")

    class DummyKMeans:
        def __init__(self, *a, **k):
            pass

        def fit(self, X):
            self.X = X

        def predict(self, X):
            return np.array([1])

    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = DummyKMeans
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules["sklearn.cluster"] = cluster_mod

    import importlib
    import artibot.regime as regime

    importlib.reload(regime)
    regime.KMeans = DummyKMeans
    classify_market_regime = regime.classify_market_regime

    prices = np.linspace(1.0, 10.0, 200)
    label = classify_market_regime(prices)
    assert label == 1
