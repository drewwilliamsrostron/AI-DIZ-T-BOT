import types
import sys
import numpy as np


def test_classify_market_regime_batch_stub(monkeypatch):
    """Batch helper returns list[int] same length as price series."""

    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")

    hmm_mod = types.ModuleType("hmmlearn.hmm")
    hmm_mod.GaussianHMM = object
    sys.modules.setdefault("hmmlearn", types.ModuleType("hmmlearn"))
    sys.modules["hmmlearn.hmm"] = hmm_mod

    kmeans_stub = types.SimpleNamespace(
        fit_predict=lambda x: np.zeros(len(x), dtype=int)
    )
    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = lambda *a, **k: kmeans_stub
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules["sklearn.cluster"] = cluster_mod

    import importlib
    import artibot.regime as rg

    importlib.reload(rg)

    prices = np.linspace(100, 120, 200, dtype=float)
    labels = rg.classify_market_regime_batch(prices)
    assert isinstance(labels, list) and len(labels) == len(prices)
    assert set(labels) == {0}

