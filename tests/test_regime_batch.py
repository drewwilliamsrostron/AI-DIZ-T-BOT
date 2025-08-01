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
            return probs

    import importlib
    import artibot.regime as rg

    importlib.reload(rg)
    rg.RegimeEncoder = DummyEncoder

    prices = np.linspace(100, 120, 200, dtype=float)
    labels = rg.classify_market_regime_batch(prices)
    assert isinstance(labels, list) and len(labels) == len(prices)
    assert set(labels) == {0}
