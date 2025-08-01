import sys
import types
from importlib.machinery import ModuleSpec

for name in ["openai", "ccxt", "tkinter", "tkinter.ttk"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

import numpy as np
import artibot.regime_encoder as re

# ensure openai stub has a spec after import
sys.modules["openai"].__spec__ = ModuleSpec("openai", loader=None)

def test_encoder_train_and_encode(monkeypatch):
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")
    prices = np.linspace(100.0, 110.0, 128)
    enc = re.RegimeEncoder(seq_len=16, n_regimes=3)
    enc.train_unsupervised(prices, epochs=1)
    probs = enc.encode_sequence(prices)
    assert probs.shape[1] == 3
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
