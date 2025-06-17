import sys
import types
from importlib.machinery import ModuleSpec

# ruff: noqa: E402

# stub heavy deps so utils can import without installing them
for name in [
    "torch",
    "pandas",
    "numpy",
    "openai",
    "ccxt",
    "tkinter",
    "tkinter.ttk",
]:
    sys.modules.setdefault(name, types.ModuleType(name))

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules.setdefault("matplotlib", matplotlib)
pyplot = types.ModuleType("pyplot")
pyplot.__spec__ = ModuleSpec("pyplot", loader=None)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules.setdefault("matplotlib.backends.backend_tkagg", backend)

from artibot.utils import feature_dim_for
from artibot.hyperparams import IndicatorHyperparams


def test_feature_dim_matches_flags():
    hp = IndicatorHyperparams(
        use_atr=True,
        use_vortex=False,
        use_macd=True,
        use_sma=False,
        use_rsi=False,
        use_ema=False,
        use_cmf=False,
        use_donchian=False,
        use_kijun=False,
        use_tenkan=False,
        use_displacement=False,
    )
    assert feature_dim_for(hp) == 4 + 2
