import sys
import types
from importlib.machinery import ModuleSpec

# ---------------------------------------------------------------------------
# Stub heavy or GUI-only deps so the test suite can run in a slim environment
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------
from artibot.hyperparams import IndicatorHyperparams  # noqa: E402
from artibot.utils import feature_dim_for  # noqa: E402


def test_base_dim_all_on():
    """5 OHLCV + 3 context = 8 columns when all context flags are True
    and every technical indicator is OFF."""
    hp = IndicatorHyperparams(
        use_sma=False,
        use_rsi=False,
        use_macd=False,
        use_atr=False,
        use_ema=False,
    )
    assert feature_dim_for(hp) == 8


def test_toggle_context_flags():
    """Turn off sentiment & realised-vol, keep macro ⇒ 5 + 1 = 6."""
    hp = IndicatorHyperparams(
        use_sentiment=False,
        use_macro=True,
        use_rvol=False,
        use_sma=False,
        use_rsi=False,
        use_macd=False,
        use_atr=False,
        use_ema=False,
    )
    assert feature_dim_for(hp) == 6


def test_add_technical_flags():
    """No context, ATR ON ⇒ 5 + 1 tech = 6."""
    hp = IndicatorHyperparams(
        use_atr=True,
        use_sentiment=False,
        use_macro=False,
        use_rvol=False,
        use_sma=False,
        use_rsi=False,
        use_macd=False,
        use_ema=False,
    )
    assert feature_dim_for(hp) == 6


def test_feature_dim_matches_flags():
    """Context default ON (3 cols) + ATR + MACD ⇒ 5 + 3 + 2 = 10."""
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
    expected = 5 + 3 + 2  # OHLCV + context + (ATR, MACD)
    assert feature_dim_for(hp) == expected
