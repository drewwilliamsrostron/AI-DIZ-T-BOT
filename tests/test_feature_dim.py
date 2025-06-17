from artibot.hyperparams import IndicatorHyperparams
from artibot.utils import feature_dim_for


def test_base_dim_all_on():
    hp = IndicatorHyperparams(
        use_sma=False,
        use_rsi=False,
        use_macd=False,
        use_atr=False,
        use_ema=False,
    )
    assert feature_dim_for(hp) == 8


def test_toggle_context_flags():
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
