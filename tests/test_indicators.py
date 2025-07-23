import numpy as np
from artibot import indicators
from artibot import dataset
from artibot.hyperparams import IndicatorHyperparams


def test_vortex_length():
    high = np.arange(10.0)
    low = high - 1
    close = high - 0.5
    vp, vn = indicators.vortex(high, low, close, period=3)
    assert len(vp) == len(close)
    assert len(vn) == len(close)


def test_cmf_length():
    high = np.linspace(1, 10, 10)
    low = high - 0.5
    close = high - 0.25
    vol = np.ones_like(high)
    out = indicators.cmf(high, low, close, vol, period=4)
    assert len(out) == len(close)


def test_dataset_columns_flags():
    data = [[i, 1.0, 1.1, 0.9, 1.0, 1.0] for i in range(25)]
    base = {
        "sma_period": 5,
        "rsi_period": 10,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "vortex_period": 14,
        "cmf_period": 20,
        "ema_period": 20,
        "donchian_period": 20,
        "kijun_period": 26,
        "tenkan_period": 9,
        "displacement": 26,
    }
    hp = IndicatorHyperparams(**base)
    ds = dataset.HourlyDataset(data, seq_len=3, indicator_hparams=hp)
    sample, _ = ds[0]
    base_cols = sample.shape[1]
    hp_off = IndicatorHyperparams(
        use_atr=False, use_vortex=False, use_cmf=False, **base
    )
    ds_off = dataset.HourlyDataset(data, seq_len=3, indicator_hparams=hp_off)
    sample_off, _ = ds_off[0]
    assert sample_off.shape[1] < base_cols
