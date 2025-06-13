import numpy as np
from artibot import indicators
from artibot import dataset


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
    hp = dataset.IndicatorHyperparams()
    ds = dataset.HourlyDataset(data, seq_len=3, indicator_hparams=hp)
    sample, _ = ds[0]
    base_cols = sample.shape[1]
    hp_off = dataset.IndicatorHyperparams(use_atr=False, use_vortex=False, use_cmf=False)
    ds_off = dataset.HourlyDataset(data, seq_len=3, indicator_hparams=hp_off)
    sample_off, _ = ds_off[0]
    assert sample_off.shape[1] < base_cols
