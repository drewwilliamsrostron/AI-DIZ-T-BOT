"""Technical indicator helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def vortex(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """Return Vortex Indicator (VI+, VI-) for ``high``, ``low`` and ``close``.

    The calculation is causal: values at ``t`` only use data up to ``t``.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    prev_close = np.concatenate(([np.nan], close[:-1]))
    prev_low = np.concatenate(([np.nan], low[:-1]))
    prev_high = np.concatenate(([np.nan], high[:-1]))

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    vm_plus = np.abs(high - prev_low)
    vm_minus = np.abs(low - prev_high)

    tr_sum = pd.Series(tr).rolling(period, min_periods=1).sum()
    vp = pd.Series(vm_plus).rolling(period, min_periods=1).sum() / tr_sum
    vn = pd.Series(vm_minus).rolling(period, min_periods=1).sum() / tr_sum

    return vp.to_numpy(dtype=float), vn.to_numpy(dtype=float)


def cmf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    vol: np.ndarray,
    period: int = 20,
):
    """Return Chaikin Money Flow for OHLCV data."""

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    vol = np.asarray(vol, dtype=float)

    hl_diff = np.where(high - low == 0, 1e-9, high - low)
    mfm = ((close - low) - (high - close)) / hl_diff
    mfv = mfm * vol

    mfv_sum = pd.Series(mfv).rolling(period, min_periods=1).sum()
    vol_sum = pd.Series(vol).rolling(period, min_periods=1).sum().replace(0, 1e-9)

    return (mfv_sum / vol_sum).to_numpy(dtype=float)


def ichimoku(high: np.ndarray, low: np.ndarray):
    """Return Ichimoku components: Tenkan, Kijun, Span A and Span B."""

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    high_s = pd.Series(high)
    low_s = pd.Series(low)

    tenkan = (
        high_s.rolling(9, min_periods=1).max() + low_s.rolling(9, min_periods=1).min()
    ) / 2
    kijun = (
        high_s.rolling(26, min_periods=1).max() + low_s.rolling(26, min_periods=1).min()
    ) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (
        high_s.rolling(52, min_periods=1).max() + low_s.rolling(52, min_periods=1).min()
    ) / 2

    return (
        tenkan.to_numpy(dtype=float),
        kijun.to_numpy(dtype=float),
        span_a.to_numpy(dtype=float),
        span_b.to_numpy(dtype=float),
    )


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Return Average True Range with trailing windows."""

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )

    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy(dtype=float)


def ema(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Return exponential moving average for ``closes``."""

    closes = np.asarray(closes, dtype=float)
    return talib.EMA(closes, timeperiod=period)


def donchian(highs: np.ndarray, lows: np.ndarray, period: int = 20):
    """Return Donchian channel ``(upper, lower, middle)``."""

    highs_s = pd.Series(np.asarray(highs, dtype=float))
    lows_s = pd.Series(np.asarray(lows, dtype=float))
    upper = highs_s.rolling(period, min_periods=1).max().to_numpy()
    lower = lows_s.rolling(period, min_periods=1).min().to_numpy()
    middle = (upper + lower) / 2
    return upper, lower, middle


def kijun(highs: np.ndarray, lows: np.ndarray, period: int = 26) -> np.ndarray:
    """Return Ichimoku Kijun-sen line."""

    _tenkan, kijun_, _span_a, _span_b = ichimoku(highs, lows)
    return kijun_


def tenkan(highs: np.ndarray, lows: np.ndarray, period: int = 9) -> np.ndarray:
    """Return Ichimoku Tenkan-sen line."""

    tenkan_, _kijun, _span_a, _span_b = ichimoku(highs, lows)
    return tenkan_
