import numpy as np
from artibot.indicators import vortex, cmf, ichimoku


def manual_vortex(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int):
    prev_close = np.concatenate(([np.nan], close[:-1]))
    prev_low = np.concatenate(([np.nan], low[:-1]))
    prev_high = np.concatenate(([np.nan], high[:-1]))
    tr = np.maximum(
        high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    vm_plus = np.abs(high - prev_low)
    vm_minus = np.abs(low - prev_high)
    vp = np.empty_like(high, dtype=float)
    vn = np.empty_like(high, dtype=float)
    for t in range(len(high)):
        start = max(0, t - period + 1)
        tr_win = tr[start : t + 1]
        vp_win = vm_plus[start : t + 1]
        vn_win = vm_minus[start : t + 1]
        if np.all(np.isnan(tr_win)):
            vp[t] = np.nan
            vn[t] = np.nan
        else:
            tr_sum = np.nansum(tr_win)
            vp_sum = np.nansum(vp_win)
            vn_sum = np.nansum(vn_win)
            vp[t] = vp_sum / tr_sum
            vn[t] = vn_sum / tr_sum
    return vp, vn


def manual_cmf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, vol: np.ndarray, period: int
):
    """Manual Chaikin Money Flow computed with a causal window.

    Uses 1e-9 guards to avoid divide‑by‑zero when high == low or volume sum == 0.
    """
    res = np.empty_like(high, dtype=float)
    for t in range(len(high)):
        start = max(0, t - period + 1)
        h_slice = high[start : t + 1]
        l_slice = low[start : t + 1]
        c_slice = close[start : t + 1]
        v_slice = vol[start : t + 1]

        # Money Flow Multiplier (MFM)
        hl_diff = np.where(h_slice - l_slice == 0, 1e-9, h_slice - l_slice)
        mfm = ((c_slice - l_slice) - (h_slice - c_slice)) / hl_diff

        # Money Flow Volume (MFV)
        mfv_sum = np.dot(mfm, v_slice)
        vol_sum = v_slice.sum() if v_slice.sum() != 0 else 1e-9
        res[t] = mfv_sum / vol_sum
    return res


def manual_ichimoku(high: np.ndarray, low: np.ndarray):
    n = len(high)
    tenkan = np.empty(n, dtype=float)
    kijun = np.empty(n, dtype=float)
    span_a = np.empty(n, dtype=float)
    span_b = np.empty(n, dtype=float)
    for t in range(n):
        tenkan[t] = (
            high[max(0, t - 8) : t + 1].max() + low[max(0, t - 8) : t + 1].min()
        ) / 2
        kijun[t] = (
            high[max(0, t - 25) : t + 1].max() + low[max(0, t - 25) : t + 1].min()
        ) / 2
        span_a[t] = (tenkan[t] + kijun[t]) / 2
        span_b[t] = (
            high[max(0, t - 51) : t + 1].max() + low[max(0, t - 51) : t + 1].min()
        ) / 2
    return tenkan, kijun, span_a, span_b


# -------------------------
#  Unit‑tests (causality)
# -------------------------

def test_vortex_is_causal():
    high = np.linspace(10, 19, 10)
    low = high - 1
    close = high - 0.5
    vp, vn = vortex(high, low, close, period=3)
    vp_manual, vn_manual = manual_vortex(high, low, close, 3)
    assert np.allclose(vp, vp_manual, equal_nan=True)
    assert np.allclose(vn, vn_manual, equal_nan=True)


def test_cmf_is_causal():
    high = np.linspace(10, 19, 10)
    low = high - 1
    close = high - 0.5
    vol = np.arange(1, 11)
    cmf_val = cmf(high, low, close, vol, period=4)
    cmf_manual = manual_cmf(high, low, close, vol, 4)
    assert np.allclose(cmf_val, cmf_manual, equal_nan=True)


def test_ichimoku_is_causal():
    high = np.linspace(10, 19, 10)
    low = high - 1
    tenkan, kijun, span_a, span_b = ichimoku(high, low)
    t_m, k_m, sa_m, sb_m = manual_ichimoku(high, low)
    assert np.allclose(tenkan, t_m, equal_nan=True)
    assert np.allclose(kijun, k_m, equal_nan=True)
    assert np.allclose(span_a, sa_m, equal_nan=True)
    assert np.allclose(span_b, sb_m, equal_nan=True)
