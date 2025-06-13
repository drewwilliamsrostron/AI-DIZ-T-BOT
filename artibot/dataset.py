"""Dataset utilities for the trading bot."""

# ruff: noqa: F403, F405
import os
import random
from typing import NamedTuple

import numpy as np
import pandas as pd
import talib
import torch

from .utils import rolling_zscore
from torch.utils.data import Dataset

import artibot.globals as G
import logging


###############################################################################
# NamedTuple for per-bar trade parameters
###############################################################################
class TradeParams(NamedTuple):
    risk_fraction: torch.Tensor
    sl_multiplier: torch.Tensor
    tp_multiplier: torch.Tensor
    attention: torch.Tensor


###############################################################################
# NamedTuple for global indicator hyperparams
###############################################################################
class IndicatorHyperparams(NamedTuple):
    rsi_period: int
    sma_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int


###############################################################################
# CSV Loader and Dataset
###############################################################################
def load_csv_hourly(csv_path: str) -> list[list[float]]:
    """Return parsed hourly OHLCV data from ``csv_path``.

    The previous implementation iterated row by row with :func:`DataFrame.iterrows`,
    which is noticeably slow for large files.  This version leverages vectorised
    ``pandas`` operations so the entire CSV is processed in bulk.
    """

    if not os.path.isfile(csv_path):
        logging.warning(f"CSV file '{csv_path}' not found.")
        return []

    try:
        df = pd.read_csv(
            csv_path,
            sep=r"[,\t]+",
            engine="python",
            skiprows=1,
            header=0,
        )
    except Exception as e:  # pragma: no cover - IO failures
        logging.warning(f"Error reading CSV: {e}")
        return []

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"unix", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        logging.warning("CSV missing required columns: %s", required - set(df.columns))
        return []

    num = pd.to_numeric
    df["unix"] = num(df["unix"], errors="coerce").fillna(0).astype("int64")
    df.loc[df["unix"] > 1e12, "unix"] //= 1000
    df["open"] = num(df["open"], errors="coerce")
    df["high"] = num(df["high"], errors="coerce")
    df["low"] = num(df["low"], errors="coerce")
    df["close"] = num(df["close"], errors="coerce")

    # ----------------------------------------------------------------------------
    # Adaptive scaling: legacy datasets may store prices multiplied by 1e5 or 1e3
    # ----------------------------------------------------------------------------
    if not df.empty:
        ref_price = float(df["open"].iloc[0])
        scale = 1.0
        if ref_price > 1e8:
            scale = 1e5
        elif ref_price > 1e5:
            scale = 1e3
        if scale != 1.0:
            for col in ["open", "high", "low", "close"]:
                df[col] /= scale

    if "volume_btc" in df.columns:
        df["volume_btc"] = num(df["volume_btc"], errors="coerce").fillna(0.0)
    else:
        df["volume_btc"] = 0.0

    cols = ["unix", "open", "high", "low", "close", "volume_btc"]
    arr = df[cols].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]

    return arr[np.argsort(arr[:, 0])].tolist()


def trailing_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing (causal) SMA: value at t uses [t-window+1 … t].

    Pads the first ``window-1`` positions with ``np.nan`` so the returned array
    matches the input length.
    """

    if window < 1:
        raise ValueError("window must be ≥1")

    valid = np.convolve(arr, np.ones(window) / window, mode="valid")
    padded = np.concatenate([np.full(window - 1, np.nan, dtype=float), valid])
    return np.nan_to_num(padded)


###############################################################################
# HourlyDataset
###############################################################################
class HourlyDataset(Dataset):
    def __init__(
        self,
        data,
        seq_len=24,
        threshold=G.GLOBAL_THRESHOLD,
        sma_period=10,
        atr_period: int = 50,
        atr_threshold_k: float = 1.5,
        train_mode=True,
        rebalance=True,
        *,
        use_vortex: bool = False,
        use_cmf: bool = False,
        use_ichimoku: bool = False,
        use_atr: bool = False,
    ):
        self.data = data
        self.seq_len = seq_len
        self.threshold = threshold
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_threshold_k = atr_threshold_k
        self.train_mode = train_mode
        self.rebalance = rebalance
        self.use_vortex = use_vortex
        self.use_cmf = use_cmf
        self.use_ichimoku = use_ichimoku
        self.use_atr = use_atr
        self.samples, self.labels = self.preprocess()

    def preprocess(self):
        data_np = np.array(self.data, dtype=np.float32)
        closes = data_np[:, 4].astype(np.float64)
        sma = trailing_sma(closes, self.sma_period)
        try:
            rsi = talib.RSI(closes, timeperiod=14)
        except Exception:
            rsi = None
        if rsi is None:
            rsi = np.zeros_like(closes, dtype=np.float64)
        try:
            macd, _, _ = talib.MACD(closes)
        except Exception:
            macd = None
        if macd is None:
            macd = np.zeros_like(closes, dtype=np.float64)

        highs = data_np[:, 2].astype(np.float64)
        lows = data_np[:, 3].astype(np.float64)
        volume = data_np[:, 5].astype(np.float64)

        from .indicators import atr

        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        cols = [
            data_np[:, 1:6],
            sma.astype(np.float32),
            rsi.astype(np.float32),
            macd.astype(np.float32),
        ]

        if self.use_atr:
            cols.append(atr_vals.astype(np.float32))

        if self.use_vortex:
            from .indicators import vortex

            vp, vn = vortex(highs, lows, closes)
            cols.extend([vp.astype(np.float32), vn.astype(np.float32)])

        if self.use_cmf:
            from .indicators import cmf

            cmf_v = cmf(highs, lows, closes, volume)
            cols.append(cmf_v.astype(np.float32))

        if self.use_ichimoku:
            from .indicators import ichimoku

            tenkan, kijun, span_a, span_b = ichimoku(highs, lows)
            cols.extend(
                [
                    tenkan.astype(np.float32),
                    kijun.astype(np.float32),
                    span_a.astype(np.float32),
                    span_b.astype(np.float32),
                ]
            )

        feats = np.column_stack(cols)

        # ``ta-lib`` leaves the first few rows as NaN which would otherwise
        # propagate through scaling and ultimately make the training loss
        # explode to ``nan``.  Replace them with zeros before normalisation and
        # again afterwards to be safe.
        feats = np.nan_to_num(feats)

        scaled_feats = rolling_zscore(feats, window=50)

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(
            scaled_feats, (self.seq_len, scaled_feats.shape[1])
        )[:, 0]
        windows = windows[:-1]

        raw_closes = closes.astype(np.float32)
        last_close_raw = raw_closes[self.seq_len - 1 : -1]
        next_close_raw = raw_closes[self.seq_len :]
        rets = (next_close_raw - last_close_raw) / (last_close_raw + 1e-8)
        thresholds = self.atr_threshold_k * atr_vals[self.seq_len - 1 : -1]
        labels = np.where(rets > thresholds, 0, np.where(rets < -thresholds, 1, 2))

        mask = (
            np.isfinite(windows).all(axis=(1, 2))
            & np.isfinite(rets)
            & np.isfinite(thresholds)
        )
        windows = windows[mask]
        labels = labels[mask]

        windows = np.nan_to_num(windows)

        return windows.astype(np.float32), labels.astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        if self.train_mode and random.random() < 0.2:
            sample += np.random.normal(0, 0.01, sample.shape)
        # Explicit dtype avoids "Could not infer dtype" errors on some
        # platforms when NumPy 2.x is installed.
        sample_t = torch.as_tensor(sample, dtype=torch.float32)
        label = self.labels[idx]
        if self.rebalance and label == 2 and random.random() < 0.5:
            label = random.choice([0, 1])
        label_t = torch.tensor(label, dtype=torch.long)
        return sample_t, label_t
