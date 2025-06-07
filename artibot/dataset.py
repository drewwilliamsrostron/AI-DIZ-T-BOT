"""Dataset utilities for the trading bot."""

# ruff: noqa: F403, F405
from typing import NamedTuple

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import talib
from .globals import *


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
    df["open"] = num(df["open"], errors="coerce") / 100000.0
    df["high"] = num(df["high"], errors="coerce") / 100000.0
    df["low"] = num(df["low"], errors="coerce") / 100000.0
    df["close"] = num(df["close"], errors="coerce") / 100000.0

    if "volume_btc" in df.columns:
        df["volume_btc"] = num(df["volume_btc"], errors="coerce").fillna(0.0)
    else:
        df["volume_btc"] = 0.0

    cols = ["unix", "open", "high", "low", "close", "volume_btc"]
    arr = df[cols].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]

    return arr[np.argsort(arr[:, 0])].tolist()


###############################################################################
# HourlyDataset
###############################################################################
class HourlyDataset(Dataset):
    def __init__(
        self,
        data,
        seq_len=24,
        threshold=GLOBAL_THRESHOLD,
        sma_period=10,
        train_mode=True,
    ):
        self.data = data
        self.seq_len = seq_len
        self.threshold = threshold
        self.sma_period = sma_period
        self.train_mode = train_mode
        self.samples, self.labels = self.preprocess()

    def preprocess(self):
        data_np = np.array(self.data, dtype=np.float32)
        closes = data_np[:, 4].astype(np.float64)
        sma = np.convolve(
            closes, np.ones(self.sma_period) / self.sma_period, mode="same"
        )
        rsi = talib.RSI(closes, timeperiod=14)
        macd, _, _ = talib.MACD(closes)

        feats = np.column_stack(
            [
                data_np[:, 1:6],
                sma.astype(np.float32),
                rsi.astype(np.float32),
                macd.astype(np.float32),
            ]
        )

        # ``ta-lib`` leaves the first few rows as NaN which would otherwise
        # propagate through scaling and ultimately make the training loss
        # explode to ``nan``.  Replace them with zeros before normalisation and
        # again afterwards to be safe.
        feats = np.nan_to_num(feats)

        scaler = StandardScaler()
        scaled_feats = scaler.fit_transform(feats)

        scaled_feats = np.clip(scaled_feats, -10.0, 10.0)

        scaled_feats = np.nan_to_num(scaled_feats)

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(
            scaled_feats, (self.seq_len, scaled_feats.shape[1])
        )[:, 0]
        windows = windows[:-1]

        raw_closes = closes.astype(np.float32)
        last_close_raw = raw_closes[self.seq_len - 1 : -1]
        next_close_raw = raw_closes[self.seq_len :]
        rets = (next_close_raw - last_close_raw) / (last_close_raw + 1e-8)
        labels = np.where(
            rets > self.threshold, 0, np.where(rets < -self.threshold, 1, 2)
        )

        mask = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(rets)
        windows = windows[mask]
        labels = labels[mask]

        windows = np.nan_to_num(windows)

        return windows.astype(np.float32), labels.astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        # (8) Data Augmentation: bigger probability + bigger noise
        # from 0.2 => 0.5 probability, and 0.01 => 0.02 stdev
        if self.train_mode and random.random() < 0.5:
            sample += np.random.normal(0, 0.02, sample.shape)
        # Explicit dtype avoids "Could not infer dtype" errors on some
        # platforms when NumPy 2.x is installed.
        sample_t = torch.as_tensor(sample, dtype=torch.float32)
        label_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample_t, label_t
