"""Dataset utilities for the trading bot."""

# ruff: noqa: F403, F405
import os
import random
from typing import NamedTuple

import numpy as np
import pandas as pd
import talib
import sys
import importlib.machinery as _machinery

if "openai" in sys.modules and getattr(sys.modules["openai"], "__spec__", None) is None:
    sys.modules["openai"].__spec__ = _machinery.ModuleSpec("openai", None)

import torch

from .utils import rolling_zscore, feature_version_hash, validate_features
from torch.utils.data import Dataset

import artibot.globals as G
import logging
from .hyperparams import IndicatorHyperparams


###############################################################################
# NamedTuple for per-bar trade parameters
###############################################################################
class TradeParams(NamedTuple):
    risk_fraction: torch.Tensor
    sl_multiplier: torch.Tensor
    tp_multiplier: torch.Tensor
    attention: torch.Tensor


###############################################################################
# Dataclass for global indicator hyperparams
###############################################################################


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


def generate_fixed_features(
    data_np: np.ndarray,
    hp: IndicatorHyperparams,
    *,
    use_ichimoku: bool = False,
) -> np.ndarray:
    """Return a feature matrix with exactly 16 columns."""

    closes = data_np[:, 4].astype(np.float64)
    highs = data_np[:, 2].astype(np.float64)
    lows = data_np[:, 3].astype(np.float64)
    volume = data_np[:, 5].astype(np.float64)
    opens = data_np[:, 1].astype(np.float64)

    def zeros():
        return np.zeros_like(closes, dtype=np.float64)

    from .indicators import ema, atr, vortex, cmf, donchian, ichimoku

    sma = trailing_sma(closes, hp.sma_period) if hp.use_sma else zeros()
    rsi = talib.RSI(closes, timeperiod=hp.rsi_period) if hp.use_rsi else zeros()
    macd, _, _ = (
        talib.MACD(
            closes,
            fastperiod=hp.macd_fast,
            slowperiod=hp.macd_slow,
            signalperiod=hp.macd_signal,
        )
        if hp.use_macd
        else (zeros(), zeros(), zeros())
    )
    ema20 = ema(closes, period=hp.ema_period) if hp.use_ema else zeros()
    ema50 = ema(closes, period=50)
    atr_vals = atr(highs, lows, closes, period=hp.atr_period) if hp.use_atr else zeros()
    if hp.use_vortex:
        vp, vn = vortex(highs, lows, closes, period=hp.vortex_period)
    else:
        vp, vn = zeros(), zeros()
    cmf_v = (
        cmf(highs, lows, closes, volume, period=hp.cmf_period)
        if hp.use_cmf
        else zeros()
    )
    don_mid = (
        donchian(highs, lows, period=hp.donchian_period)[2]
        if hp.use_donchian
        else zeros()
    )
    ichi_tenkan = ichimoku(highs, lows)[0] if use_ichimoku else zeros()

    feats = np.column_stack(
        [
            opens,
            highs,
            lows,
            closes,
            volume,
            sma,
            rsi,
            macd,
            ema20,
            ema50,
            atr_vals,
            vp,
            vn,
            cmf_v,
            don_mid,
            ichi_tenkan,
        ]
    )

    if feats.shape[1] != 16:
        raise ValueError(f"Generated {feats.shape[1]} features, expected 16")

    return feats.astype(np.float32)


###############################################################################
# HourlyDataset
###############################################################################
class HourlyDataset(Dataset):
    def __init__(
        self,
        data,
        seq_len: int = 24,
        threshold: float = G.GLOBAL_THRESHOLD,
        *,
        indicator_hparams: IndicatorHyperparams = IndicatorHyperparams(),
        atr_threshold_k: float = 1.5,
        train_mode: bool = True,
        rebalance: bool = True,
        use_ichimoku: bool = False,
    ) -> None:
        self.data = data
        self.seq_len = seq_len
        self.threshold = threshold
        self.hp = indicator_hparams
        self.sma_period = indicator_hparams.sma_period
        self.atr_period = indicator_hparams.atr_period
        self.atr_threshold_k = atr_threshold_k
        self.train_mode = train_mode
        self.rebalance = rebalance
        self.use_vortex = indicator_hparams.use_vortex
        self.vortex_period = indicator_hparams.vortex_period
        self.use_cmf = indicator_hparams.use_cmf
        self.cmf_period = indicator_hparams.cmf_period
        self.use_ichimoku = use_ichimoku
        sample_np = np.array(self.data[: min(len(self.data), 100)], dtype=float)
        check_feats = generate_fixed_features(
            sample_np, indicator_hparams, use_ichimoku=use_ichimoku
        )
        if check_feats.shape[1] != 16:
            raise ValueError(
                f"Feature engineering produces {check_feats.shape[1]} features, expected 16."
            )
        self.samples, self.labels = self.preprocess()

    def preprocess(self):
        data_np = np.array(self.data, dtype=float)
        feats = generate_fixed_features(
            data_np, self.hp, use_ichimoku=self.use_ichimoku
        )
        print(f"[DEBUG] Actual feature count: {feats.shape[1]}")
        if feats.shape[1] != 16:
            raise ValueError("Feature dimension mismatch")
        validate_features(feats)
        self.feature_hash = feature_version_hash(feats)

        closes = data_np[:, 4].astype(np.float64)
        highs = data_np[:, 2].astype(np.float64)
        lows = data_np[:, 3].astype(np.float64)
        from .indicators import atr

        atr_vals = atr(highs, lows, closes, period=self.hp.atr_period)

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
        assert windows.shape[2] == 16
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
