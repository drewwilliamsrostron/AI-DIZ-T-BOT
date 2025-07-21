"""Dataset utilities for the trading bot."""

# ruff: noqa: F403, F405
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import NamedTuple

import importlib.machinery as _machinery
import tomllib

import numpy as np
import pandas as pd
import talib
import torch
from sklearn.impute import KNNImputer
from torch.utils.data import Dataset

import artibot.globals as G
from artibot.rules.risk_filter import apply as risk_filter
from config import FEATURE_COLUMNS

from .constants import FEATURE_DIMENSION
from .hyperparams import IndicatorHyperparams
from .utils import (
    clean_features,
    enforce_feature_dim,
    feature_mask_for,
    feature_version_hash,
    rolling_zscore,
    validate_feature_dimension,
    validate_features,
    zero_disabled,
)

# --------------------------------------------------------------------------- #
# openai stub-spec guard (avoids import side-effects in some IDEs)
# --------------------------------------------------------------------------- #
if "openai" in sys.modules and getattr(sys.modules["openai"], "__spec__", None) is None:
    sys.modules["openai"].__spec__ = _machinery.ModuleSpec("openai", None)

# --------------------------------------------------------------------------- #
# caching objects
# --------------------------------------------------------------------------- #
try:
    _IMPUTER = KNNImputer(n_neighbors=5) if callable(KNNImputer) else None
except Exception:
    _IMPUTER = None


# ---------------- dataset range constants ---------------- #
# These are populated when ``load_csv_hourly`` successfully loads a file.
DATA_START: int | None = None
DATA_END: int | None = None


# --------------------------------------------------------------------------- #
# NamedTuple – per-bar trade parameters
# --------------------------------------------------------------------------- #
class TradeParams(NamedTuple):
    risk_fraction: torch.Tensor
    sl_multiplier: torch.Tensor
    tp_multiplier: torch.Tensor
    attention: torch.Tensor


# --------------------------------------------------------------------------- #
# CSV loader
# --------------------------------------------------------------------------- #
def _risk_filter_enabled_from_toml(path: str = "config/default.toml") -> bool:
    """Fallback to the default config file (True if missing/error)."""
    try:
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
            return bool(cfg.get("risk_filter", {}).get("enabled", True))
    except Exception:
        return True


def _determine_risk_filter_flag(cfg: dict | None) -> bool:
    """
    Priority (high → low):

    1. `cfg["risk_filter"]["enabled"]` given directly to `load_csv_hourly`
    2. Runtime global `G.is_risk_filter_enabled()` (returns None if unset)
    3. Value in *config/default.toml*
    4. Hard default `True`
    """
    if cfg and "risk_filter" in cfg:
        return bool(cfg["risk_filter"].get("enabled", True))

    runtime_flag = getattr(G, "is_risk_filter_enabled", lambda: None)()
    if runtime_flag is not None:
        return bool(runtime_flag)

    return _risk_filter_enabled_from_toml()


def load_csv_hourly(csv_path: str, *, cfg: dict | None = None) -> list[list[float]]:
    """Fast vectorised CSV→list loader for hourly OHLCV bars."""

    log = logging.getLogger("dataset.loader")

    if not Path(csv_path).is_file():
        log.warning("CSV file '%s' not found.", csv_path)
        return []

    try:
        df = pd.read_csv(
            csv_path,
            sep=r"[,\t]+",  # accept comma OR tab
            engine="python",
            skiprows=1,
            header=0,
        )
    except Exception as exc:  # pragma: no cover – IO failures
        log.warning("Error reading CSV: %s", exc)
        return []

    # normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    must_have = {"unix", "open", "high", "low", "close"}
    if not must_have.issubset(df.columns):
        log.warning("CSV missing required columns: %s", must_have - set(df.columns))
        return []

    num = pd.to_numeric
    df["unix"] = num(df["unix"], errors="coerce").fillna(0).astype("int64")
    df.loc[df["unix"] > 1e12, "unix"] //= 1000  # ms → s

    for col in ("open", "high", "low", "close"):
        df[col] = num(df[col], errors="coerce")

    # legacy price scaling
    if not df.empty:
        ref = float(df["open"].iloc[0])
        factor = 1e5 if ref > 1e8 else (1e3 if ref > 1e5 else 1.0)
        if factor != 1.0:
            df[["open", "high", "low", "close"]] /= factor

    df["volume_btc"] = num(df.get("volume_btc", 0.0), errors="coerce").fillna(0.0)

    # ---------------- risk filter ---------------- #
    enabled = _determine_risk_filter_flag(cfg)
    df = risk_filter(df, enabled=enabled)

    # ---------------- ndarray cleanup ------------ #
    cols = ["unix", "open", "high", "low", "close", "volume_btc"]
    arr = df[cols].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = arr[np.argsort(arr[:, 0])]
    if arr.size:
        global DATA_START, DATA_END
        DATA_START = int(arr[0, 0])
        DATA_END = int(arr[-1, 0])
    return arr.tolist()


# --------------------------------------------------------------------------- #
# Technical-indicator feature engineering
# --------------------------------------------------------------------------- #
def trailing_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal SMA (pads first *window-1* values with NaN)."""
    if window < 1:
        raise ValueError("window must be ≥1")

    valid = np.convolve(arr, np.ones(window) / window, mode="valid")
    padded = np.concatenate([np.full(window - 1, np.nan, dtype=float), valid])
    return np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)


def generate_fixed_features(
    data_np: np.ndarray,
    hp: IndicatorHyperparams,
    *,
    use_ichimoku: bool = False,
) -> np.ndarray:
    """Return a raw feature matrix with **exactly** `FEATURE_DIMENSION` columns."""

    closes = data_np[:, 4].astype(np.float64)
    highs = data_np[:, 2].astype(np.float64)
    lows = data_np[:, 3].astype(np.float64)
    volume = data_np[:, 5].astype(np.float64)
    opens = data_np[:, 1].astype(np.float64)

    from .indicators import ema, atr, vortex, cmf, donchian, ichimoku

    # Compute every indicator upfront so toggling on new features can immediately
    # use historical values.  Feature masks will zero-out anything disabled.

    sma = trailing_sma(closes, hp.sma_period)
    rsi = talib.RSI(closes, timeperiod=hp.rsi_period)
    macd, _, _ = talib.MACD(
        closes,
        fastperiod=hp.macd_fast,
        slowperiod=hp.macd_slow,
        signalperiod=hp.macd_signal,
    )
    ema20 = ema(closes, period=hp.ema_period)
    ema50 = ema(closes, period=50)
    atr_vals = atr(highs, lows, closes, period=hp.atr_period)
    vp, vn = vortex(highs, lows, closes, period=hp.vortex_period)
    cmf_v = cmf(highs, lows, closes, volume, period=hp.cmf_period)
    don_mid = donchian(highs, lows, period=hp.donchian_period)[2]
    ichi_tenkan = ichimoku(highs, lows)[0]

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

    # Sanitise invalid values before validation
    if np.isnan(feats).any() or np.isinf(feats).any():
        logging.debug("Raw features contained NaN/Inf – cleaned")

    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_features(
    data_np: np.ndarray,
    hp: IndicatorHyperparams,
    *,
    use_ichimoku: bool = False,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Return normalised feature matrix for *data_np*."""
    feats = generate_fixed_features(data_np, hp, use_ichimoku=use_ichimoku)
    feats = clean_features(feats, replace_value=0.0)
    feats = enforce_feature_dim(feats, FEATURE_DIMENSION)
    feats = validate_feature_dimension(
        feats, FEATURE_DIMENSION, logger or logging.getLogger("build_features")
    )

    mask = feature_mask_for(hp, use_ichimoku=use_ichimoku)
    validate_features(feats, enabled_mask=mask)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    if _IMPUTER is not None:
        feats[:, mask] = _IMPUTER.fit_transform(feats[:, mask])
    feats = zero_disabled(feats, mask)
    feats = rolling_zscore(feats, window=50, mask=mask)
    feats = zero_disabled(feats, mask)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess_features(
    data_np: np.ndarray,
    hp: IndicatorHyperparams,
    seq_len: int,
    *,
    use_ichimoku: bool = False,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Return feature matrix for *data_np* without creating windows eagerly."""

    log = logger or logging.getLogger(__name__)

    features = build_features(
        data_np,
        hp,
        use_ichimoku=use_ichimoku,
        logger=log,
    )

    return features.astype(np.float32)


# --------------------------------------------------------------------------- #
# PyTorch dataset wrapper
# --------------------------------------------------------------------------- #
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
        self.logger = logging.getLogger("dataset")

        self.data = data
        self.seq_len = seq_len
        self.threshold = threshold
        self.hp = indicator_hparams
        self.atr_threshold_k = atr_threshold_k
        self.train_mode = train_mode
        self.rebalance = rebalance
        self.use_ichimoku = use_ichimoku

        self.expected_features = FEATURE_DIMENSION
        self.logger.debug("[INIT] Expected features type: %s", type(FEATURE_DIMENSION))

        # sanity-check with a small sample
        sample_np = np.array(self.data[: min(len(self.data), 100)], dtype=float)
        check_feats = build_features(
            sample_np, indicator_hparams, use_ichimoku=use_ichimoku, logger=self.logger
        )
        if check_feats.shape[1] != self.expected_features:
            raise ValueError(
                f"Feature engineering produces {check_feats.shape[1]} features, "
                f"expected {self.expected_features}."
            )

        self.mask = feature_mask_for(indicator_hparams, use_ichimoku=use_ichimoku)
        self.features, self.labels = self.preprocess()

    # --------------------------------------------------------------------- #
    def apply_feature_mask(
        self, hp: IndicatorHyperparams, *, use_ichimoku: bool | None = None
    ) -> None:
        """Zero-out features based on *hp* without rebuilding the dataset."""
        if use_ichimoku is not None:
            self.use_ichimoku = use_ichimoku
        self.mask = feature_mask_for(hp, use_ichimoku=self.use_ichimoku)
        self.features = zero_disabled(self.features, self.mask)

    # --------------------------------------------------------------------- #
    def preprocess(self):
        data_np = np.array(self.data, dtype=float)

        features = preprocess_features(
            data_np,
            self.hp,
            self.seq_len,
            use_ichimoku=self.use_ichimoku,
            logger=self.logger,
        )
        if getattr(self.hp, "debug", False):
            enabled = [n for f, n in zip(self.mask, FEATURE_COLUMNS) if f]
            self.logger.info("[TRACE] mask enabled=%s (len=%d)", enabled, len(enabled))

        self.feature_hash = feature_version_hash(features)

        # label generation
        closes = data_np[:, 4].astype(np.float64)
        highs = data_np[:, 2].astype(np.float64)
        lows = data_np[:, 3].astype(np.float64)
        from .indicators import atr

        atr_vals = atr(highs, lows, closes, period=self.hp.atr_period)

        last_close = closes[self.seq_len - 1 : -1]
        next_close = closes[self.seq_len :]
        rets = (next_close - last_close) / (last_close + 1e-8)
        thresholds = self.atr_threshold_k * atr_vals[self.seq_len - 1 : -1]
        labels = np.where(rets > thresholds, 0, np.where(rets < -thresholds, 1, 2))

        self.features = zero_disabled(
            np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), self.mask
        )

        labels = labels.astype(np.int64)
        return self.features, labels

    # --------------------------------------------------------------------- #
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        sample = self.features[start:end].copy()
        if sample.shape[-1] != self.expected_features:
            self.logger.error("Feature dimension mismatch: %s", sample.shape)
            raise ValueError(
                f"Expected {self.expected_features} features, got {sample.shape[-1]}"
            )

        if self.train_mode and random.random() < 0.2:
            sample += np.random.normal(0, 0.01, sample.shape)

        sample_t = torch.as_tensor(sample, dtype=torch.float32, device="cpu")
        label = self.labels[idx]
        if self.rebalance and label == 2 and random.random() < 0.5:
            label = random.choice([0, 1])
        label_t = torch.tensor(label, dtype=torch.long, device="cpu")
        return sample_t, label_t

    # convenience for external callers
    def get_feature_dimension(self):
        return self.expected_features
