"""Background worker that writes contextual features to DuckDB every 15 min."""

# ruff: noqa: E402

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np
import requests

from .environment import ensure_dependencies
from .auto_install import require
import importlib.machinery as _machinery
import sys

ensure_dependencies()
require("schedule")

if "openai" in sys.modules and getattr(sys.modules["openai"], "__spec__", None) is None:
    sys.modules["openai"].__spec__ = _machinery.ModuleSpec("openai", None)

import schedule

require("ccxt")
import ccxt
import pandas as pd
from .finbert_helper import score as finbert_score

import artibot.feature_store as fs

_LOG = logging.getLogger("feature_ingest")


def ingest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values and add a feature mask column."""

    valid_mask = ~df.isna()
    df = df.ffill(limit=7)
    df["feature_mask"] = valid_mask.all(axis=1).astype(int)
    return df


_MACRO_SRC = (
    "https://api.tradingeconomics.com/calendar/country/united states?c=guest:guest"
)


def _fetch_btc_usdt_hourly() -> pd.DataFrame:
    """Return recent BTC/USDT hourly bars from Phemex."""

    ex = ccxt.phemex()
    bars = ex.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=168)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    rvol = df["close"].pct_change().rolling(24).std() * np.sqrt(24)
    fs.upsert_rvol(int(df.index[-1].timestamp()), float(rvol.iloc[-1]))
    return df


def _ts_hour() -> int:
    """Return the current hour timestamp."""

    return int(time.time()) // 3600 * 3600


# ---- News sentiment ---------------------------------------------------------
def load_headlines() -> List[str]:
    """Return latest crypto headlines from GDELT."""

    try:
        from tools.backfill_gdelt import fetch_docs

        arts = fetch_docs("bitcoin OR crypto")
        return [a["title"] for a in arts]
    except Exception as exc:  # pragma: no cover - network
        _LOG.warning("GDELT DOC fetch failed: %s", exc)
        return []


def upd_sentiment() -> None:
    """Compute FinBERT sentiment score and store it."""

    headlines = load_headlines()
    if not headlines:
        return
    scores = []
    for h in headlines:
        lbl = finbert_score(h)
        scores.append({"positive": 1, "negative": -1}.get(lbl, 0))
    fs.upsert_news(_ts_hour(), float(np.mean(scores)))


# ---- Macro surprise ---------------------------------------------------------
def latest_cpi_surprise() -> float:
    """Return difference between actual and survey CPI."""

    try:
        rows = requests.get(_MACRO_SRC, timeout=10).json()[:1]
        try:
            actual = float(rows[0]["Actual"])
            survey = float(rows[0]["Consensus"])
        except KeyError:
            return 0.0
        return actual - survey
    except Exception as exc:  # pragma: no cover - network
        _LOG.warning("macro fetch failed: %s", exc)
        return 0.0


def upd_macro() -> None:
    """Store the latest macro surprise value."""

    fs.upsert_macro(_ts_hour(), latest_cpi_surprise())


# ---- Realised vol -----------------------------------------------------------
def upd_rvol() -> None:
    """Compute and store 7d realised volatility from BTC prices."""
    try:
        _fetch_btc_usdt_hourly()
    except Exception as exc:  # pragma: no cover - network
        _LOG.error("rvol fetch failed: %s", exc)


def start_worker() -> None:
    """Run scheduled updates in an infinite loop."""

    schedule.every(15).minutes.do(upd_sentiment)
    schedule.every(60).minutes.do(upd_macro)
    schedule.every(60).minutes.do(upd_rvol)
    _LOG.info("feature-ingest thread started")
    while True:  # pragma: no cover - infinite loop
        schedule.run_pending()
        time.sleep(10)
