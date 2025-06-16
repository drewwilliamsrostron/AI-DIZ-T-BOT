"""Background worker that writes contextual features to DuckDB every 15 min."""

# ruff: noqa: E402

from __future__ import annotations

import logging
import math
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

require("yfinance")
import yfinance as yf
from .finbert_helper import score as finbert_score

import artibot.feature_store as fs

_LOG = logging.getLogger("feature_ingest")


_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/docsearch"
_MACRO_SRC = (
    "https://api.tradingeconomics.com/calendar/country/united states?c=guest:guest"
)


def _ts_hour() -> int:
    """Return the current hour timestamp."""

    return int(time.time()) // 3600 * 3600


# ---- News sentiment ---------------------------------------------------------
def load_headlines() -> List[str]:
    """Return latest crypto headlines from GDELT."""

    try:
        js = requests.get(
            _DOC_URL,
            params={"query": "bitcoin OR crypto", "maxrecords": 250, "format": "json"},
            timeout=8,
        ).json()
        return [a["title"] for a in js.get("articles", [])]
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
        actual = float(rows[0]["Actual"])
        survey = float(rows[0]["Consensus"])
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
        btc = yf.download("BTC-USD", period="8d", interval="1h", progress=False)[
            "Close"
        ]
        logret = np.log(btc).diff().dropna()
        if len(logret) < 168:
            return
        rvol = logret.rolling(168).std(ddof=0).iloc[-1] * math.sqrt(8760)
        fs.upsert_rvol(_ts_hour(), float(rvol))
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
