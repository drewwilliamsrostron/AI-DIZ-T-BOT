"""Background worker that writes contextual features to DuckDB every 15 min."""

from __future__ import annotations

import logging
import math
import time
from typing import List

import numpy as np
import requests

try:  # schedule may be missing on first run
    import schedule
except ModuleNotFoundError:  # pragma: no cover - installer path
    from artibot.auto_install import install as _auto

    _auto("schedule")
    import schedule
import yfinance as yf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch

import artibot.feature_store as fs

_LOG = logging.getLogger("feature_ingest")

_FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

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
    inputs = _FINBERT_TOKENIZER(
        headlines, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        logits = _FINBERT_MODEL(**inputs).logits
    probs = logits.softmax(dim=-1)
    # labels: 0=neutral 1=positive 2=negative
    score = float((probs[:, 1] - probs[:, 2]).mean().item())
    fs.upsert_news(_ts_hour(), score)


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
