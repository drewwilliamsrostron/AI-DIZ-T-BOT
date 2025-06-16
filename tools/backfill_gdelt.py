#!/usr/bin/env python
"""Back-fill DuckDB feature store using GDELT archives and TradingEconomics."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import logging
import math
from typing import Iterator

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from artibot.finbert_helper import score as finbert_score
from tqdm import tqdm

import artibot.feature_store as fs


LOG = logging.getLogger("backfill_gdelt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

START = dt.datetime(2015, 2, 1)
END = dt.datetime.utcnow()
GDELT_ARCH = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
TE_CAL = "https://api.tradingeconomics.com/calendar"


def hour_ts(d: dt.datetime) -> int:
    return int(d.replace(minute=0, second=0, microsecond=0).timestamp())


###############################################################################
# 1. Realised volatility (BTC)
###############################################################################
def realised_vol_series() -> pd.Series:
    btc = yf.download("BTC-USD", start=START, end=END, interval="1h", progress=False)[
        "Close"
    ]
    logret = np.log(btc).diff()
    rvol = logret.rolling(168).std(ddof=0) * math.sqrt(8760)
    rvol.name = "rvol"
    return rvol


###############################################################################
# 2. Macro surprise (CPI, NFP)
###############################################################################
def fetch_macro() -> pd.DataFrame:
    params = {
        "c": "guest:guest",
        "d1": START.strftime("%Y-%m-%d"),
        "d2": END.strftime("%Y-%m-%d"),
        "s": "cpi:us, nfp:us",
    }
    rel = requests.get(TE_CAL, params=params, timeout=20).json()
    rows = []
    for r in rel:
        if r.get("Actual") and r.get("Consensus"):
            ts = hour_ts(dt.datetime.fromisoformat(r["Date"]))
            rows.append((ts, float(r["Actual"]) - float(r["Consensus"])))
    df = pd.DataFrame(rows, columns=["ts", "surprise"])
    df["surprise_z"] = (df["surprise"] - df["surprise"].expanding().mean()) / df[
        "surprise"
    ].expanding().std(ddof=0)
    return df[["ts", "surprise_z"]]


###############################################################################
# 3. Sentiment from GDELT archives
###############################################################################
def iter_archive_hours() -> Iterator[tuple[dt.datetime, float]]:
    """Yield (ts_hour, tone) pairs from hourly CSV archives."""

    cur = START
    while cur <= END:
        ts_id = cur.strftime("%Y%m%d%H%M%S")
        url = GDELT_ARCH.format(ts=ts_id)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                cur += dt.timedelta(hours=1)
                continue
            buf = io.BytesIO(resp.content)
        except Exception:
            cur += dt.timedelta(hours=1)
            continue
        with gzip.GzipFile(fileobj=buf) as gf:
            rdr = csv.reader(io.TextIOWrapper(gf, errors="ignore", newline=""))
            for row in rdr:
                themes, tone = row[3], row[9]
                if "CRYPTO_CURRENCY" in themes or "BITCOIN" in themes:
                    yield cur, float(tone.split(",")[0])
        cur += dt.timedelta(hours=1)


def sentiment_series() -> pd.Series:
    rows: dict[int, list[float]] = {}
    for ts, base_tone in tqdm(iter_archive_hours(), desc="GDELT sentiment"):
        hts = hour_ts(ts)
        rows.setdefault(hts, []).append(base_tone)
    return pd.Series({k: np.mean(v) for k, v in rows.items()})


###############################################################################
# 4. Merge & write
###############################################################################
def main(progress_cb=lambda pct, msg: None) -> None:
    LOG.info("Fetching realised volatility")
    rvol = realised_vol_series()

    LOG.info("Fetching macro surprises")
    macro = fetch_macro().set_index("ts")["surprise_z"]

    LOG.info("Fetching GDELT sentiment")
    sent = sentiment_series()

    idx = pd.Index(
        sorted(set(rvol.index) | set(macro.index) | set(sent.index)), name="ts"
    )
    df = pd.DataFrame(index=idx)
    df["sent"] = sent.reindex(idx).fillna(method="ffill").fillna(0)
    df["macro"] = macro.reindex(idx).fillna(method="ffill").fillna(0)
    df["rvol"] = (
        rvol.reindex(idx).fillna(method="ffill").fillna(method="bfill").fillna(0)
    )

    LOG.info("Writing to DuckDB (%d rows)", len(df))
    batch: list[tuple[int, float, float, float]] = []
    total = len(df)
    for i, (ts, row) in enumerate(df.iterrows(), 1):
        batch.append((ts, row.sent, row.macro, row.rvol))
        if i % 1000 == 0:
            progress_cb(i / total * 100.0, f"Ingested {i}")
        if len(batch) >= 5000:
            _flush(batch)
            batch.clear()
    if batch:
        _flush(batch)
    progress_cb(100.0, "Done")


def _flush(batch: list[tuple[int, float, float, float]]) -> None:
    for ts, sent, macro, rvol in batch:
        fs.upsert_news(ts, float(sent))
        fs.upsert_macro(ts, float(macro))
        fs.upsert_rvol(ts, float(rvol))


if __name__ == "__main__":
    main()
