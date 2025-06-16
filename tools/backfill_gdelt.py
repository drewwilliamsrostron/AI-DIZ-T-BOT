#!/usr/bin/env python
"""Back-fill DuckDB feature store using GDELT archives and TradingEconomics."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import logging
import math
import time
from datetime import timedelta
from typing import Iterator

import numpy as np
import pandas as pd
import requests
import os
import yfinance as yf
from tqdm import tqdm

import artibot.feature_store as fs


LOG = logging.getLogger("backfill_gdelt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

START = dt.datetime(2015, 1, 1)
END = dt.datetime.utcnow()
GDELT_ARCH = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
TE_CAL = "https://api.tradingeconomics.com/calendar"

NO_HEAVY = os.environ.get("NO_HEAVY") == "1"


def _download_btc_hourly(progress_cb=None) -> pd.DataFrame:
    """Return hourly BTC prices via yfinance in chunks."""

    if NO_HEAVY:
        return pd.DataFrame()

    step = timedelta(days=700)
    frames = []
    total = int((END - START) / step) + 1
    s = START
    idx = 0
    last = 0.0
    while s < END:
        idx += 1
        e = min(s + step, END)
        LOG.info("YF %s→%s", s.date(), e.date())
        try:
            df = yf.download(
                "BTC-USD",
                start=s,
                end=e,
                interval="1h",
                progress=False,
            )
            frames.append(df)
        except Exception as exc:  # pragma: no cover - network
            LOG.warning("YF chunk failed: %s", exc)
        s = e
        pct = idx / total * 100.0
        if progress_cb and pct - last >= 5:
            progress_cb(pct, f"YF slice {idx}/{total} downloaded")
            last = pct
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def _fetch_gdelt(query: str, progress_cb=None) -> list[dict]:
    """Return paged DOC results with retry and an overall 60s timeout."""

    if NO_HEAVY:
        return []

    base = "https://api.gdeltproject.org/api/v2/doc/docsearch"
    out: list[dict] = []
    page = 1
    last = time.time()
    start = time.time()
    while True:
        if time.time() - start > 60:
            raise requests.RequestException("GDELT timeout")
        params = {
            "query": query,
            "maxrecords": 250,
            "format": "json",
            "page": page,
        }
        for attempt in range(5):
            try:
                r = requests.get(base, params=params, timeout=(3, 8))
                r.raise_for_status()
                break
            except requests.RequestException as exc:  # pragma: no cover - network
                delay = 2**attempt
                LOG.debug("GDELT page %s failed: %s", page, exc)
                if time.time() - start + delay >= 60:
                    raise
                time.sleep(delay)
        else:
            break

        arts = r.json().get("articles", [])
        if not arts:
            break
        out.extend(arts)
        page += 1
        if progress_cb and time.time() - last > 5:
            progress_cb(page * 100 / 50.0, f"GDELT page {page}")
            last = time.time()
        time.sleep(1.2)
    return out


def hour_ts(d: dt.datetime) -> int:
    return int(d.replace(minute=0, second=0, microsecond=0).timestamp())


###############################################################################
# 1. Realised volatility (BTC)
###############################################################################
def realised_vol_series(progress_cb=None) -> pd.Series:
    btc_df = _download_btc_hourly(progress_cb)
    btc = btc_df["Close"]
    logret = np.log(btc).diff()
    rvol = logret.rolling(168).std(ddof=0) * math.sqrt(8760)
    rvol.name = "rvol"
    return rvol


###############################################################################
# 2. Macro surprise (CPI, NFP)
###############################################################################
def fetch_macro() -> pd.DataFrame:
    if NO_HEAVY:
        return pd.DataFrame(columns=["ts", "surprise_z"])
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

    if NO_HEAVY:
        return iter([])

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
    if NO_HEAVY:
        return pd.Series(dtype=float)
    try:
        _fetch_gdelt("bitcoin")
    except requests.RequestException:
        LOG.warning("[WARN] GDELT unavailable – continuing without news sentiment")
        return pd.Series(dtype=float)

    rows: dict[int, list[float]] = {}
    for ts, base_tone in tqdm(iter_archive_hours(), desc="GDELT sentiment"):
        hts = hour_ts(ts)
        rows.setdefault(hts, []).append(base_tone)
    return pd.Series({k: np.mean(v) for k, v in rows.items()})


###############################################################################
# 4. Merge & write
###############################################################################
def main(progress_cb=lambda pct, msg: None) -> None:
    if NO_HEAVY:
        progress_cb(100.0, "Done")
        return
    LOG.info("Fetching realised volatility")
    rvol = realised_vol_series(progress_cb)

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
