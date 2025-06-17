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
from typing import Iterator

import numpy as np
import pandas as pd
import requests
import os
import ccxt
import importlib.machinery as _machinery
import sys

if "tqdm" in sys.modules and getattr(sys.modules["tqdm"], "__spec__", None) is None:
    sys.modules["tqdm"].__spec__ = _machinery.ModuleSpec("tqdm", None)

from tqdm import tqdm

import artibot.feature_store as fs


LOG = logging.getLogger("backfill_gdelt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

START = dt.datetime(2015, 1, 1)
END = dt.datetime.utcnow()
GDELT_ARCH = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
TE_CAL = "https://api.tradingeconomics.com/calendar"

NO_HEAVY = os.environ.get("NO_HEAVY") == "1"


def _fetch_phemex_bars(exchange, since: int, limit: int):
    """Return raw OHLCV bars or ``None`` on fatal error."""

    try:
        return exchange.fetch_ohlcv(
            "BTCUSD",
            timeframe="1h",
            since=since,
            limit=limit,
        )
    except ccxt.RateLimitExceeded:  # pragma: no cover - rate limit
        time.sleep(exchange.rateLimit / 1000.0)
        return []
    except ValueError:  # limit too high
        raise
    except Exception as exc:  # pragma: no cover - network
        LOG.warning("Phemex fetch failed: %s", exc)
        return None


def _download_btc_hourly(progress_cb=None) -> pd.DataFrame:
    """Return BTC prices read from CSV plus live Phemex extension."""

    csv_path = os.path.join(os.path.dirname(__file__), "..", "Gemini_BTCUSD_1h.csv")
    if not os.path.isfile(csv_path):
        LOG.warning("BTC CSV not found: %s", csv_path)
        return pd.DataFrame()

    with open(csv_path, "r", newline="") as fh:
        comment = fh.readline().strip()
    df = pd.read_csv(csv_path, skiprows=1, dtype={"unix": "int64"})
    df = df.sort_values("unix").drop_duplicates("unix", keep="last")

    if NO_HEAVY:
        df.rename(columns={"close": "Close"}, inplace=True)
        return df

    exchange = ccxt.phemex({"enableRateLimit": True})
    last_unix = int(df["unix"].iloc[-1]) if not df.empty else 0
    since = last_unix
    live: list[list[int | float]] = []

    while True:
        chunk = None
        for lim in (2000, 1500, 1000, 500):
            try:
                chunk = _fetch_phemex_bars(exchange, since, lim)
                if chunk is not None:
                    break
            except ValueError:
                continue
        if not chunk:
            break
        live.extend(chunk)
        since = chunk[-1][0]
        if len(chunk) < lim:
            break

    if live:
        cols = df.columns.tolist()
        live_df = pd.DataFrame(live, columns=["unix", "open", "high", "low", "close", "Volume BTC"])  # type: ignore[list-item]
        live_df["date"] = pd.to_datetime(live_df["unix"], unit="ms")
        live_df["symbol"] = "BTC/USD"
        live_df["Volume USD"] = live_df["close"] * live_df["Volume BTC"]
        live_df = live_df[cols]
        df = pd.concat([df, live_df], ignore_index=True)
        before = len(df)
        df = df.sort_values("unix").drop_duplicates("unix", keep="last")
        if len(df) > before:
            csv_data = df.to_csv(index=False, float_format="%.8f")
            with open(csv_path, "w", newline="") as fh:
                fh.write(comment + "\n")
                fh.write(csv_data)

    df.rename(columns={"close": "Close"}, inplace=True)
    return df


def fetch_docs(query: str, progress_cb=None) -> list[dict]:
    """Return paged DOC results with retry and a 60s overall timeout."""

    if NO_HEAVY:
        return []

    base = "https://api.gdeltproject.org/api/v2/doc/docsearch"
    out: list[dict] = []
    page = 1
    last = time.time()
    start = time.time()
    attempt = 0
    while True:
        if time.time() - start > 60:
            LOG.warning("GDELT DOC fetch failed: timeout")
            return []
        params = {
            "query": query,
            "maxrecords": 250,
            "format": "json",
            "page": page,
        }
        try:
            r = requests.get(base, params=params, timeout=(8, 30))
            r.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            delay = min(2**attempt, 60)
            attempt += 1
            if time.time() - start + delay >= 60:
                LOG.warning("GDELT DOC fetch failed: %s", exc)
                return []
            time.sleep(delay)
            continue

        attempt = 0
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
        fetch_docs("bitcoin")
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
