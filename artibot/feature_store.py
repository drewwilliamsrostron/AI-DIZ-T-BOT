"""Local feature store backed by DuckDB + Feast for sentiment, macro surprises,
   realised volatility.  All functions return *np.float32* scalars."""

from __future__ import annotations
import datetime as _dt
import os
import duckdb
import numpy as np

_DB_PATH = os.path.join(os.path.dirname(__file__), "_features.duckdb")
_con = duckdb.connect(_DB_PATH, read_only=False)

# ensure tables exist
_con.execute(
    """
CREATE TABLE IF NOT EXISTS news_sentiment  (
    ts  TIMESTAMP PRIMARY KEY,
    score DOUBLE
);
"""
)
_con.execute(
    """
CREATE TABLE IF NOT EXISTS macro_surprise (
    ts  TIMESTAMP PRIMARY KEY,
    surprise_z DOUBLE
);
"""
)
_con.execute(
    """
CREATE TABLE IF NOT EXISTS realised_vol (
    ts  TIMESTAMP PRIMARY KEY,
    rvol DOUBLE
);
"""
)


# ---------- ingestion helpers ----------
def upsert_news(ts: int, score: float) -> None:
    _con.execute(
        "INSERT OR REPLACE INTO news_sentiment VALUES (?, ?)",
        (_dt.datetime.fromtimestamp(ts), float(score)),
    )


def upsert_macro(ts: int, z: float) -> None:
    _con.execute(
        "INSERT OR REPLACE INTO macro_surprise VALUES (?, ?)",
        (_dt.datetime.fromtimestamp(ts), float(z)),
    )


def upsert_rvol(ts: int, rvol: float) -> None:
    _con.execute(
        "INSERT OR REPLACE INTO realised_vol VALUES (?, ?)",
        (_dt.datetime.fromtimestamp(ts), float(rvol)),
    )


# ---------- retrieval API (hourly resolution) ----------
def _fetch_scalar(col: str, table: str, ts: int) -> float:
    res = _con.execute(
        f"SELECT {col} FROM {table} WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
        (_dt.datetime.fromtimestamp(ts),),
    ).fetchone()
    return float(res[0]) if res else 0.0


def news_sentiment(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("score", "news_sentiment", ts))


def macro_surprise(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("surprise_z", "macro_surprise", ts))


def realised_vol(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("rvol", "realised_vol", ts))
