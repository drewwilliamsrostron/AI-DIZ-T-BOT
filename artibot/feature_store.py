"""
Local feature store backed by DuckDB.

Tables
------
news_sentiment(ts TIMESTAMP PRIMARY KEY, score DOUBLE)
macro_surprise(ts TIMESTAMP PRIMARY KEY, surprise_z DOUBLE)
realised_vol(ts TIMESTAMP PRIMARY KEY, rvol DOUBLE)

All getters return np.float32.
"""

from __future__ import annotations
import datetime as _dt
import os
import duckdb
import numpy as np

###############################################################################
#  initialise / migrate
###############################################################################
_DB_PATH = os.path.join(os.path.dirname(__file__), "_features.duckdb")
_con = duckdb.connect(_DB_PATH, read_only=False)

# create-if-missing
_con.execute(
    """
CREATE TABLE IF NOT EXISTS news_sentiment  (
    ts TIMESTAMP PRIMARY KEY,
    score DOUBLE
);
"""
)
_con.execute(
    """
CREATE TABLE IF NOT EXISTS macro_surprise (
    ts TIMESTAMP PRIMARY KEY,
    surprise_z DOUBLE
);
"""
)
_con.execute(
    """
CREATE TABLE IF NOT EXISTS realised_vol (
    ts TIMESTAMP PRIMARY KEY,
    rvol DOUBLE
);
"""
)

# migrate-if-old (adds missing columns without raising if they already exist)
_con.execute("ALTER TABLE news_sentiment  ADD COLUMN IF NOT EXISTS score      DOUBLE;")
_con.execute("ALTER TABLE macro_surprise ADD COLUMN IF NOT EXISTS surprise_z DOUBLE;")
_con.execute("ALTER TABLE realised_vol  ADD COLUMN IF NOT EXISTS rvol        DOUBLE;")

###############################################################################
#  ingestion helpers
###############################################################################
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

###############################################################################
#  retrieval helpers (hourly resolution, last-known-value)
###############################################################################
def _fetch_scalar(column: str, table: str, ts: int) -> float:
    """Return the most recent value <= *ts* from *table.column* (or 0.0)."""
    res = _con.execute(
        f"""SELECT {column}
              FROM {table}
             WHERE ts <= ?
         ORDER BY ts DESC
            LIMIT 1""",
        (_dt.datetime.fromtimestamp(ts),),
    ).fetchone()
    return float(res[0]) if res else 0.0


def news_sentiment(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("score", "news_sentiment", ts))


def macro_surprise(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("surprise_z", "macro_surprise", ts))


def realised_vol(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("rvol", "realised_vol", ts))
