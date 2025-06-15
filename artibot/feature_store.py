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

# ---------------------------------------------------------------------------
# table creation + idempotent migrations
# ---------------------------------------------------------------------------
_DDL = {
    "news_sentiment": ("score", "DOUBLE"),
    "macro_surprise": ("surprise_z", "DOUBLE"),
    "realised_vol": ("rvol", "DOUBLE"),
}


def _ensure_tables() -> None:
    """Create tables and add missing columns if necessary."""

    _con.execute(
        """CREATE TABLE IF NOT EXISTS news_sentiment (ts TIMESTAMP PRIMARY KEY);"""
    )
    _con.execute(
        """CREATE TABLE IF NOT EXISTS macro_surprise (ts TIMESTAMP PRIMARY KEY);"""
    )
    _con.execute(
        """CREATE TABLE IF NOT EXISTS realised_vol (ts TIMESTAMP PRIMARY KEY);"""
    )

    for tbl, (col, typ) in _DDL.items():
        _con.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {typ};")


_ensure_tables()


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
    """Return the most recent value <= *ts* from *table.column* (or ``0.0``)."""

    sql = f"SELECT {column} FROM {table} WHERE ts<=? ORDER BY ts DESC LIMIT 1"
    try:
        res = _con.execute(sql, (_dt.datetime.fromtimestamp(ts),)).fetchone()
        return float(res[0]) if res else 0.0
    except (duckdb.InvalidInputException, duckdb.BinderException) as exc:
        if "column" in str(exc).lower():
            _ensure_tables()
            res = _con.execute(sql, (_dt.datetime.fromtimestamp(ts),)).fetchone()
            return float(res[0]) if res else 0.0
        raise


def news_sentiment(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("score", "news_sentiment", ts))


def macro_surprise(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("surprise_z", "macro_surprise", ts))


def realised_vol(ts: int) -> np.float32:
    return np.float32(_fetch_scalar("rvol", "realised_vol", ts))


# ---------------------------------------------------------------------------
# CLI helper:  python -m artibot.feature_store --repair
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if "--repair" in sys.argv:
        _ensure_tables()
        print("feature_store: schema verified / repaired ✅")
        sys.exit(0)

    if "--info" in sys.argv:
        info = {
            "tables": _con.execute(
                "SELECT table_name, column_name, column_type "
                "FROM duckdb_columns WHERE table_schema='main'"
            ).fetchall()
        }
        print(json.dumps(info, indent=2))
        sys.exit(0)
