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
import threading
import numpy as np
import json
import pathlib

_STORE = pathlib.Path(".feature_dim.json")


# [FIX]# helper to avoid invalid divisions
def safe_divide(a, b, default=0.0):
    """Avoid division by zero and invalid values"""

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = default
    return result


# Fixed feature dimension used by training pipelines
FEATURE_DIM = 17
_FROZEN_DIM = None


def freeze_feature_dim(dim: int) -> int:
    """Persist the maximum feature dimension across runs."""

    if _STORE.exists():
        dim = max(dim, json.load(_STORE.open())["dim"])
    json.dump({"dim": dim}, _STORE.open("w"))
    return dim


def get_frozen_dim() -> int:
    """Return the frozen feature dimension or ``FEATURE_DIM`` when unset."""

    return json.load(_STORE.open())["dim"] if _STORE.exists() else FEATURE_DIM


###############################################################################
#  initialise / migrate
###############################################################################
# ---------------------------------------------------------------------------
# thread-local connection helper
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(os.path.dirname(__file__), "_features.duckdb")

_tls = threading.local()


def _get_con() -> duckdb.DuckDBPyConnection:
    """Return a thread-local DuckDB connection."""

    conn = getattr(_tls, "db", None)
    if conn is None:
        conn = duckdb.connect(_DB_PATH, read_only=False)
        _tls.db = conn
        _ensure_tables(conn)
    else:
        try:
            conn.execute("SELECT 1")
        except duckdb.Error:
            conn = duckdb.connect(_DB_PATH, read_only=False)
            _tls.db = conn
            _ensure_tables(conn)
    return conn


def _con() -> duckdb.DuckDBPyConnection:
    """Backward compatible alias used in tests."""

    return _get_con()


# ---------------------------------------------------------------------------
# table creation + idempotent migrations
# ---------------------------------------------------------------------------
_DDL = {
    "news_sentiment": ("score", "DOUBLE"),
    "macro_surprise": ("surprise_z", "DOUBLE"),
    "realised_vol": ("rvol", "DOUBLE"),
}


def _ensure_tables(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create tables and add missing columns if necessary."""

    conn = conn or _get_con()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS news_sentiment (ts TIMESTAMP PRIMARY KEY);"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS macro_surprise (ts TIMESTAMP PRIMARY KEY);"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS realised_vol (ts TIMESTAMP PRIMARY KEY);"""
    )

    for tbl, (col, typ) in _DDL.items():
        conn.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {typ};")


###############################################################################
#  ingestion helpers
###############################################################################
def upsert_news(ts: int, score: float) -> None:
    _get_con().execute(
        "INSERT OR REPLACE INTO news_sentiment VALUES (?, ?)",
        (_dt.datetime.fromtimestamp(ts), float(score)),
    )


def upsert_macro(ts: int, z: float) -> None:
    _get_con().execute(
        "INSERT OR REPLACE INTO macro_surprise VALUES (?, ?)",
        (_dt.datetime.fromtimestamp(ts), float(z)),
    )


def upsert_rvol(ts: int, rvol: float) -> None:
    _get_con().execute(
        "INSERT OR REPLACE INTO realised_vol VALUES (?, ?)",
        (_dt.datetime.fromtimestamp(ts), float(rvol)),
    )


###############################################################################
#  retrieval helpers (hourly resolution, last-known-value)
###############################################################################
def _fetch_scalar(column: str, table: str, ts: int) -> float:
    """Return the most recent value <= *ts* from *table.column* (or ``0.0``)."""

    conn = _get_con()
    sql = f"SELECT {column} FROM {table} WHERE ts<=? ORDER BY ts DESC LIMIT 1"
    try:
        res = conn.execute(sql, (_dt.datetime.fromtimestamp(ts),)).fetchone()
        return float(res[0]) if res else 0.0
    except duckdb.InvalidInputException:
        _tls.db = duckdb.connect(_DB_PATH, read_only=False)
        _ensure_tables(_tls.db)
        res = _tls.db.execute(sql, (_dt.datetime.fromtimestamp(ts),)).fetchone()
        return float(res[0]) if res else 0.0
    except duckdb.BinderException as exc:
        if "column" in str(exc).lower():
            _ensure_tables(conn)
            res = conn.execute(sql, (_dt.datetime.fromtimestamp(ts),)).fetchone()
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
        _ensure_tables(_get_con())
        print("feature_store: schema verified / repaired ✅")
        sys.exit(0)

    if "--info" in sys.argv:
        info = {
            "tables": _get_con()
            .execute(
                "SELECT table_name, column_name, column_type "
                "FROM duckdb_columns WHERE table_schema='main'"
            )
            .fetchall()
        }
        print(json.dumps(info, indent=2))
        sys.exit(0)
