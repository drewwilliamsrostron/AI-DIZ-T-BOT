"""Dataframe risk filtering utilities."""

from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def apply(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Return either ``df`` or a filtered copy depending on ``enabled``."""

    if not enabled:
        return df

    n_total = len(df)
    mask = (
        df["open"].gt(0)
        & df["high"].gt(0)
        & df["low"].gt(0)
        & df["close"].gt(0)
        & df["volume_btc"].ge(0)
        & df["high"].ge(df["low"])
    )
    filtered = df.loc[mask].copy()
    n_dropped = n_total - len(filtered)
    logger.debug("[RISK_FILTER] pruned %d of %d rows", n_dropped, n_total)
    return filtered


__all__ = ["apply"]
