"""Reward shaping utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def ema(series: torch.Tensor, tau: float = 96.0) -> torch.Tensor:
    """Return exponential moving average of ``series`` with decay ``tau``."""

    alpha = 1 - torch.exp(torch.tensor(-1.0 / tau))
    out = torch.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * series[i]
    return out


def differential_sharpe(returns: torch.Tensor) -> torch.Tensor:
    """Return Sharpe ratio of the first difference of ``returns``."""

    dr = torch.diff(returns, prepend=returns[:1])
    return dr.mean() / (dr.std(unbiased=False) + 1e-6)


def sortino_ratio(returns: torch.Tensor, target: float = 0.0) -> torch.Tensor:
    """Return Sortino ratio of ``returns`` relative to ``target``.

    Uses downside deviation to penalise negative volatility. A small
    epsilon is added to avoid division by zero.
    """
    downside = torch.clamp(target - returns, min=0.0)
    dd = downside.std(unbiased=False)
    excess = returns.mean() - target
    return excess / (dd + 1e-6)


def omega_ratio(returns: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Return Omega ratio of ``returns`` around ``threshold``."""
    gains = torch.clamp(returns - threshold, min=0.0).mean()
    losses = torch.clamp(threshold - returns, min=0.0).mean()
    return gains / (losses + 1e-6)


def calmar_ratio(net_pct: float, max_drawdown: float, period_days: int) -> float:
    """Return Calmar ratio based on annualised return and drawdown."""
    annualised = net_pct / max(period_days / 365.0, 1e-6)
    return annualised / (abs(max_drawdown) + 1e-6)


def trade_pnl(df: pd.DataFrame) -> np.ndarray:
    """Return per-trade profit and loss values.

    Parameters
    ----------
    df:
        DataFrame containing trade records with ``entry_price``, ``exit_price``,
        ``side`` and ``size`` columns.

    Returns
    -------
    numpy.ndarray
        Array of P&L for each trade in the same currency units as ``size``.
    """

    if df.empty:
        return np.array([], dtype=float)

    entry = df["entry_price"].to_numpy(dtype=float)
    exit_p = df["exit_price"].to_numpy(dtype=float)
    size = df["size"].to_numpy(dtype=float)
    side = df["side"].to_numpy()

    long_mask = side == "long"
    pnl = np.empty(len(df), dtype=float)
    pnl[long_mask] = (exit_p[long_mask] - entry[long_mask]) * size[long_mask]
    pnl[~long_mask] = (entry[~long_mask] - exit_p[~long_mask]) * np.abs(
        size[~long_mask]
    )
    return pnl
