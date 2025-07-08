"""Reward shaping utilities."""

from __future__ import annotations

import pandas as pd
import torch


def ema(series: torch.Tensor, tau: int = 96) -> torch.Tensor:
    """Return exponential moving average of ``series`` with decay ``tau``."""

    alpha = 1 - torch.exp(torch.tensor(-1.0 / tau))
    out = [series[0]]
    for prev, curr in zip(series[:-1], series[1:]):
        out.append((1 - alpha) * prev + alpha * curr)
    return torch.stack(out)


def differential_sharpe(returns: pd.Series) -> float:
    """Return Sharpe ratio of the first difference of ``returns``."""

    dr = returns.diff().fillna(0)
    return float(dr.mean() / (dr.std() + 1e-6))
