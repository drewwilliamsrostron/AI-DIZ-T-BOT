"""Reward shaping utilities."""

from __future__ import annotations

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
