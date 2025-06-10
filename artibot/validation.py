"""Monthly walk-forward validation utilities."""

import threading
from typing import Iterable

import numpy as np
import pandas as pd
import logging

import artibot.globals as G
from .backtest import robust_backtest
from .dataset import load_csv_hourly
from .ensemble import EnsembleModel
from .training import csv_training_thread
from .utils import get_device

YEAR_HOURS = 365 * 24
MONTH_SECONDS = 30 * 24 * 3600


def equity_returns(curve: Iterable[tuple[int, float]]) -> list[float]:
    """Return daily returns from an equity curve."""
    if not curve:
        return []
    df = pd.DataFrame(curve, columns=["ts", "balance"])
    if df["ts"].max() > 1_000_000_000_000:
        df["ts"] //= 1000
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    df.set_index("dt", inplace=True)
    return df["balance"].resample("1D").last().pct_change().dropna().to_list()


def monte_carlo_sharpe(returns: Iterable[float], runs: int = 1000) -> list[float]:
    """Return distribution of Sharpe ratios from resampled ``returns``."""
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return []
    dist: list[float] = []
    for _ in range(runs):
        sample = np.random.choice(arr, size=arr.size, replace=True)
        mu = sample.mean()
        sigma = sample.std() or 1e-8
        dist.append((mu * np.sqrt(252)) / sigma)
    return dist


def walk_forward_analysis(csv_path: str, config: dict) -> list[dict]:
    """Train and evaluate on rolling 5-year windows."""
    data = load_csv_hourly(csv_path)
    if not data:
        return []
    device = get_device()
    ensemble = EnsembleModel(device=device, n_models=1, lr=3e-4, weight_decay=1e-4)
    results = []
    one_year = YEAR_HOURS
    six_years = 6 * one_year
    for start in range(0, len(data) - six_years + 1, one_year):
        train = data[start : start + 5 * one_year]
        test = data[start + 5 * one_year : start + six_years]
        if len(test) < one_year:
            break
        stop_event = threading.Event()
        csv_training_thread(
            ensemble,
            train,
            stop_event,
            config,
            use_prev_weights=False,
            max_epochs=1,
        )
        results.append(robust_backtest(ensemble, test))
    return results


def gate_nuclear_key(sharpes: Iterable[float], threshold: float = 1.0) -> bool:
    """Update ``G.nuclear_key_enabled`` based on ``sharpes``."""
    mean_sharpe = float(np.mean(list(sharpes))) if sharpes else 0.0
    enabled = mean_sharpe >= threshold
    G.set_nuclear_key(enabled)
    return enabled


def validate_and_gate(csv_path: str, config: dict) -> dict:
    """Run validation and update globals."""
    logging.info("VALIDATION_START")
    results = walk_forward_analysis(csv_path, config)
    distributions = [
        monte_carlo_sharpe(equity_returns(r.get("equity_curve", []))) for r in results
    ]
    flat = [s for dist in distributions for s in dist]
    gate_nuclear_key(flat, threshold=float(config.get("MIN_SHARPE", 1.0)))
    summary = {
        "windows": len(results),
        "mean_sharpe": float(np.mean(flat)) if flat else 0.0,
    }
    G.global_validation_summary = summary
    logging.info("VALIDATION_DONE", extra=summary)
    return summary


def schedule_monthly_validation(
    csv_path: str, config: dict, *, interval: float = MONTH_SECONDS
) -> threading.Timer:
    """Start recurring validation every ``interval`` seconds."""

    # TODO: replace ``threading.Timer`` with ``APScheduler`` for more robust
    # scheduling and persistence of jobs.

    def _run() -> None:
        logging.info("VALIDATION_TRIGGER")
        validate_and_gate(csv_path, config)
        schedule_monthly_validation(csv_path, config, interval=interval)

    timer = threading.Timer(interval, _run)
    timer.daemon = True
    timer.start()
    return timer
