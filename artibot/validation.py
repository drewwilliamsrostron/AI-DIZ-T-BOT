"""Monthly walk-forward validation utilities."""

import threading
from typing import Iterable
import torch

import numpy as np
import pandas as pd
import logging

from .constants import FEATURE_DIMENSION

from .feature_manager import sanitize_features, FeatureDimensionError


import artibot.globals as G
from .backtest import robust_backtest
from .dataset import load_csv_hourly, HourlyDataset
from .ensemble import EnsembleModel
from .hyperparams import IndicatorHyperparams, WARMUP_STEPS
from .training import csv_training_thread
from artibot.core.device import get_device

YEAR_HOURS = 365 * 24
MONTH_SECONDS = 30 * 24 * 3600


def validate_dataset(dataset: np.ndarray) -> np.ndarray:
    """Return sanitized ``dataset`` or raise :class:`FeatureDimensionError`."""

    if dataset.shape[1] != FEATURE_DIMENSION:
        raise FeatureDimensionError(
            f"Dataset has {dataset.shape[1]} features, expected {FEATURE_DIMENSION}"
        )

    sanitized = sanitize_features(dataset)
    nan_count = int(np.isnan(sanitized).sum())
    inf_count = int(np.isinf(sanitized).sum())
    if nan_count > 0 or inf_count > 0:
        print(f"[WARN] Sanitization cleared {nan_count} NaNs and {inf_count} Infs")

    return sanitized


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
    """Train on 6 months, test the next month then roll forward."""
    data = load_csv_hourly(csv_path, cfg=config)
    if not data:
        return []
    raw_data = np.array(data, dtype=float)
    # Raw CSV data only contains OHLCV columns. Skip dimension checks and
    # simply sanitise before feature generation.
    data = sanitize_features(raw_data)
    device = get_device()

    indicator_hp = IndicatorHyperparams()
    ds_tmp = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=indicator_hp,
        atr_threshold_k=getattr(indicator_hp, "atr_threshold_k", 1.5),
        train_mode=False,
    )
    n_features = ds_tmp[0][0].shape[1]

    ensemble = EnsembleModel(
        device=device,
        n_models=1,
        lr=3e-4,
        weight_decay=1e-4,
        n_features=n_features,
        warmup_steps=WARMUP_STEPS,
        indicator_hp=indicator_hp,
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    from artibot.utils.safe_compile import safe_compile

    ensemble.models = [safe_compile(m) for m in ensemble.models]
    results = []
    one_month = YEAR_HOURS // 12
    seven_months = 7 * one_month
    for start in range(0, len(data) - seven_months + 1, one_month):
        train = data[start : start + 6 * one_month]
        test = data[start + 6 * one_month : start + seven_months]
        if len(test) < one_month:
            break
        stop_event = threading.Event()
        csv_training_thread(
            ensemble,
            train,
            stop_event,
            config,
            use_prev_weights=False,
            max_epochs=1,
            update_globals=False,
        )
        # [FIXED]# debug logging for test set
        if not isinstance(test, np.ndarray):
            test = np.array(test)
        print(f"[VALIDATION] Test data shape: {test.shape}")
        if hasattr(test, "iloc"):
            sample = test.iloc[0, :FEATURE_DIMENSION]
        else:
            sample = test[0, :FEATURE_DIMENSION]
        print(f"[VALIDATION] Feature sample: {sample}")
        metrics = robust_backtest(
            ensemble, test, indicator_hp=ensemble.indicator_hparams
        )
        if metrics.get("trades", 0) == 0:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
        else:
            G.push_backtest_metrics(metrics)
        results.append(metrics)
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
    gate_nuclear_key(
        flat,
        threshold=float(config.get("RISK_FILTER", config).get("MIN_REWARD", 1.0)),
    )
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
