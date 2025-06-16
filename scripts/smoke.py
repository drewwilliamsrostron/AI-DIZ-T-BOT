#!/usr/bin/env python3
"""Run a 10-epoch training smoke test on a small slice of data."""

import threading
import json
import logging
import argparse
import numpy as np
import torch

from artibot.environment import ensure_dependencies
from artibot.dataset import load_csv_hourly, HourlyDataset
from artibot.ensemble import EnsembleModel
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.training import csv_training_thread
from artibot.backtest import robust_backtest
from artibot.utils import get_device, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    setup_logging()
    ensure_dependencies()
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[:500]
    indicator_hp = IndicatorHyperparams(
        rsi_period=14, sma_period=10, macd_fast=12, macd_slow=26, macd_signal=9
    )
    ds_tmp = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=indicator_hp,
        atr_threshold_k=getattr(indicator_hp, "atr_threshold_k", 1.5),
        train_mode=False,
    )
    n_features = ds_tmp[0][0].shape[1]
    ensemble = EnsembleModel(
        device=get_device(),
        n_models=1,
        lr=1e-4,
        weight_decay=1e-4,
        n_features=n_features,
    )
    ensemble.indicator_hparams = indicator_hp
    ensemble.hp = HyperParams(indicator_hp=indicator_hp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch, "compile"):
        ensemble.models = [torch.compile(m) for m in ensemble.models]
    stop = threading.Event()
    csv_training_thread(
        ensemble,
        data,
        stop,
        {"ADAPT_TO_LIVE": False},
        use_prev_weights=False,
        max_epochs=10,
    )
    result = robust_backtest(ensemble, data)
    arr = np.array(data[:500], dtype=np.float64)
    mean_range = (arr[:, 2] - arr[:, 3]).mean()
    logging.info(
        json.dumps({"reward": result["composite_reward"], "range": mean_range})
    )

    if args.summary:
        with open("bot.log") as f, open("smoke_summary.log", "w") as out:
            for line in f:
                if "ATTN_STATS" in line or "REJECTED" in line:
                    out.write(line)
        print("Wrote smoke_summary.log")


if __name__ == "__main__":
    main()
