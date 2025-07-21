#!/usr/bin/env python3
"""Run a 10-epoch training smoke test on a small slice of data."""

import threading
import json
import logging
import argparse
import numpy as np
import torch
import os

from artibot.environment import ensure_dependencies
from artibot.utils.torch_threads import set_threads
from artibot.dataset import load_csv_hourly, HourlyDataset
from artibot.ensemble import EnsembleModel
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.training import csv_training_thread
from backtest import run_backtest
from artibot.utils import get_device, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override default LR (HyperParams)",
    )
    args = parser.parse_args()

    setup_logging()
    set_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    ensure_dependencies()
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[:500]
    indicator_hp = IndicatorHyperparams()
    ds_tmp = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=indicator_hp,
        atr_threshold_k=getattr(indicator_hp, "atr_threshold_k", 1.5),
        train_mode=False,
    )
    n_features = ds_tmp[0][0].shape[1]
    hp = HyperParams(indicator_hp=indicator_hp)
    if args.learning_rate:
        hp.learning_rate = args.learning_rate
    ensemble = EnsembleModel(
        device=get_device(),
        n_models=1,
        lr=hp.learning_rate,
        weight_decay=1e-4,
        n_features=n_features,
        total_steps=2000,
        grad_accum_steps=4,
    )
    ensemble.indicator_hparams = indicator_hp
    ensemble.hp = hp
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
    data_path = "Gemini_BTCUSD_1h.csv"
    try:
        result = run_backtest(ensemble, data_path)
        arr = np.array(data[:500], dtype=np.float64)
        mean_range = (arr[:, 2] - arr[:, 3]).mean()
        logging.info(
            json.dumps({"reward": result["composite_reward"], "range": mean_range})
        )
    except Exception as e:
        logging.warning(f"Backtest failed: {e}")

    if args.summary:
        with open("bot.log") as f, open("smoke_summary.log", "w") as out:
            for line in f:
                if "ATTN_STATS" in line or "REJECTED" in line:
                    out.write(line)
        print("Wrote smoke_summary.log")


if __name__ == "__main__":
    main()
