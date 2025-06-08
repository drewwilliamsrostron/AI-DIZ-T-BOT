#!/usr/bin/env python3
"""Run a 50-epoch training smoke test on one month of data."""

import threading
import json
import logging
import argparse
import numpy as np

from artibot.environment import ensure_dependencies
from artibot.dataset import load_csv_hourly
from artibot.ensemble import EnsembleModel
from artibot.training import csv_training_thread
from artibot.backtest import robust_backtest
from artibot.utils import get_device, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    setup_logging()
    ensure_dependencies()
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[-720:]
    ensemble = EnsembleModel(
        device=get_device(), n_models=1, lr=1e-4, weight_decay=1e-4
    )
    stop = threading.Event()
    csv_training_thread(
        ensemble,
        data,
        stop,
        {"ADAPT_TO_LIVE": False},
        use_prev_weights=False,
        max_epochs=50,
    )
    result = robust_backtest(ensemble, data)
    arr = np.array(data[-720:], dtype=np.float64)
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
