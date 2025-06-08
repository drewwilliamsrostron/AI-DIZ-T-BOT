#!/usr/bin/env python3
"""Grid search over LR and TP/SL multipliers."""

import csv
import itertools
import json
import logging
import threading
import os
import numpy as np


from artibot.environment import ensure_dependencies
from artibot.dataset import load_csv_hourly
from artibot.ensemble import EnsembleModel, reject_if_risky
from artibot.training import csv_training_thread
from artibot.backtest import robust_backtest
from artibot.utils import get_device, setup_logging


def main() -> None:
    setup_logging()
    ensure_dependencies()
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[-720:]
    lrs = [1e-4, 5e-5, 1e-5]
    mults = [1.5, 2.0, 2.5]
    rows = []
    for lr, sl_m, tp_m in itertools.product(lrs, mults, mults):
        ensemble = EnsembleModel(
            device=get_device(), n_models=1, lr=lr, weight_decay=1e-4
        )
        import artibot.globals as G

        G.global_SL_multiplier = sl_m
        G.global_TP_multiplier = tp_m
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
        attn_entropy = (
            float(np.mean(G.global_attention_entropy_history[-100:]))
            if G.global_attention_entropy_history
            else 0.0
        )
        if reject_if_risky(result["sharpe"], result["max_drawdown"], attn_entropy):
            continue
        rows.append(
            {
                "lr": lr,
                "sl_mult": sl_m,
                "tp_mult": tp_m,
                "reward": result["composite_reward"],
                "mean_sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
                "attn_entropy": attn_entropy,
            }
        )
    os.makedirs("sweeps", exist_ok=True)
    out_path = "sweeps/results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr",
                "sl_mult",
                "tp_mult",
                "reward",
                "mean_sharpe",
                "max_drawdown",
                "attn_entropy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logging.info(json.dumps({"results_file": out_path}))


if __name__ == "__main__":
    main()
