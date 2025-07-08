#!/usr/bin/env python3
"""Hyperparameter sweep using a simple random search."""

import argparse
import csv
import json
import logging
import math
import os
import random
import threading
import numpy as np
import torch
import yaml


from artibot.environment import ensure_dependencies
from artibot.utils.torch_threads import set_threads
from artibot.dataset import load_csv_hourly, HourlyDataset
from artibot.ensemble import EnsembleModel
from artibot.hyperparams import HyperParams, IndicatorHyperparams
from artibot.training import csv_training_thread
from artibot.backtest import robust_backtest
from artibot.utils import get_device, setup_logging


def _log_uniform(lo: float, hi: float) -> float:
    return math.exp(random.uniform(math.log(lo), math.log(hi)))


def _sample_axes(axes: dict) -> dict:
    out = {}
    for k, v in axes.items():
        if v.get("sampling") == "log_uniform":
            out[k] = _log_uniform(float(v["min"]), float(v["max"]))
        elif v.get("sampling") == "dirichlet":
            out[k] = np.random.dirichlet(np.ones(3)).tolist()
        elif "choices" in v:
            out[k] = random.choice(v["choices"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="config/hyperparams.yaml")
    args = parser.parse_args()

    setup_logging()
    set_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    ensure_dependencies()
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")[-720:]
    with open(args.config_file, "r") as f:
        axes = yaml.safe_load(f).get("experiment_axes", {})

    params = _sample_axes(axes)
    seq_len = int(params.get("window_length", 24))
    indicator_hp = IndicatorHyperparams(
        rsi_period=14, sma_period=10, macd_fast=12, macd_slow=26, macd_signal=9
    )
    ds_tmp = HourlyDataset(
        data,
        seq_len=seq_len,
        indicator_hparams=indicator_hp,
        atr_threshold_k=getattr(indicator_hp, "atr_threshold_k", 1.5),
        train_mode=False,
    )
    n_features = ds_tmp[0][0].shape[1]
    lr = float(params.get("learning_rate", 1e-4))
    entropy_beta = float(params.get("entropy_beta", 5e-4))

    rows = []
    ensemble = EnsembleModel(
        device=get_device(),
        n_models=1,
        lr=lr,
        weight_decay=1e-4,
        n_features=n_features,
        total_steps=2000,
        grad_accum_steps=4,
    )
    ensemble.indicator_hparams = indicator_hp
    ensemble.hp = HyperParams(indicator_hp=indicator_hp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch, "compile"):
        ensemble.models = [torch.compile(m) for m in ensemble.models]

    stop = threading.Event()
    neg_streak = 0
    best_ckpts: list[tuple[float, str]] = []
    epoch = 0
    while epoch < 10:
        csv_training_thread(
            ensemble,
            data,
            stop,
            {"ADAPT_TO_LIVE": False},
            use_prev_weights=False,
            max_epochs=1,
        )
        result = robust_backtest(ensemble, data)
        sharpe = result.get("sharpe", 0.0)
        reward = result.get("composite_reward", 0.0)
        ckpt_path = f"sweeps/ckpt_{epoch}.pth"
        torch.save(ensemble.models[0].state_dict(), ckpt_path)
        best_ckpts.append((reward, ckpt_path))
        best_ckpts.sort(key=lambda x: x[0], reverse=True)
        best_ckpts = best_ckpts[:3]
        if sharpe < 0:
            neg_streak += 1
        else:
            neg_streak = 0
        if neg_streak >= 3:
            break
        epoch += 1

    rows.append(
        {
            "lr": lr,
            "entropy_beta": entropy_beta,
            "reward": best_ckpts[0][0] if best_ckpts else 0.0,
            "sharpe": sharpe,
        }
    )
    os.makedirs("sweeps", exist_ok=True)
    out_path = "sweeps/results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lr", "entropy_beta", "reward", "sharpe"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logging.info(json.dumps({"results_file": out_path}))


if __name__ == "__main__":
    main()
