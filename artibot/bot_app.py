"""Entry point for running the trading bot.

This module starts the training loop, live market polling, the
meta–reinforcement learning agent and the Tkinter GUI. API credentials
are loaded from ``master_config.json`` so no secrets live in the codebase.
"""

# ruff: noqa: F403, F405, E402

import datetime
import json
import logging
import os
import threading
import tkinter as tk

import openai
import torch


def load_master_config(path: str = "master_config.json") -> dict:
    """Return configuration loaded from ``path`` located at the repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    cfg_path = os.path.join(root, path)
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


from .dataset import HourlyDataset, load_csv_hourly
from .ensemble import EnsembleModel
import artibot.globals as G
from .gui import TradingGUI
from .rl import MetaTransformerRL, meta_control_loop
from .training import (
    PhemexConnector,
    csv_training_thread,
    phemex_live_thread,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Configuration – loaded from ``master_config.json`` at repo root
# ---------------------------------------------------------------------------
CONFIG = load_master_config()


def run_bot(max_epochs: int | None = None) -> None:
    """Launch all threads and the Tkinter GUI."""

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    G.global_training_loss = []
    G.global_validation_loss = []
    G.global_backtest_profit = []
    G.global_equity_curve = []
    G.global_ai_adjustments_log = "No adjustments yet"
    G.global_current_prediction = None
    G.global_ai_confidence = None
    G.epoch_count = 0
    G.global_attention_weights_history = []
    G.global_ai_adjustments = ""

    config = CONFIG
    G.global_min_hold_seconds = config.get(
        "MIN_HOLD_SECONDS", G.global_min_hold_seconds
    )
    openai.api_key = config["CHATGPT"]["API_KEY"]
    csv_path = config["CSV_PATH"]
    if not os.path.isabs(csv_path):
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "..", csv_path)
    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    logging.info("%s", json.dumps({"event": "load_csv", "path": csv_path}))
    data = load_csv_hourly(csv_path)

    if len(data) < 10:
        logging.error("No usable CSV data found")
        G.set_status("CSV load failed", "")
        return

    use_prev_weights = bool(config.get("USE_PREV_WEIGHTS", True))
    if os.path.isfile("best_model_weights.pth"):
        if not use_prev_weights:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"best_model_weights_backup_{ts}.pth"
            try:
                os.rename("best_model_weights.pth", backup)
                logging.info("%s", json.dumps({"event": "backup", "file": backup}))
            except OSError:
                logging.warning("Failed to backup existing weights")
        else:
            use_prev_weights = True
    else:
        use_prev_weights = False

    from .utils import get_device

    device = get_device()

    ensemble = EnsembleModel(device=device, n_models=2, lr=3e-4, weight_decay=1e-4)
    connector = PhemexConnector(config)
    stop_event = threading.Event()

    train_th = threading.Thread(
        target=csv_training_thread,
        args=(ensemble, data, stop_event, config, use_prev_weights, max_epochs),
        kwargs={"debug_anomaly": __debug__},
        daemon=True,
    )
    train_th.start()

    poll_interval = config["LIVE_POLL_INTERVAL"]
    phemex_th = threading.Thread(
        target=phemex_live_thread,
        args=(connector, stop_event, poll_interval),
        daemon=True,
    )
    phemex_th.start()

    ds = HourlyDataset(data, seq_len=24, threshold=G.GLOBAL_THRESHOLD, train_mode=False)
    meta_agent = MetaTransformerRL(ensemble=ensemble, lr=1e-3)
    meta_th = threading.Thread(
        target=lambda: meta_control_loop(ensemble, ds, meta_agent), daemon=True
    )
    meta_th.start()

    root = tk.Tk()
    TradingGUI(root, ensemble)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    threads = [("train", train_th), ("phemex", phemex_th), ("meta", meta_th)]
    for name, th in threads:
        if th.is_alive():
            th.join(timeout=5.0)
            if th.is_alive():
                logging.warning("%s thread failed to terminate", name)
            else:
                logging.info("%s thread stopped", name)
        else:
            logging.info("%s thread already stopped", name)

    with open("training_history.json", "w") as f:
        json.dump(
            {
                "global_training_loss": G.global_training_loss,
                "global_validation_loss": G.global_validation_loss,
                "global_backtest_profit": G.global_backtest_profit,
            },
            f,
        )
    ensemble.save_best_weights("best_model_weights.pth")
    save_checkpoint()


if __name__ == "__main__":
    run_bot()
