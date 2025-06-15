"""Command line entry point for Artibot.

This script loads ``master_config.json`` and asks whether to use the live
Phemex API. It then launches the training threads, live data polling,
Tkinter dashboard and validation workers.  The dashboard runs on the main
thread so closing the window cleanly shuts down all workers.
"""

# ruff: noqa: E402

from artibot.environment import ensure_dependencies

ensure_dependencies()

import json
import logging
import os
import threading
import torch

from artibot.utils import setup_logging, get_device
from artibot.ensemble import EnsembleModel
from artibot.dataset import HourlyDataset, load_csv_hourly
from artibot.training import (
    PhemexConnector,
    csv_training_thread,
    phemex_live_thread,
)
from artibot.rl import MetaTransformerRL, meta_control_loop
from artibot.validation import validate_and_gate
from artibot.gui import TradingGUI, ask_use_prev_weights
import artibot.globals as G
from artibot.bot_app import load_master_config


CONFIG = load_master_config()


def main() -> None:
    """Prompt for live mode and run the trading bot."""

    setup_logging()
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_num_interop_threads(os.cpu_count() or 1)
    G.start_equity = 0.0
    G.live_equity = 0.0
    G.live_trade_count = 0

    ans = input("Use LIVE trading? (y/N): ").strip().lower()
    use_live = ans.startswith("y")
    CONFIG.setdefault("API", {})["LIVE_TRADING"] = use_live
    G.use_sandbox = not use_live
    mode = "LIVE" if use_live else "SANDBOX"
    logging.info("Trading mode: %s", mode)

    connector = PhemexConnector(CONFIG)
    try:
        stats = connector.get_account_stats()
        logging.info("ACCOUNT_BALANCE %s", json.dumps(stats))
        G.global_account_stats = stats

        equity = stats.get("total", {}).get("USDT", 0.0)
        G.start_equity = equity
        G.live_equity = equity
        G.live_trade_count = 0
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("Balance fetch failed: %s", exc)

    device = get_device()
    ensemble = EnsembleModel(device=device, n_models=2, lr=3e-4, weight_decay=1e-4)
    weights_dir = os.path.abspath(
        os.path.expanduser(CONFIG.get("WEIGHTS_DIR", "weights"))
    )
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "best.pt")
    use_prev_weights = ask_use_prev_weights(CONFIG.get("USE_PREV_WEIGHTS", True))
    if os.path.isfile(weights_path) and use_prev_weights:
        ensemble.load_best_weights(weights_path)

    csv_path = CONFIG["CSV_PATH"]
    if not os.path.isabs(csv_path):
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, csv_path)
    data = load_csv_hourly(csv_path)
    if not data:
        logging.error("No usable CSV data found")
        return

    stop_event = threading.Event()
    train_th = threading.Thread(
        target=csv_training_thread,
        args=(
            ensemble,
            data,
            stop_event,
            CONFIG,
            use_prev_weights,
            None,
            weights_path,
        ),
        kwargs={"debug_anomaly": __debug__},
        daemon=True,
    )
    train_th.start()

    poll_interval = CONFIG.get("LIVE_POLL_INTERVAL", 60)
    live_path = os.path.join(os.path.dirname(weights_path), "live_model.pt")
    live_feed_th = threading.Thread(
        target=phemex_live_thread,
        args=(connector, stop_event, poll_interval, ensemble, live_path),
        daemon=True,
    )
    live_feed_th.start()

    ds = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=ensemble.indicator_hparams,
        atr_threshold_k=getattr(ensemble.indicator_hparams, "atr_threshold_k", 1.5),
        train_mode=False,
    )
    clamp_min = CONFIG.get("CLAMP_MIN", -10.0)
    clamp_max = CONFIG.get("CLAMP_MAX", 10.0)
    meta_agent = MetaTransformerRL(
        ensemble=ensemble,
        lr=1e-3,
        value_range=(clamp_min, clamp_max),
        target_range=(clamp_min, clamp_max),
    )
    meta_th = threading.Thread(
        target=lambda: meta_control_loop(
            ensemble,
            ds,
            meta_agent,
            ensemble.hp,
            ensemble.indicator_hparams,
        ),
        daemon=True,
    )
    meta_th.start()

    validate_th = threading.Thread(
        target=lambda: validate_and_gate(csv_path, CONFIG), daemon=True
    )
    validate_th.start()

    # ---------------------- start feature-ingest thread ---------------------
    import artibot.feature_ingest as _fi

    ingest_th = threading.Thread(target=_fi.start_worker, daemon=True)
    ingest_th.start()

    import tkinter as tk

    root = tk.Tk()
    gui = TradingGUI(root, ensemble, weights_path, connector)  # noqa: F841

    logging.info("Tkinter dashboard starting on main thread…")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    for th in (train_th, live_feed_th, meta_th, validate_th):
        th.join()


if __name__ == "__main__":
    main()
