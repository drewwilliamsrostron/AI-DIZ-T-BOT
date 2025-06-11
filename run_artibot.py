"""Command line entry point for Artibot.

This script loads ``master_config.json`` and asks whether to use the live
Phemex API. It then launches the training threads, live data polling,
Tkinter dashboard and validation workers in the background while running
an order execution loop in the foreground.
"""

# ruff: noqa: E402

from artibot.environment import ensure_dependencies

ensure_dependencies()

import json
import logging
import math
import os
import threading
import time
from typing import Any

import numpy as np
import talib
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


def get_account_equity(exchange: Any) -> float:
    """Return total BTC equity converted to USDT."""
    spot = exchange.fetch_balance()
    btc_spot = spot.get("BTC", {}).get("total", 0)

    params = {"type": "swap", "code": "BTC"}
    swap = exchange.fetch_balance(params=params)
    btc_swap = swap.get("BTC", {}).get("total", 0)

    if btc_spot == btc_swap == 0:
        logging.warning("No BTC balance found – equity = 0")

    btc_usdt = exchange.fetch_ticker("BTC/USDT")["close"]
    return round((btc_spot + btc_swap) * btc_usdt, 8)


def _gui_thread(ensemble: EnsembleModel, weights_path: str) -> None:
    import tkinter as tk

    root = tk.Tk()
    TradingGUI(root, ensemble, weights_path)
    root.mainloop()


def main() -> None:
    """Prompt for live mode and run the trading bot."""

    setup_logging()
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_num_interop_threads(os.cpu_count() or 1)

    ans = input("Use LIVE trading? (y/N): ").strip().lower()
    use_live = ans.startswith("y")
    CONFIG.setdefault("API", {})["LIVE_TRADING"] = use_live
    G.use_sandbox = not use_live
    mode = "LIVE" if use_live else "SANDBOX"
    logging.info("Trading mode: %s", mode)

    connector = PhemexConnector(CONFIG)
    try:
        bal = connector.exchange.fetch_balance()
        logging.info("ACCOUNT_BALANCE %s", json.dumps(bal))
        G.global_account_stats = bal
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("Balance fetch failed: %s", exc)

    device = get_device()
    ensemble = EnsembleModel(device=device, n_models=2, lr=3e-4, weight_decay=1e-4)
    weights_dir = os.path.abspath(
        os.path.expanduser(CONFIG.get("WEIGHTS_DIR", "weights"))
    )
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "best_model_weights.pth")
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

    gui_th = threading.Thread(
        target=_gui_thread, args=(ensemble, weights_path), daemon=True
    )
    gui_th.start()

    validate_th = threading.Thread(
        target=lambda: validate_and_gate(csv_path, CONFIG), daemon=True
    )
    validate_th.start()

    risks = CONFIG.get("RISKS", {})
    timeframe = CONFIG.get("TIMEFRAME", "1h")
    leverage = float(CONFIG.get("LEVERAGE", 10))
    poll_int = float(poll_interval)

    try:
        while True:
            logging.info("Fetching latest bars…")
            try:
                bars = connector.fetch_latest_bars(limit=100)
            except Exception as exc:  # pragma: no cover - network errors
                logging.error("fetch_ohlcv failed: %s", exc)
                time.sleep(poll_int)
                continue
            if not bars:
                logging.warning(
                    "No bars retrieved from exchange, retrying in %.0f seconds…",
                    poll_int,
                )
                time.sleep(poll_int)
                continue

            arr = np.array(bars, dtype=np.float64)
            closes = arr[:, 4]
            sma = np.convolve(closes, np.ones(10) / 10, mode="same")
            rsi = talib.RSI(closes, timeperiod=14)
            macd, _, _ = talib.MACD(closes)
            feats = np.column_stack(
                [
                    arr[:, 1:6],
                    sma.astype(np.float32),
                    rsi.astype(np.float32),
                    macd.astype(np.float32),
                ]
            )
            feats = np.nan_to_num(feats).astype(np.float32)
            seq_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            idx, conf, _ = ensemble.predict(seq_t)
            side_signal = {0: "buy", 1: "sell", 2: "hold"}[idx]
            logging.info("SIGNAL %s conf=%.3f", side_signal, conf)

            if G.trading_paused or side_signal == "hold":
                time.sleep(poll_int)
                continue

            try:
                equity = get_account_equity(connector.exchange)
                bal = connector.exchange.fetch_balance()
                G.global_account_stats = bal
            except Exception as exc:  # pragma: no cover - network errors
                logging.error("Equity fetch error: %s", exc)
                time.sleep(poll_int)
                continue

            risk_pct = float(risks.get(timeframe, 10)) / 100.0
            usd_risk = equity * risk_pct
            contracts = usd_risk * leverage

            market = connector.exchange.market(connector.symbol)
            step = market.get("precision", {}).get("amount") or 1
            min_qty = market.get("limits", {}).get("amount", {}).get("min") or 1
            contracts = math.floor(contracts / step) * step

            if not G.is_nuclear_key_enabled():
                logging.info("NUCLEAR_KEY_DISABLED – skipping trade")
                time.sleep(poll_int)
                continue

            if contracts < min_qty:
                logging.info(
                    "ORDER_SKIPPED – qty %.2f < min lot %d", contracts, min_qty
                )
                time.sleep(poll_int)
                continue

            try:
                order = connector.create_order(side_signal, contracts)
                logging.info("ORDER_PLACED %s", order)
                G.global_position_side = side_signal
                G.global_position_size = contracts
            except Exception as exc:  # pragma: no cover - order errors
                logging.error("Order failed: %s", exc)
                G.trading_paused = True

            time.sleep(poll_int)
    except KeyboardInterrupt:
        logging.info("Stopping…")
    finally:
        stop_event.set()
        for th in [train_th, phemex_th, meta_th, validate_th, gui_th]:
            if th.is_alive():
                th.join(timeout=5.0)
        try:
            G.cancel_open_orders()
            G.close_position()
        except Exception as exc:  # pragma: no cover - network errors
            logging.error("Cleanup failed: %s", exc)


if __name__ == "__main__":
    main()
