"""Command line entry point for Artibot.

This script loads ``master_config.json`` and asks whether to use the live
Phemex API. It then launches the training threads, live data polling,
Tkinter dashboard and validation workers.  The dashboard runs on the main
thread so closing the window cleanly shuts down all workers.
"""

# ruff: noqa: E402
from __future__ import annotations

from artibot.environment import ensure_dependencies

ensure_dependencies()

import os
import sys
import subprocess

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
import logging
import threading
import tkinter as tk
from queue import Queue, Empty


SKIP_SENTIMENT = False


def ask_skip_sentiment(default: bool = False, tk_module=None) -> bool:
    """Ask whether to skip the heavy sentiment download."""
    global SKIP_SENTIMENT
    try:
        if tk_module is None:
            from tkinter import Tk, messagebox
        else:
            Tk = tk_module.Tk
            messagebox = tk_module.messagebox
    except Exception as exc:  # pragma: no cover - headless env
        print(f"[WARN] Tk unavailable: {exc}; skipping sentiment download")
        SKIP_SENTIMENT = True
        os.environ["NO_HEAVY"] = "1"
        return True

    root = Tk()
    root.withdraw()
    try:
        result = messagebox.askyesno(
            "Skip sentiment download?",
            "GDELT is slow right now. Do you want to skip historical sentiment?",
        )
    finally:
        root.destroy()
    if result is None:
        result = default
    if result:
        print("↪ Skipping sentiment pull; continuing …")
        SKIP_SENTIMENT = True
        os.environ["NO_HEAVY"] = "1"
    else:
        SKIP_SENTIMENT = False
        os.environ.pop("NO_HEAVY", None)
    return bool(result)


def load_master_config(path: str = "master_config.json") -> dict:
    """Return configuration dictionary from the repo root."""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    try:
        with open(cfg_path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def _maybe_relaunch_for_threads() -> None:
    """Ask for CPU threads and relaunch subprocess if needed."""
    if os.environ.get("ARTIBOT_RERUN") == "1":
        return

    from tkinter import simpledialog, messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        if messagebox.askyesno(
            title="Thread tuning",
            message="Change CPU thread count before training?",
        ):
            n_threads = simpledialog.askinteger(
                "Threads",
                "Enter desired # CPU threads:",
                minvalue=1,
                maxvalue=os.cpu_count() or 1,
            )
            if n_threads:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(n_threads)
                env["MKL_NUM_THREADS"] = str(n_threads)
                env["ARTIBOT_RERUN"] = "1"
                cmd = [sys.executable, os.path.abspath(__file__)]
                print(f"\u21AA Restarting with {n_threads} CPU threads …")
                subprocess.Popen(cmd, env=env)
                sys.exit(0)
        else:
            print("\u21AA Keeping current CPU-thread setting …")
    finally:
        root.destroy()


def _launch_loading(
    root: tk.Tk, msg_queue: "Queue[tuple[float, str] | tuple[str, str]]"
) -> tuple[tk.Toplevel, tk.StringVar, object]:
    from tkinter import ttk

    root.withdraw()
    msg_var = tk.StringVar(value="Initialising…")
    win = tk.Toplevel(root)
    win.title("Loading…")
    ttk.Label(win, textvariable=msg_var, font=("Helvetica", 12)).pack(padx=10, pady=10)
    pb = ttk.Progressbar(win, length=200, mode="determinate", maximum=100)
    pb.pack(padx=10, pady=5)
    win.grab_set()

    def _poll() -> None:
        try:
            pct, msg = msg_queue.get_nowait()
            if pct == "DONE":
                win.destroy()
                root.deiconify()
                return
            msg_var.set(msg)
            pb["value"] = pct
        except Empty:
            pass
        finally:
            root.after(100, _poll)

    root.after(100, _poll)
    return win, msg_var, pb


CONFIG = load_master_config()


def main() -> None:
    """Prompt for live mode and run the trading bot."""

    ask_skip_sentiment()
    _maybe_relaunch_for_threads()

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

    setup_logging()
    root = tk.Tk()
    progress_q: Queue[tuple[float, str] | tuple[str, str]] = Queue()
    _launch_loading(root, progress_q)

    G.set_cpu_limit(CONFIG.get("CPU_LIMIT", os.cpu_count() or 1))
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
    init_done = threading.Event()
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
        kwargs={"debug_anomaly": __debug__, "init_event": init_done},
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

    def _bg_init() -> None:
        if SKIP_SENTIMENT:
            init_done.set()
            progress_q.put(("DONE", ""))
            return
        progress_q.put((0.0, "Downloading historical sentiment + macro data…"))
        try:
            import tools.backfill_gdelt as _bf

            _bf.main(progress_cb=lambda pct, msg: progress_q.put((pct, msg)))
        finally:
            init_done.set()
            progress_q.put(("DONE", ""))

    threading.Thread(target=_bg_init, daemon=True).start()

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
