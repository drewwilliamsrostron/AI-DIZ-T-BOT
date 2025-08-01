"""Command line entry point for Artibot.

This script loads ``master_config.json`` and asks whether to use the live
Phemex API. It then launches the training threads, live data polling,
Tkinter dashboard and validation workers.  The dashboard runs on the main
thread so closing the window cleanly shuts down all workers.
"""

# ruff: noqa: E402
from __future__ import annotations

from artibot.environment import ensure_dependencies
from artibot.utils.torch_threads import set_threads
from artibot.gui import startup_options_dialog
from dataclasses import fields
from artibot.ensemble import EnsembleModel
from artibot.hyperparams import IndicatorHyperparams
import artibot.globals as G

import os
import sys
import subprocess
import argparse

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
import logging
import threading
import tkinter as tk
from queue import Queue, Empty
import tomllib


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


def load_default_config(path: str = "config/default.toml") -> dict:
    """Return optional TOML configuration."""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    try:
        with open(cfg_path, "rb") as fh:
            return tomllib.load(fh)
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
                print(f"\u21aa Restarting with {n_threads} CPU threads …")
                subprocess.Popen(cmd, env=env)
                sys.exit(0)
        else:
            print("\u21aa Keeping current CPU-thread setting …")
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
DEFAULT_CFG = load_default_config()


def build_model(
    *,
    device: object,
    n_features: int,
    lr: float = 1e-3,
    entropy_beta: float | None = None,
    warmup_steps: int | None = None,
    indicator_hp: IndicatorHyperparams | None = None,
    freeze_features: bool | None = None,
) -> "EnsembleModel":
    """Return an :class:`EnsembleModel` configured with HPO params."""
    from artibot.hyperparams import WARMUP_STEPS

    model = EnsembleModel(
        device=device,
        n_models=2,
        lr=lr,
        weight_decay=0.0,
        n_features=n_features,
        total_steps=10000,
        grad_accum_steps=4,
        warmup_steps=warmup_steps or WARMUP_STEPS,
        indicator_hp=indicator_hp,
        freeze_features=freeze_features,
    )
    if entropy_beta is not None:
        model.entropy_beta = entropy_beta
    return model


def main() -> None:
    """Prompt for startup options and run the trading bot."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--no-tune", action="store_true", help="skip Optuna search")
    args, _ = parser.parse_known_args()
    dev_mode = args.dev
    no_tune = args.no_tune

    defaults = {
        "skip_sentiment": False,
        "use_live": CONFIG.get("API", {}).get("LIVE_TRADING", False),
        "use_prev_weights": CONFIG.get("USE_PREV_WEIGHTS", True),
        "threads": CONFIG.get("CPU_LIMIT", os.cpu_count() or 1),
        "use_net_term": DEFAULT_CFG.get("reward", {}).get(
            "use_net_term", G.use_net_term
        ),
        "use_sharpe_term": DEFAULT_CFG.get("reward", {}).get(
            "use_sharpe_term", G.use_sharpe_term
        ),
        "use_drawdown_term": DEFAULT_CFG.get("reward", {}).get(
            "use_drawdown_term", G.use_drawdown_term
        ),
        "use_trade_term": DEFAULT_CFG.get("reward", {}).get(
            "use_trade_term", G.use_trade_term
        ),
        "use_profit_days_term": DEFAULT_CFG.get("reward", {}).get(
            "use_profit_days_term", G.use_profit_days_term
        ),
        "use_sortino_term": DEFAULT_CFG.get("reward", {}).get(
            "use_sortino_term", G.use_sortino_term
        ),
        "use_omega_term": DEFAULT_CFG.get("reward", {}).get(
            "use_omega_term", G.use_omega_term
        ),
        "use_calmar_term": DEFAULT_CFG.get("reward", {}).get(
            "use_calmar_term", G.use_calmar_term
        ),
        "theta": DEFAULT_CFG.get("reward", {}).get("theta", G.theta),
        "phi": DEFAULT_CFG.get("reward", {}).get("phi", G.phi),
        "chi": DEFAULT_CFG.get("reward", {}).get("chi", G.chi),
        "beta": DEFAULT_CFG.get("reward", {}).get("beta", G.beta),
        "warmup_steps": DEFAULT_CFG.get("WARMUP_STEPS", G.warmup_steps),
        "risk_filter": G.is_risk_filter_enabled(),
    }
    opts = startup_options_dialog(defaults)
    global SKIP_SENTIMENT
    SKIP_SENTIMENT = bool(opts.get("skip_sentiment", False))
    if SKIP_SENTIMENT:
        os.environ["NO_HEAVY"] = "1"
    else:
        os.environ.pop("NO_HEAVY", None)

    set_threads(int(opts.get("threads", defaults["threads"])))
    ensure_dependencies()

    G.use_net_term = bool(opts.get("use_net_term", defaults["use_net_term"]))
    G.use_sharpe_term = bool(opts.get("use_sharpe_term", defaults["use_sharpe_term"]))
    G.use_drawdown_term = bool(
        opts.get("use_drawdown_term", defaults["use_drawdown_term"])
    )
    G.use_trade_term = bool(opts.get("use_trade_term", defaults["use_trade_term"]))
    G.use_profit_days_term = bool(
        opts.get("use_profit_days_term", defaults["use_profit_days_term"])
    )
    G.use_sortino_term = bool(
        opts.get("use_sortino_term", defaults["use_sortino_term"])
    )
    G.use_omega_term = bool(opts.get("use_omega_term", defaults["use_omega_term"]))
    G.use_calmar_term = bool(opts.get("use_calmar_term", defaults["use_calmar_term"]))
    G.theta = float(opts.get("theta", defaults["theta"]))
    G.phi = float(opts.get("phi", defaults["phi"]))
    G.chi = float(opts.get("chi", defaults["chi"]))
    G.beta = float(opts.get("beta", defaults["beta"]))
    from artibot.hyperparams import _CONFIG

    _CONFIG["WARMUP_STEPS"] = int(opts.get("warmup_steps", defaults["warmup_steps"]))
    G.warmup_steps = _CONFIG["WARMUP_STEPS"]
    G.set_risk_filter_enabled(bool(opts.get("risk_filter", defaults["risk_filter"])))

    from artibot.utils import setup_logging
    from artibot.core.device import get_device
    from artibot.dataset import HourlyDataset, load_csv_hourly
    from artibot.hyperparams import IndicatorHyperparams
    from artibot.training import (
        PhemexConnector,
        csv_training_thread,
        phemex_live_thread,
    )
    from artibot.rl import MetaTransformerRL, meta_control_loop
    from artibot.validation import validate_and_gate
    from artibot.gui import TradingGUI

    setup_logging()
    from artibot.utils import heartbeat

    hb_interval = DEFAULT_CFG.get("logging", {}).get("heartbeat_interval", 120)
    heartbeat.start(interval=hb_interval)
    root = tk.Tk()
    gui: TradingGUI = TradingGUI(root, None, None, None, dev=dev_mode)
    G.set_status("DEFCON 5: Hyperparameter Search", "initializing")

    G.set_cpu_limit(int(opts.get("threads", defaults["threads"])))
    G.start_equity = 0.0
    G.live_equity = 0.0
    G.live_trade_count = 0

    use_live = bool(opts.get("use_live", defaults["use_live"]))
    CONFIG.setdefault("API", {})["LIVE_TRADING"] = use_live
    G.use_sandbox = not use_live
    mode = "LIVE" if use_live else "SANDBOX"
    logging.info("Trading mode: %s", mode)

    connector = PhemexConnector(CONFIG)
    gui.connector = connector
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

    stop_event = threading.Event()
    train_th: threading.Thread | None = None
    live_feed_th: threading.Thread | None = None
    meta_th: threading.Thread | None = None
    validate_th: threading.Thread | None = None

    def setup_worker() -> None:
        nonlocal train_th, live_feed_th, meta_th, validate_th
        from artibot.training import run_hpo

        device = get_device()

        csv_path = CONFIG["CSV_PATH"]
        if not os.path.isabs(csv_path):
            here = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(here, csv_path)

        data = load_csv_hourly(csv_path, cfg=DEFAULT_CFG)
        if not data:
            logging.error("No usable CSV data found")
            G.set_status("Init", "CSV data missing")
            return

        if no_tune:
            G.set_status("DEFCON 5: Hyperparameter Search", "skipped")
            best = {}
            indicator_hp = IndicatorHyperparams()
        else:
            G.set_status("DEFCON 5: Hyperparameter Search", "")
            best = run_hpo()

        param_map = {f.name.upper(): f.name for f in fields(IndicatorHyperparams)}
        ind_params = {param_map[k]: v for k, v in best.items() if k in param_map}
        indicator_hp = IndicatorHyperparams(**ind_params)

        temp_ds = HourlyDataset(
            data,
            seq_len=24,
            indicator_hparams=indicator_hp,
            atr_threshold_k=getattr(indicator_hp, "atr_threshold_k", 1.5),
            train_mode=False,
        )
        n_features = temp_ds[0][0].shape[1]

        lr = best.get("lr", 1e-3)
        entropy_beta = best.get("entropy_beta")
        ensemble = build_model(
            device=device,
            n_features=n_features,
            lr=lr,
            entropy_beta=entropy_beta,
            warmup_steps=int(opts.get("warmup_steps", defaults["warmup_steps"])),
            indicator_hp=indicator_hp,
            freeze_features=True,
        )

        weights_dir = os.path.abspath(
            os.path.expanduser(CONFIG.get("WEIGHTS_DIR", "weights"))
        )
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "best.pt")
        use_prev_weights = bool(
            opts.get("use_prev_weights", defaults["use_prev_weights"])
        )
        if os.path.isfile(weights_path) and use_prev_weights:
            ensemble.load_best_weights(weights_path)

        root.after(0, lambda: gui.set_ensemble(ensemble, weights_path))

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
            kwargs={"debug_anomaly": dev_mode, "init_event": init_done},
            daemon=True,
        )
        train_th.start()
        G.set_status("DEFCON 4: Training in Progress", "")

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
            target=lambda: validate_and_gate(
                csv_path, CONFIG, indicator_hp=indicator_hp
            ),
            daemon=True,
        )
        validate_th.start()

        import artibot.feature_ingest as _fi

        ingest_th = threading.Thread(target=_fi.start_worker, daemon=True)
        ingest_th.start()

        def _bg_init() -> None:

            if SKIP_SENTIMENT:
                init_done.set()
                G.set_status("Init", "Skipping sentiment pull")
                return

            G.set_status("Init", "Downloading sentiment data")
            try:
                import tools.backfill_gdelt as _bf

                _bf.main(progress_cb=lambda pct, msg: None)
            finally:
                init_done.set()

        threading.Thread(target=_bg_init, daemon=True).start()

    init_th = threading.Thread(target=setup_worker, daemon=True)
    init_th.start()

    logging.info("Tkinter dashboard starting on main thread…")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover - unexpected GUI error
        logging.exception("Tkinter mainloop failed: %s", exc)

    stop_event.set()
    if init_th.is_alive():
        init_th.join()
    for th in (train_th, live_feed_th, meta_th, validate_th):
        if th is not None:
            th.join()


if __name__ == "__main__":
    main()
