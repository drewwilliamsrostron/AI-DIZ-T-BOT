# Threaded training loop and exchange polling utilities.
# Handles CSV backtesting and live data fetching.
"""Background CSV training thread and exchange connector."""

# ruff: noqa: F403, F405
import artibot.globals as G
import artibot.globals as globals
import artibot.hyperparams as hyperparams

import logging
from torch.utils.tensorboard import SummaryWriter
import optuna
import pandas as pd
import datetime
import time
import threading
import os
from pathlib import Path

from .dataset import HourlyDataset, trailing_sma, load_csv_hourly
from .ensemble import reject_if_risky, EnsembleModel
from .backtest import robust_backtest, compute_indicators
from .core.device import get_device
from .utils import heartbeat
from .feature_manager import enforce_feature_dim
from artibot.hyperparams import RISK_FILTER
from artibot.utils.reward_utils import ema, differential_sharpe

import sys
import json
import torch
import multiprocessing
import gc


def quick_fit(model: EnsembleModel, data: list[list[float]], epochs: int = 1) -> None:
    """Train ``model`` on ``data`` for a small number of epochs."""

    ds = HourlyDataset(
        data,
        seq_len=24,
        indicator_hparams=model.indicator_hparams,
        atr_threshold_k=getattr(model.indicator_hparams, "atr_threshold_k", 1.5),
        train_mode=True,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train_one_epoch(dl, None, data, update_globals=False)


logger = logging.getLogger(__name__)

CPU_LIMIT_DEFAULT = max(1, multiprocessing.cpu_count() - 2)
with G.state_lock:
    G.cpu_limit = CPU_LIMIT_DEFAULT
torch.set_num_threads(CPU_LIMIT_DEFAULT)
os.environ["OMP_NUM_THREADS"] = str(CPU_LIMIT_DEFAULT)

# TensorBoard writer for real-time metrics
writer = SummaryWriter(log_dir="runs/experiment1")


CHECKPOINTS_DIR = Path("models/checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def save_epoch_checkpoint(model: torch.nn.Module, epoch: int) -> None:
    """Save ``model`` state and keep only the last 5 checkpoints."""

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / f"epoch_{epoch}.pt"
    torch.save(model.state_dict(), path)
    files = sorted(CHECKPOINTS_DIR.glob("epoch_*.pt"))
    if len(files) > 5:
        for f in files[:-5]:
            try:
                f.unlink()
            except OSError:
                logging.warning("Failed to remove %s", f)


def smooth(series: list[float], span: int = 10) -> list[float]:
    """Return exponentially smoothed values for ``series``."""

    if not series:
        return []
    return pd.Series(series).ewm(span=span).mean().tolist()


def rebuild_loader(
    old_loader: torch.utils.data.DataLoader | None,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 512,
    shuffle: bool = True,
    *,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    """Dispose of ``old_loader`` and return a new ``DataLoader``."""

    # Gracefully stop and free all worker processes
    if old_loader is not None:
        if getattr(old_loader, "_iterator", None) is not None:
            old_loader._iterator._shutdown_workers()
        del old_loader
        gc.collect()

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return torch.utils.data.DataLoader(**loader_kwargs)


def profile_data_copy(
    loader: torch.utils.data.DataLoader, device: torch.device, steps: int = 5
) -> None:
    """Profile CPU to GPU copy times for ``loader``."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3)
    with torch.profiler.profile(activities=activities, schedule=schedule) as prof:
        for step, batch in enumerate(loader):
            for t in batch:
                t.to(device, non_blocking=True)
            prof.step()
            if step >= steps:
                break
    logging.info("\n%s", prof.key_averages().table(sort_by="self_cuda_time_total"))


def apply_risk_curriculum(epoch: int) -> None:
    """Adjust ``RISK_FILTER`` thresholds as training progresses."""
    if epoch <= 20:
        RISK_FILTER["MIN_REWARD"] = -2.0
        RISK_FILTER["MAX_DRAWDOWN"] = -0.80
    elif epoch <= 40:
        RISK_FILTER["MIN_REWARD"] = -1.0
        RISK_FILTER["MAX_DRAWDOWN"] = -0.50
    else:
        RISK_FILTER["MIN_REWARD"] = 0.0
        RISK_FILTER["MAX_DRAWDOWN"] = -0.25


###############################################################################
def csv_training_thread(
    ensemble,
    data,
    stop_event,
    config,
    use_prev_weights=True,
    max_epochs: int | None = None,
    weights_path: str = "best.pt",
    *,
    debug_anomaly: bool = False,
    init_event: threading.Event | None = None,
    update_globals: bool = True,
    overfit_toy: bool = False,
):
    """Train on CSV data in a background thread.

    ``max_epochs`` stops the loop after N iterations when set.  When
    ``debug_anomaly`` is ``True`` PyTorch's autograd anomaly detection is enabled.

    TODO: add granular ``set_status`` calls for each major step and
    integrate more gating checks.
    ``update_globals`` controls whether :mod:`artibot.globals` is mutated.
    """
    import traceback

    import numpy as np
    from torch.utils.data import random_split

    if debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # training history lists live on the globals module

    try:
        if init_event is not None:
            init_event.wait()
        G.epoch_count = 0
        if overfit_toy:
            data = data[:100]
            max_epochs = 50
        train_data: list[list[float]] = []
        for row in data:
            train_data.append([float(v) for v in row])

        # Perform walk-forward validation to establish holdout metrics
        one_month = 24 * 30
        walk_results = walk_forward_backtest(train_data, 12 * one_month, one_month)
        if walk_results:
            mean_sharpe = float(np.mean([r.get("sharpe", 0.0) for r in walk_results]))
            mean_dd = float(np.mean([r.get("max_drawdown", 0.0) for r in walk_results]))
            G.global_holdout_sharpe = mean_sharpe
            G.global_holdout_max_drawdown = mean_dd
        else:
            G.global_holdout_sharpe = 0.0
            G.global_holdout_max_drawdown = 0.0

        ds_full = HourlyDataset(
            train_data,
            seq_len=24,
            indicator_hparams=ensemble.indicator_hparams,
            atr_threshold_k=getattr(ensemble.indicator_hparams, "atr_threshold_k", 1.5),
            train_mode=True,
        )
        if use_prev_weights:
            ensemble.load_best_weights(weights_path, data_full=train_data)

        n_total = len(ds_full)
        if n_total >= 10:
            n_train = int(n_total * 0.9)
            n_val = n_total - n_train
            ds_train, ds_val = random_split(ds_full, [n_train, n_val])
        else:
            logging.warning(
                "Dataset too small for validation split (%d samples)", n_total
            )
            ds_train = ds_full
            ds_val = None

        train_indicators = compute_indicators(
            train_data,
            ensemble.indicator_hparams,
            with_scaled=True,
        )

        workers = int(config.get("NUM_WORKERS", 0))
        logging.info(
            "DATALOADER", extra={"workers": workers, "device": ensemble.device.type}
        )
        dl_train = rebuild_loader(
            None, ds_train, batch_size=512, shuffle=True, num_workers=workers
        )
        dl_val = (
            rebuild_loader(
                None, ds_val, batch_size=512, shuffle=False, num_workers=workers
            )
            if ds_val is not None
            else None
        )
        if config.get("PROFILE", False):
            profile_data_copy(dl_train, ensemble.device)
        steps_per_epoch = len(dl_train)
        if max_epochs is not None:
            total_steps = steps_per_epoch * max_epochs
        else:
            total_steps = ensemble.total_steps
        ensemble.configure_one_cycle(total_steps)

        adapt_live = bool(config.get("ADAPT_TO_LIVE", False))
        dummy_input = torch.randn(
            1, 24, ensemble.models[0].input_dim, device=ensemble.device
        )
        ensemble.optimize_models(dummy_input)
        lr_base = ensemble.optimizers[0].param_groups[0]["lr"]
        schedulers = []
        one_cycle = getattr(torch.optim.lr_scheduler, "OneCycleLR", None)
        if callable(one_cycle) and getattr(one_cycle, "__name__", "") == "OneCycleLR":
            schedulers = [
                one_cycle(
                    opt,
                    max_lr=lr_base * 10,
                    total_steps=max_epochs or 30,
                )
                for opt in ensemble.optimizers
            ]

        trade_count_series: list[float] = []
        days_in_profit_series: list[float] = []
        returns_series: list[float] = []

        import talib

        epochs = 0
        best_reward = float("-inf")
        no_gain = 0
        final_loss = None
        while not stop_event.is_set():
            if not G.is_bot_running():
                time.sleep(1.0)
                continue
            if max_epochs is not None and epochs >= max_epochs:
                break
            ensemble.train_steps += 1
            epochs += 1
            apply_risk_curriculum(ensemble.train_steps)

            progress = int(100 * ensemble.train_steps / max_epochs) if max_epochs else 0
            G.global_progress_pct = progress

            logging.info(
                "START_EPOCH",
                extra={"epoch": ensemble.train_steps},
            )

            logging.debug(json.dumps({"event": "status", "msg": G.get_status()}))
            tl, vl = ensemble.train_one_epoch(
                dl_train,
                dl_val,
                train_data,
                stop_event,
                features=train_indicators,
                update_globals=update_globals,
            )
            final_loss = tl

            if ensemble.train_steps == 1:
                train_indicators = compute_indicators(
                    train_data,
                    ensemble.indicator_hparams,
                    with_scaled=True,
                )
                ds_train = HourlyDataset(
                    train_data,
                    seq_len=24,
                    indicator_hparams=ensemble.indicator_hparams,
                    atr_threshold_k=getattr(
                        ensemble.indicator_hparams, "atr_threshold_k", 1.5
                    ),
                    train_mode=True,
                )
                dl_train = rebuild_loader(
                    dl_train,
                    ds_train,
                    batch_size=512,
                    shuffle=True,
                    num_workers=workers,
                )

            status_msg = (
                f"Epoch {ensemble.train_steps}/{max_epochs} – loss {tl:.4f}"
                if max_epochs
                else f"Epoch {ensemble.train_steps} – loss {tl:.4f}"
            )
            G.set_status("Training", status_msg)

            if walk_results:
                mean_sharpe = float(
                    np.mean([r.get("sharpe", 0.0) for r in walk_results])
                )
                mean_dd = float(
                    np.mean([r.get("max_drawdown", 0.0) for r in walk_results])
                )
                G.global_holdout_sharpe = mean_sharpe
                G.global_holdout_max_drawdown = mean_dd
            else:
                G.global_holdout_sharpe = 0.0
                G.global_holdout_max_drawdown = 0.0
            G.global_training_loss.append(tl)
            if vl is not None:
                G.global_validation_loss.append(vl)
            else:
                G.global_validation_loss.append(None)
            smoothed_loss = smooth(G.global_training_loss)
            eq_vals = [b for _, b in G.global_equity_curve]
            smoothed_equity = smooth(eq_vals)
            if smoothed_loss:
                logger.info("Smoothed Loss: %.4f", smoothed_loss[-1])
            if smoothed_equity:
                logger.info("Smoothed Equity: %.4f", smoothed_equity[-1])
            if G.global_backtest_profit:
                _ = G.global_backtest_profit[-1]
            last_reward = (
                G.global_composite_reward if G.global_composite_reward else 0.0
            )
            G.global_composite_reward_ema = (
                0.9 * (G.global_composite_reward_ema or 0.0) + 0.1 * last_reward
            )

            trade_count_series.append(float(G.global_num_trades))
            days_in_profit_series.append(float(G.global_days_in_profit))
            ret_val = G.global_backtest_profit[-1] if G.global_backtest_profit else 0.0
            returns_series.append(float(ret_val))

            trade_term = ema(torch.tensor(trade_count_series), tau=96.0)[-1]
            days_term = ema(torch.tensor(days_in_profit_series), tau=96.0)[-1]
            sharpe_term = differential_sharpe(torch.tensor(returns_series))
            ensemble.reward_loss_weight = min(
                ensemble.max_reward_loss_weight,
                ensemble.reward_loss_weight + 0.01,
            )
            if ensemble.cycle:
                lr_now = ensemble.cycle[0].get_last_lr()[0]
            else:
                lr_now = ensemble.optimizers[0].param_groups[0]["lr"]
            attn_mean = (
                float(np.mean(G.global_attention_weights_history[-100:]))
                if G.global_attention_weights_history
                else 0.0
            )
            attn_entropy = (
                float(np.mean(G.global_attention_entropy_history[-100:]))
                if G.global_attention_entropy_history
                else 0.0
            )
            log_obj = {
                "epoch": ensemble.train_steps,
                "loss": tl,
                "val": None if vl is None else vl,
                "reward": last_reward,
                "best_reward": best_reward,
                "max_dd": G.global_max_drawdown,
                "net_pct": G.global_net_pct,
                "holdout_dd": G.global_holdout_max_drawdown,
                "attn": attn_mean,
                "attn_entropy": attn_entropy,
                "lr": lr_now,
                "profit_factor": G.global_profit_factor,
                "trade_term": trade_term,
                "days_term": days_term,
                "sharpe_term": sharpe_term,
            }
            logging.info(
                "EPOCH_METRICS",
                extra=log_obj,
            )

            # Trace reward against loss and entropy for debugging
            logging.info(
                "REWARD_TRACE",
                extra={
                    "epoch": ensemble.train_steps,
                    "reward": last_reward,
                    "loss": tl,
                    "entropy": attn_entropy,
                },
            )

            from artibot.hyperparams import RISK_FILTER, WARMUP_STEPS

            if G.get_warmup_step() >= WARMUP_STEPS:
                RISK_FILTER["MIN_REWARD"] = 0.5
                RISK_FILTER["MAX_DRAWDOWN"] = -0.30

            if globals.get_trade_count() >= 1000 and not globals.PROD_RISK:
                hyperparams.RISK_FILTER.update(
                    {"MIN_REWARD": 0.5, "MAX_DRAWDOWN": -0.30}
                )
                globals.PROD_RISK = True

            reward_val = last_reward
            max_dd = G.global_max_drawdown
            entropy = attn_entropy
            if overfit_toy:
                print(f"Toy epoch {epochs} loss {tl:.4f}")
            skip_risk_check = ensemble.train_steps < 3
            if (
                not overfit_toy
                and not skip_risk_check
                and reject_if_risky(reward_val, max_dd, entropy)
            ):
                logging.info(
                    "REJECTED",
                    extra={
                        "epoch": ensemble.train_steps,
                        "reward": reward_val,
                        "max_dd": max_dd,
                        "attn_entropy": entropy,
                    },
                )
                G.set_status("Risk", "Epoch rejected")
                G.inc_epoch()
                continue

            if last_reward > best_reward:
                best_reward = last_reward
                no_gain = 0
            else:
                no_gain += 1
            if not overfit_toy and no_gain >= 10:
                logging.info(
                    "EARLY_STOP",
                    extra={"epoch": ensemble.train_steps},
                )
                break

            # quick "latest prediction"
            if len(train_data) >= 24:
                tail = np.array(train_data[-24:], dtype=np.float64)
                closes = tail[:, 4]
                sma = trailing_sma(closes, 10)
                rsi = talib.RSI(closes, timeperiod=14)
                macd, _, _ = talib.MACD(closes)
                ext = []
                for i, row in enumerate(tail):
                    ext.append(
                        [
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            float(sma[i]),
                            float(rsi[i]),
                            float(macd[i]),
                        ]
                    )
                ext = np.array(ext, dtype=np.float32)
                # Pad preview tensor to match model input width
                ext = enforce_feature_dim(ext, expected=16)
                seq_t = (
                    torch.tensor(ext, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(ensemble.device)
                )
                idx, conf, _ = ensemble.predict(seq_t)
                label_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
                G.global_current_prediction = label_map.get(idx, "N/A")
                G.global_ai_confidence = conf
                G.inc_epoch()
                G.global_attention_weights_history.append(0)
                G.global_attention_entropy_history.append(0)

            if adapt_live:
                changed = False
                while not G.live_bars_queue.empty():
                    new_b = G.live_bars_queue.get()
                    for bar in new_b:
                        ts, o_, h_, l_, c_, v_ = bar

                        ts = int(ts)
                        if ts > 1_000_000_000_000:
                            ts //= 1000

                        o_ = float(o_)
                        h_ = float(h_)
                        l_ = float(l_)
                        c_ = float(c_)
                        v_ = float(v_)
                        if ts > train_data[-1][0]:
                            train_data.append([ts, o_, h_, l_, c_, v_])
                            changed = True
                if changed:

                    G.set_status("Training", "Adapting to live data")

                    ds_updated = HourlyDataset(
                        train_data,
                        seq_len=24,
                        indicator_hparams=ensemble.indicator_hparams,
                        atr_threshold_k=getattr(
                            ensemble.indicator_hparams, "atr_threshold_k", 1.5
                        ),
                        train_mode=True,
                    )
                    nt_ = len(ds_updated)
                    if nt_ >= 10:
                        ntr_ = int(nt_ * 0.9)
                        nv_ = nt_ - ntr_
                        ds_tr_, ds_val_ = random_split(ds_updated, [ntr_, nv_])
                    else:
                        logging.warning(
                            "Dataset too small for validation split during live adapt (%d samples)",
                            nt_,
                        )
                        ds_tr_ = ds_updated
                        ds_val_ = None
                    logging.info(
                        "DATALOADER",
                        extra={"workers": workers, "device": ensemble.device.type},
                    )
                    dl_train = rebuild_loader(
                        dl_train,
                        ds_tr_,
                        batch_size=512,
                        shuffle=True,
                        num_workers=workers,
                    )
                    dl_val = (
                        rebuild_loader(
                            dl_val,
                            ds_val_,
                            batch_size=512,
                            shuffle=False,
                            num_workers=workers,
                        )
                        if ds_val_ is not None
                        else dl_val
                    )
                    train_indicators = compute_indicators(
                        train_data,
                        ensemble.indicator_hparams,
                        with_scaled=True,
                    )
                    tl, vl = ensemble.train_one_epoch(
                        dl_train,
                        dl_val,
                        train_data,
                        stop_event,
                        features=train_indicators,
                        update_globals=update_globals,
                    )
                    G.global_training_loss.append(tl)
                    G.global_validation_loss.append(vl)
                    writer.add_scalar("Loss/train", tl, ensemble.train_steps)
                    if vl is not None:
                        writer.add_scalar("Loss/val", vl, ensemble.train_steps)

            equity = G.global_equity_curve[-1][1] if G.global_equity_curve else 0.0
            logging.info(
                "EPOCH",
                extra={
                    "epoch": ensemble.train_steps,
                    "loss": tl,
                    "net_pct": G.global_net_pct,
                    "lr": lr_now,
                    "equity": equity,
                    "reward": G.global_composite_reward,
                    "entropy": attn_entropy,
                },
            )
            writer.add_scalar("Loss/train", tl, ensemble.train_steps)
            if vl is not None:
                writer.add_scalar("Loss/val", vl, ensemble.train_steps)
            writer.add_scalar("LR", lr_now, ensemble.train_steps)
            writer.add_scalar("Equity", equity, ensemble.train_steps)
            heartbeat.update(
                epoch=ensemble.train_steps,
                best_reward=best_reward,
            )

            if ensemble.train_steps % 5 == 0 and ensemble.best_state_dicts:
                ensemble.save_best_weights(weights_path)

            save_epoch_checkpoint(ensemble, ensemble.train_steps)
            for sch in schedulers:
                if not hasattr(sch, "step"):
                    continue
                try:
                    if getattr(sch, "step_num", 0) < getattr(sch, "total_steps", 0):
                        sch.step()
                        sch.step_num = getattr(sch, "step_num", 0) + 1
                except ValueError as e:
                    logging.warning("LR scheduler skipped: %s", e)

        if overfit_toy:
            if final_loss is None or final_loss >= 0.3:
                raise AssertionError(
                    f"Final loss {final_loss:.4f} did not drop below 0.3"
                )

    except Exception as e:
        traceback.print_exc()

        G.set_status(f"Training error: {e}", "")

        stop_event.set()


def phemex_live_thread(
    connector,
    stop_event,
    poll_interval: float,
    ensemble=None,
    live_weight_path: str = "live_model.pt",
) -> None:
    """Continuously fetch recent bars from Phemex at a configurable interval."""
    import traceback

    while not stop_event.is_set():
        if not G.is_bot_running():
            time.sleep(1.0)
            continue
        try:

            bars = connector.fetch_latest_bars(limit=100)
            if bars:
                G.global_phemex_data = bars

                G.live_bars_queue.put(bars)

            G.set_status(
                "Fetching",
                f"{len(bars)} bars received at {datetime.datetime.now():%H:%M:%S}",
            )

        except Exception as e:
            traceback.print_exc()

            G.set_status(f"Fetch error: {e}", "")
            stop_event.set()

        if ensemble is not None and G.pop_live_weights_updated():
            try:
                ensemble.load_best_weights(live_weight_path)
                logging.info("LIVE_WEIGHTS_LOADED")
            except Exception as exc:
                logging.error("Live weight load failed: %s", exc)

        time.sleep(max(0, 3600 - (time.time() % 3600)))


###############################################################################
# Connector
###############################################################################
class PhemexConnector:
    """Thin wrapper around ccxt for polling Phemex."""

    def __init__(self, config):
        """Create a connector using API credentials from ``config``."""
        # Symbol is currently fixed for simplicity
        self.symbol = "BTCUSD"
        api_conf = config.get("API", {})
        self.live_trading = bool(api_conf.get("LIVE_TRADING", True))

        key = os.getenv(
            "PHEMEX_KEY",
            (
                api_conf.get("API_KEY_LIVE", "")
                if self.live_trading
                else api_conf.get("API_KEY_TEST", "")
            ),
        )
        secret = os.getenv(
            "PHEMEX_SECRET",
            (
                api_conf.get("API_SECRET_LIVE", "")
                if self.live_trading
                else api_conf.get("API_SECRET_TEST", "")
            ),
        )

        api_url_live = api_conf.get("API_URL_LIVE", "https://api.phemex.com")
        api_url_test = api_conf.get("API_URL_TEST", "https://testnet-api.phemex.com")
        self.default_type = api_conf.get("DEFAULT_TYPE", "swap")

        self._last_code = None

        import ccxt

        try:
            self.exchange = ccxt.phemex(
                {
                    "urls": {"api": {"live": api_url_live, "test": api_url_test}},
                    "apiKey": key,
                    "secret": secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": self.default_type},
                }
            )
            self.exchange.set_sandbox_mode(not self.live_trading)
        except Exception as exc:
            logging.error("Error initializing exchange: %s", exc)
            sys.exit(1)

        self.exchange.load_markets()

    def _handle_err(self, exc: Exception) -> bool:
        """Return ``True`` when ``exc`` is a maintenance message."""
        msg = str(exc)
        if "39998" in msg:
            if self._last_code != 39998:
                logging.warning("Exchange maintenance")
                self._last_code = 39998
            G.set_status("Exchange maintenance", "")
            return True
        self._last_code = None
        return False

    def get_account_stats(self) -> dict:
        """Return account balances including USDT equity."""
        from .utils.account import get_account_equity

        try:
            bal = self.exchange.fetch_balance()
            equity = get_account_equity(self.exchange)
        except Exception as exc:  # pragma: no cover - network errors
            logging.error("Balance fetch error: %s", exc)
            return {"total": {}}

        totals = dict(bal.get("total", {}))
        totals["USDT"] = equity
        return {"total": totals}

    def fetch_latest_bars(self, limit=100):
        """Return the most recent OHLCV bars from Phemex."""
        logging.debug(
            "fetch_ohlcv -> %s tf=%s limit=%s",
            self.symbol,
            "1h",
            limit,
        )
        try:
            bars = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe="1h",
                limit=limit,
            )
        except Exception as exc:
            if self._handle_err(exc):
                return []
            logging.error(
                "fetch_ohlcv failed for %s tf=%s limit=%s: %s",
                self.symbol,
                "1h",
                limit,
                exc,
            )
            return []
        logging.debug("fetched %d bars", len(bars) if bars else 0)
        return bars if bars else []

    def create_order(
        self,
        side: str,
        amount: float,
        price: float,
        order_type: str = "market",
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ):
        """Submit a market order with :mod:`ccxt` applying slippage."""
        from .execution import submit_order

        params = {"type": self.default_type}
        if stop_loss is not None:
            params["stopLossPrice"] = stop_loss
        if take_profit is not None:
            params["takeProfitPrice"] = take_profit

        def _place(**kwargs):
            try:
                return self.exchange.create_order(
                    self.symbol,
                    order_type,
                    side,
                    kwargs["amount"],
                    kwargs["price"],
                    params,
                )
            except Exception as exc:
                if self._handle_err(exc):
                    return None
                logging.error("Order error: %s", exc)
                return None

        return submit_order(_place, side, amount, price)


###############################################################################
# Checkpoint
###############################################################################
def save_checkpoint():
    """Write training progress and hyperparams to ``checkpoint.json``."""
    import json

    checkpoint = {
        "global_training_loss": G.global_training_loss,
        "global_validation_loss": G.global_validation_loss,
        "global_backtest_profit": G.global_backtest_profit,
        "global_equity_curve": G.global_equity_curve,
        "global_ai_adjustments_log": G.global_ai_adjustments_log,
        "global_hyperparameters": {
            "GLOBAL_THRESHOLD": G.GLOBAL_THRESHOLD,
            "global_SL_multiplier": G.global_SL_multiplier,
            "global_TP_multiplier": G.global_TP_multiplier,
            "global_ATR_period": G.global_ATR_period,
        },
        "global_ai_epoch_count": G.epoch_count,
        "gpt_memory_squirtle": G.gpt_memory_squirtle,
        "gpt_memory_wartorttle": G.gpt_memory_wartorttle,
        "gpt_memory_bigmanblastoise": G.gpt_memory_bigmanblastoise,
        "gpt_memory_moneymaker": G.gpt_memory_moneymaker,
        "global_attention_weights_history": G.global_attention_weights_history,
    }
    with open("checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)


###############################################################################
# Hyper-parameter optimisation utilities
###############################################################################


def objective(trial: optuna.trial.Trial) -> float:
    """Objective for Optuna optimisation."""

    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "entropy_beta": trial.suggest_float("entropy_beta", 1e-4, 5e-3, log=True),
    }
    data = load_csv_hourly("Gemini_BTCUSD_1h.csv")
    if not data:
        return 0.0
    ds_tmp = HourlyDataset(data, seq_len=24, train_mode=True)
    n_features = ds_tmp[0][0].shape[1]
    model = EnsembleModel(
        device=get_device(), n_models=1, lr=params["lr"], n_features=n_features
    )
    model.entropy_beta = params["entropy_beta"]
    stop = threading.Event()
    csv_training_thread(
        model,
        data,
        stop,
        {"ADAPT_TO_LIVE": False},
        use_prev_weights=False,
        max_epochs=10,
        update_globals=False,
    )
    metrics = robust_backtest(model, data)
    if metrics.get("trades", 0) == 0:
        logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
    else:
        G.push_backtest_metrics(metrics)
    return -metrics.get("composite_reward", 0.0)


def run_hpo(n_trials: int = 50) -> dict:
    """Run Bayesian hyper-parameter search with Optuna."""

    logging.info(">>> ENTERING DEFCON 5: Hyperparameter Search")
    logging.info(">>> Starting Sweep: 0 of %d", n_trials)
    G.set_status("DEFCON 5: Hyperparameter Search", "starting")
    study = optuna.create_study(direction="minimize")
    for idx in range(1, n_trials + 1):
        G.set_status("DEFCON 5: Hyperparameter Search", f"Trial {idx}/{n_trials}")
        study.optimize(objective, n_trials=1, timeout=3600)
        params = study.trials[-1].params if study.trials else {}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        logging.info(
            "--- Hyperparam Set %d/%d --- Indicator combo: %s", idx, n_trials, param_str
        )
    best = study.best_params
    G.global_best_lr = best.get("lr")
    G.global_best_wd = best.get("entropy_beta")
    return best


def walk_forward_backtest(data: list, train_window: int, test_horizon: int) -> list:
    """Perform walk-forward validation across ``data``."""

    logging.info(">>> ENTERING DEFCON 4: Walk Forward Evaluation")
    results: list = []
    n_folds = max(1, (len(data) - train_window - test_horizon) // test_horizon + 1)
    fold_idx = 1
    for start in range(0, len(data) - train_window - test_horizon, test_horizon):
        end = start + train_window + test_horizon
        start_dt = pd.to_datetime(data[start][0], unit="s").strftime("%Y-%m")
        end_dt = pd.to_datetime(data[end - 1][0], unit="s").strftime("%Y-%m")
        logging.info(
            "Fold %d of %d - Period: %s to %s",
            fold_idx,
            n_folds,
            start_dt,
            end_dt,
        )
        fold_idx += 1
        train_slice = data[start : start + train_window]
        test_slice = data[start + train_window : start + train_window + test_horizon]
        model = EnsembleModel(device=get_device(), n_models=1)
        quick_fit(model, train_slice, epochs=1)
        metrics = robust_backtest(model, test_slice)
        if metrics.get("trades", 0) == 0:
            logging.info("IGNORED_EMPTY_BACKTEST: 0 trades in result")
        else:
            G.push_backtest_metrics(metrics)
        results.append(metrics)
    return results
