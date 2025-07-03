"""Background CSV training thread and exchange connector."""

# ruff: noqa: F403, F405
import artibot.globals as G
import artibot.globals as globals
import artibot.hyperparams as hyperparams

import logging
import datetime
import time
import threading
import os

from .dataset import HourlyDataset, trailing_sma
from .ensemble import reject_if_risky
from .backtest import robust_backtest, compute_indicators
from .feature_manager import enforce_feature_dim
from artibot.hyperparams import RISK_FILTER

import sys
import json
import torch
import multiprocessing
import gc

CPU_LIMIT_DEFAULT = max(1, multiprocessing.cpu_count() - 2)
with G.state_lock:
    G.cpu_limit = CPU_LIMIT_DEFAULT
torch.set_num_threads(CPU_LIMIT_DEFAULT)
os.environ["OMP_NUM_THREADS"] = str(CPU_LIMIT_DEFAULT)


def rebuild_loader(
    old_loader: torch.utils.data.DataLoader | None,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 128,
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

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=False,
    )


def apply_risk_curriculum(epoch: int) -> None:
    """Adjust ``RISK_FILTER`` thresholds as training progresses."""
    if epoch <= 20:
        RISK_FILTER["MIN_SHARPE"] = -2.0
        RISK_FILTER["MAX_DRAWDOWN"] = -0.80
    elif epoch <= 40:
        RISK_FILTER["MIN_SHARPE"] = -1.0
        RISK_FILTER["MAX_DRAWDOWN"] = -0.50
    else:
        RISK_FILTER["MIN_SHARPE"] = 0.0
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
        holdout_len = max(1, int(len(data) * 0.1))
        base = data[:-holdout_len] if len(data) > holdout_len else data
        train_data: list[list[float]] = []
        for row in base:
            train_data.append([float(v) for v in row])
        holdout_data = data[-holdout_len:] if len(data) > holdout_len else []

        ds_full = HourlyDataset(
            train_data,
            seq_len=24,
            indicator_hparams=ensemble.indicator_hparams,
            atr_threshold_k=getattr(ensemble.indicator_hparams, "atr_threshold_k", 1.5),
            train_mode=True,
        )
        if len(ds_full) < 10:
            logging.warning("Not enough data in CSV => exiting.")
            G.set_status("Training", "CSV data insufficient")
            return
        if use_prev_weights:
            ensemble.load_best_weights(weights_path, data_full=train_data)
        n_tot = len(ds_full)
        n_tr = int(n_tot * 0.9)
        n_val = n_tot - n_tr
        ds_train, ds_val = random_split(ds_full, [n_tr, n_val])

        train_indicators = compute_indicators(
            train_data,
            ensemble.indicator_hparams,
            with_scaled=True,
        )

        workers = int(config.get("NUM_WORKERS", G.cpu_limit))
        logging.info(
            "DATALOADER", extra={"workers": workers, "device": ensemble.device.type}
        )
        dl_train = rebuild_loader(
            None, ds_train, batch_size=128, shuffle=True, num_workers=workers
        )
        dl_val = rebuild_loader(
            None, ds_val, batch_size=128, shuffle=False, num_workers=workers
        )
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

            status_msg = (
                f"Epoch {ensemble.train_steps}/{max_epochs} – loss {tl:.4f}"
                if max_epochs
                else f"Epoch {ensemble.train_steps} – loss {tl:.4f}"
            )
            G.set_status("Training", status_msg)

            if holdout_data is not None and len(holdout_data) > 0:
                holdout_res = robust_backtest(ensemble, holdout_data)
                G.global_holdout_sharpe = holdout_res.get("sharpe", 0.0)
                G.global_holdout_max_drawdown = holdout_res.get("max_drawdown", 0.0)
            else:
                G.global_holdout_sharpe = 0.0
                G.global_holdout_max_drawdown = 0.0
            G.global_training_loss.append(tl)
            if vl is not None:
                G.global_validation_loss.append(vl)
            else:
                G.global_validation_loss.append(None)
            if G.global_backtest_profit:
                _ = G.global_backtest_profit[-1]
            last_reward = (
                G.global_composite_reward if G.global_composite_reward else 0.0
            )
            G.global_composite_reward_ema = (
                0.9 * (G.global_composite_reward_ema or 0.0) + 0.1 * last_reward
            )
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
                "sharpe": G.global_sharpe,
                "max_dd": G.global_max_drawdown,
                "net_pct": G.global_net_pct,
                "holdout_sharpe": G.global_holdout_sharpe,
                "holdout_dd": G.global_holdout_max_drawdown,
                "attn": attn_mean,
                "attn_entropy": attn_entropy,
                "lr": lr_now,
                "profit_factor": G.global_profit_factor,
            }
            logging.info(
                "EPOCH_METRICS",
                extra=log_obj,
            )

            from artibot.hyperparams import RISK_FILTER, WARMUP_STEPS

            if G.get_warmup_step() >= WARMUP_STEPS:
                RISK_FILTER["MIN_SHARPE"] = 0.5
                RISK_FILTER["MAX_DRAWDOWN"] = -0.30

            if globals.get_trade_count() >= 1000 and not globals.PROD_RISK:
                hyperparams.RISK_FILTER.update(
                    {"MIN_SHARPE": 0.5, "MAX_DRAWDOWN": -0.30}
                )
                globals.PROD_RISK = True

            sharpe = G.global_sharpe
            max_dd = G.global_max_drawdown
            entropy = attn_entropy
            if overfit_toy:
                print(f"Toy epoch {epochs} loss {tl:.4f}")
            if not overfit_toy and reject_if_risky(sharpe, max_dd, entropy):
                logging.info(
                    "REJECTED",
                    extra={
                        "epoch": ensemble.train_steps,
                        "sharpe": sharpe,
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
                    if len(ds_updated) > 10:
                        nt_ = len(ds_updated)
                        ntr_ = int(nt_ * 0.9)
                        nv_ = nt_ - ntr_
                        ds_tr_, ds_val_ = random_split(ds_updated, [ntr_, nv_])
                        logging.info(
                            "DATALOADER",
                            extra={"workers": workers, "device": ensemble.device.type},
                        )
                        dl_train = rebuild_loader(
                            dl_train,
                            ds_tr_,
                            batch_size=128,
                            shuffle=True,
                            num_workers=workers,
                        )
                        dl_val = rebuild_loader(
                            dl_val,
                            ds_val_,
                            batch_size=128,
                            shuffle=False,
                            num_workers=workers,
                        )
                        train_indicators = compute_indicators(
                            train_data,
                            ensemble.indicator_hparams,
                            with_scaled=True,
                        )
                        ensemble.train_one_epoch(
                            dl_train,
                            dl_val,
                            train_data,
                            stop_event,
                            features=train_indicators,
                            update_globals=update_globals,
                        )

            equity = G.global_equity_curve[-1][1] if G.global_equity_curve else 0.0
            val_sharpe = G.global_holdout_sharpe
            logging.info(
                "EPOCH",
                extra={
                    "epoch": ensemble.train_steps,
                    "loss": tl,
                    "sharpe": G.global_sharpe,
                    "val_sharpe": val_sharpe,
                    "net_pct": G.global_net_pct,
                    "lr": lr_now,
                    "equity": equity,
                },
            )

            if ensemble.train_steps % 5 == 0 and ensemble.best_state_dicts:
                ensemble.save_best_weights(weights_path)

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
        # Symbol is currently fixed for simplicity
        self.symbol = "BTCUSD"
        api_conf = config.get("API", {})
        self.live_trading = bool(api_conf.get("LIVE_TRADING", True))

        key = (
            api_conf.get("API_KEY_LIVE", "")
            if self.live_trading
            else api_conf.get("API_KEY_TEST", "")
        )
        secret = (
            api_conf.get("API_SECRET_LIVE", "")
            if self.live_trading
            else api_conf.get("API_SECRET_TEST", "")
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
