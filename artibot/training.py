"""Background CSV training thread and exchange connector."""

# ruff: noqa: F403, F405
import artibot.globals as G

import logging
import json

import os
import re
import sys

from .dataset import HourlyDataset
from .ensemble import reject_if_risky


###############################################################################
def csv_training_thread(
    ensemble,
    data,
    stop_event,
    config,
    use_prev_weights=True,
    max_epochs: int | None = None,
):
    """Train on CSV data in a background thread.

    ``max_epochs`` stops the loop after N iterations when set.
    """
    import traceback

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, random_split

    # training history lists live on the globals module

    try:
        ds_full = HourlyDataset(
            data, seq_len=24, threshold=G.GLOBAL_THRESHOLD, train_mode=True
        )
        if len(ds_full) < 10:
            logging.warning("Not enough data in CSV => exiting.")
            return
        if use_prev_weights:
            ensemble.load_best_weights("best_model_weights.pth", data_full=data)
        n_tot = len(ds_full)
        n_tr = int(n_tot * 0.9)
        n_val = n_tot - n_tr
        ds_train, ds_val = random_split(ds_full, [n_tr, n_val])

        pin = ensemble.device.type == "cuda"
        default_workers = max(1, os.cpu_count() or 1)
        workers = int(config.get("NUM_WORKERS", default_workers))
        dl_train = DataLoader(
            ds_train, batch_size=128, shuffle=True, num_workers=workers, pin_memory=pin
        )
        dl_val = DataLoader(
            ds_val, batch_size=128, shuffle=False, num_workers=workers, pin_memory=pin
        )

        adapt_live = bool(config.get("ADAPT_TO_LIVE", False))
        dummy_input = torch.randn(1, 24, 8, device=ensemble.device)
        ensemble.optimize_models(dummy_input)

        import talib

        epochs = 0
        best_reward = float("-inf")
        no_gain = 0
        while not stop_event.is_set():
            if max_epochs is not None and epochs >= max_epochs:
                break
            ensemble.train_steps += 1
            epochs += 1
            G.set_status(f"Training step {ensemble.train_steps}")
            logging.debug(json.dumps({"event": "status", "msg": G.get_status()}))
            tl, vl = ensemble.train_one_epoch(dl_train, dl_val, data, stop_event)
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
            lr_now = ensemble.cosine[0].get_last_lr()[0]
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
                "attn": attn_mean,
                "attn_entropy": attn_entropy,
                "lr": lr_now,
            }
            logging.info(
                json.dumps(log_obj),
                extra={
                    "epoch": ensemble.train_steps,
                    "sharpe": G.global_sharpe,
                    "max_dd": G.global_max_drawdown,
                    "attn_entropy": attn_entropy,
                    "lr": lr_now,
                },
            )

            sharpe = G.global_sharpe
            max_dd = G.global_max_drawdown
            entropy = attn_entropy
            if reject_if_risky(sharpe, max_dd, entropy):
                logging.info(
                    json.dumps(
                        {
                            "event": "REJECTED",
                            "sharpe": sharpe,
                            "max_dd": max_dd,
                            "entropy": entropy,
                        }
                    )
                )
                G.inc_epoch()
                continue

            if last_reward > best_reward:
                best_reward = last_reward
                no_gain = 0
            else:
                no_gain += 1
            if no_gain >= 10:
                logging.info(
                    json.dumps({"event": "EARLY_STOP", "epoch": ensemble.train_steps})
                )
                break

            # quick "latest prediction"
            if len(data) >= 24:
                tail = np.array(data[-24:], dtype=np.float64)
                closes = tail[:, 4]
                sma = np.convolve(closes, np.ones(10) / 10, mode="same")
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
                        if ts > data[-1][0]:
                            data.append([ts, o_, h_, l_, c_, v_])
                            changed = True
                if changed:
                    G.set_status("Adapting to live data")
                    ds_updated = HourlyDataset(
                        data,
                        seq_len=24,
                        threshold=G.GLOBAL_THRESHOLD,
                        train_mode=True,
                    )
                    if len(ds_updated) > 10:
                        nt_ = len(ds_updated)
                        ntr_ = int(nt_ * 0.9)
                        nv_ = nt_ - ntr_
                        ds_tr_, ds_val_ = random_split(ds_updated, [ntr_, nv_])
                        pin = ensemble.device.type == "cuda"
                        default_workers = max(1, os.cpu_count() or 1)
                        workers = int(config.get("NUM_WORKERS", default_workers))
                        dl_tr_ = DataLoader(
                            ds_tr_,
                            batch_size=128,
                            shuffle=True,
                            num_workers=workers,
                            pin_memory=pin,
                        )
                        dl_val_ = DataLoader(
                            ds_val_,
                            batch_size=128,
                            shuffle=False,
                            num_workers=workers,
                            pin_memory=pin,
                        )
                        ensemble.train_one_epoch(dl_tr_, dl_val_, data, stop_event)

            if ensemble.train_steps % 5 == 0 and ensemble.best_state_dicts:
                ensemble.save_best_weights("best_model_weights.pth")

    except Exception as e:
        traceback.print_exc()

        G.set_status(f"Training error: {e}")

        stop_event.set()


def phemex_live_thread(connector, stop_event, poll_interval: float) -> None:
    """Continuously fetch recent bars from Phemex at a configurable interval."""
    import traceback

    while not stop_event.is_set():
        try:
            G.set_status("Fetching live data")
            bars = connector.fetch_latest_bars(limit=100)
            if bars:
                G.global_phemex_data = bars

                G.live_bars_queue.put(bars)

        except Exception as e:
            traceback.print_exc()
            G.set_status(f"Fetch error: {e}")
            stop_event.set()
        G.status_sleep("Waiting before next fetch", poll_interval)


###############################################################################
# Connector
###############################################################################
class PhemexConnector:
    """Thin wrapper around ccxt for polling Phemex."""

    def __init__(self, config):
        self.symbol = config.get("symbol", "BTC/USDT")
        api_conf = config.get("API", {})
        self.api_key = api_conf.get("API_KEY_LIVE", "")
        self.api_secret = api_conf.get("API_SECRET_LIVE", "")
        default_type = api_conf.get("DEFAULT_TYPE", "spot")
        import ccxt

        try:
            self.exchange = ccxt.phemex(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": default_type},
                }
            )
        except Exception as e:
            logging.error(f"Error initializing exchange: {e}")
            sys.exit(1)
        self.exchange.load_markets()
        cands = generate_candidates(self.symbol)
        for c in cands:
            if c in self.exchange.markets:
                self.symbol = c
                break

    def fetch_latest_bars(self, limit=100):
        try:
            bars = self.exchange.fetch_ohlcv(self.symbol, timeframe="1h", limit=limit)
            return bars if bars else []
        except Exception as e:
            logging.error(f"Error fetching bars: {e}")
            return []


def generate_candidates(symbol):
    """Return possible market symbol permutations."""
    parts = re.split(r"[/:]", symbol)
    parts = [x for x in parts if x]
    cands = set()
    if len(parts) == 2:
        base, quote = parts
        cands.update(
            {
                f"{base}/{quote}",
                f"{base}{quote}",
                f"{base}:{quote}",
                f"{base}/USDT",
                f"{base}USDT",
            }
        )
    else:
        cands.add(symbol)
    return list(cands)


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
