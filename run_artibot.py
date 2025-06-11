"""Command line entry point for Artibot."""

# ruff: noqa: E402

from artibot.environment import ensure_dependencies

ensure_dependencies()

import logging
from logging.handlers import RotatingFileHandler
import os
import json
import time
import math
import numpy as np
import torch
import talib

from artibot.utils import setup_logging, get_device
from artibot.ensemble import EnsembleModel
import artibot.globals as G
from exchanges import ExchangeConnector

logging.basicConfig(level=logging.INFO, format="%(message)s")
root = logging.getLogger()
fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=2)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(message)s"))
root.addHandler(fh)
# allow INFO only for key modules
logging.getLogger("artibot.model").setLevel(logging.INFO)
logging.getLogger("artibot.ensemble").setLevel(logging.INFO)


def get_account_equity(exchange):
    """Return total BTC equity converted to USDT."""

    spot_bal = exchange.fetch_balance()
    btc_spot = spot_bal.get("BTC", {}).get("total", 0.0)

    swap_bal = exchange.fetch_balance(params={"type": "swap", "code": "BTC"})
    btc_swap = swap_bal.get("BTC", {}).get("total", 0.0)

    btc_usdt = exchange.fetch_ticker("BTC/USDT")["close"]
    return round((btc_spot + btc_swap) * btc_usdt, 8)


def main() -> None:
    """Run the live trading loop using the pre-trained model."""
    setup_logging()
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_num_interop_threads(os.cpu_count() or 1)

    from artibot.bot_app import CONFIG

    ans = input("Use LIVE API? [y/N]: ").strip().lower()
    use_live_api = ans.startswith("y")
    use_sandbox = not use_live_api
    CONFIG.setdefault("API", {})["LIVE_TRADING"] = use_live_api
    G.use_sandbox = use_sandbox

    mode = "LIVE" if use_live_api else "SANDBOX"
    logging.info("Trading mode: %s", mode)

    connector = ExchangeConnector(CONFIG)
    try:
        bal = connector.exchange.fetch_balance()
        logging.info("ACCOUNT_BALANCE %s", json.dumps(bal))
        G.global_account_stats = bal
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("Balance fetch failed: %s", exc)
        bal = {}

    device = get_device()
    ensemble = EnsembleModel(device=device)
    weights_dir = os.path.abspath(
        os.path.expanduser(CONFIG.get("WEIGHTS_DIR", "weights"))
    )
    weights_path = os.path.join(weights_dir, "best_model_weights.pth")
    if os.path.isfile(weights_path):
        ensemble.load_best_weights(weights_path)

    poll_int = float(CONFIG.get("LIVE_POLL_INTERVAL", 60))
    risks = CONFIG.get("RISKS", {})
    timeframe = CONFIG.get("TIMEFRAME", "1h")
    leverage = float(CONFIG.get("LEVERAGE", 10))

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
        try:
            G.cancel_open_orders()
            G.close_position()
        except Exception as exc:  # pragma: no cover - network errors
            logging.error("Cleanup failed: %s", exc)


if __name__ == "__main__":
    main()
