import logging
from typing import Any


def get_account_equity(exchange: Any) -> float:
    """Return total account equity in USDT like webbot.update_account_balance."""
    try:
        spot = exchange.fetch_balance()
        btc_spot = spot.get("BTC", {}).get("total", 0)

        swap = exchange.fetch_balance(params={"type": "swap", "code": "BTC"})
        btc_swap = swap.get("BTC", {}).get("total", 0)

        if btc_spot == btc_swap == 0:
            logging.warning("[equity] No BTC balance found – equity=0")
            return 0.0

        btc_price = exchange.fetch_ticker("BTC/USDT")["close"]
        equity_usdt = round((btc_spot + btc_swap) * btc_price, 8)
        logging.debug(
            "[equity] %.6f BTC (spot) + %.6f BTC (swap) \u2192 %.2f USDT",
            btc_spot,
            btc_swap,
            equity_usdt,
        )
        return equity_usdt
    except Exception as e:  # pragma: no cover - network errors
        logging.error("Equity fetch error: %s", e)
        return 0.0
