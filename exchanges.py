"""CCXT-based helper for simplified Phemex access."""

import logging
import re
from typing import List


class ExchangeConnector:
    """Thin wrapper around :mod:`ccxt` for simple order placement."""

    def __init__(self, config):
        import ccxt

        self.symbol = config.get("symbol", "BTC/USDT")
        api_conf = config.get("API", {})
        self.live = bool(api_conf.get("LIVE_TRADING", False))
        key = (
            api_conf.get("API_KEY_LIVE") if self.live else api_conf.get("API_KEY_TEST")
        )
        secret = (
            api_conf.get("API_SECRET_LIVE")
            if self.live
            else api_conf.get("API_SECRET_TEST")
        )
        default_type = api_conf.get("DEFAULT_TYPE", "swap")
        api_url = (
            api_conf.get("API_URL_LIVE") if self.live else api_conf.get("API_URL_TEST")
        )

        params = {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        }
        if api_url:
            params["urls"] = {"api": {"public": api_url, "private": api_url}}

        self.exchange = ccxt.phemex(params)
        if not self.live:
            self.exchange.set_sandbox_mode(True)
        self.exchange.load_markets()

        for cand in self._generate_candidates(self.symbol):
            if cand in self.exchange.markets:
                self.symbol = cand
                break

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_candidates(symbol: str) -> List[str]:
        parts = [p for p in re.split(r"[/:]", symbol) if p]
        if len(parts) == 2:
            base, quote = parts
            return [
                f"{base}/{quote}",
                f"{base}{quote}",
                f"{base}:{quote}",
                f"{base}/USDT",
                f"{base}USDT",
            ]
        return [symbol]

    def fetch_latest_bars(self, limit: int = 24):
        try:
            return self.exchange.fetch_ohlcv(self.symbol, timeframe="1h", limit=limit)
        except Exception as exc:  # pragma: no cover - network errors
            logging.error("fetch_ohlcv failed: %s", exc)
            return []

    def create_order(self, side: str, amount: float, price=None, order_type="market"):
        return self.exchange.create_order(self.symbol, order_type, side, amount, price)
