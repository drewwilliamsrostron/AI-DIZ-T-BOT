"""CCXT-based helper for simplified Phemex access."""

import logging
import time


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
        self.default_type = default_type
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

    def fetch_latest_bars(self, limit: int = 24):
        now = int(time.time())
        last_hour = now - (now % 3600)
        since_ms = (last_hour - limit * 3600) * 1000
        params = {"type": self.default_type}
        logging.debug(
            "fetch_ohlcv -> %s tf=%s since=%s limit=%s params=%s",
            self.symbol,
            "1h",
            since_ms,
            limit,
            params,
        )
        try:
            bars = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe="1h",
                since=since_ms,
                limit=limit,
                params=params,
            )
        except TypeError:
            logging.debug("TypeError on fetch_ohlcv, retrying without params")
            try:
                bars = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe="1h",
                    since=since_ms,
                    limit=limit,
                )
            except Exception as exc:  # pragma: no cover - network errors
                logging.error(
                    "fetch_ohlcv failed for %s tf=%s limit=%s: %s",
                    self.symbol,
                    "1h",
                    limit,
                    exc,
                )
                return []
        except Exception as exc:  # pragma: no cover - network errors
            logging.error(
                "fetch_ohlcv failed for %s tf=%s limit=%s: %s",
                self.symbol,
                "1h",
                limit,
                exc,
            )
            return []
        logging.debug("fetched %d bars", len(bars) if bars else 0)
        return bars

    def create_order(self, side: str, amount: float, price=None, order_type="market"):
        return self.exchange.create_order(self.symbol, order_type, side, amount, price)
