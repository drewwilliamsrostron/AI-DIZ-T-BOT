import threading
import time
import logging


def start(interval: int = 30) -> None:
    """Start a background heartbeat logger."""

    def _beat() -> None:
        while True:
            logging.info("[HEARTBEAT] bot alive @ %s", time.ctime())
            time.sleep(interval)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
