"""Command line entry point for Artibot."""

# ruff: noqa: E402

import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.INFO, format="%(message)s")
root = logging.getLogger()
fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=2)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(message)s"))
root.addHandler(fh)
# allow INFO only for key modules
logging.getLogger("artibot.model").setLevel(logging.INFO)
logging.getLogger("artibot.ensemble").setLevel(logging.INFO)

from artibot.environment import ensure_dependencies
from artibot.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    ensure_dependencies()
    from artibot.bot_app import run_bot

    run_bot()
