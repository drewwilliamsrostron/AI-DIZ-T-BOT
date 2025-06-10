"""Command line entry point for Artibot."""

# ruff: noqa: E402

from artibot.environment import ensure_dependencies

ensure_dependencies()

import logging
from logging.handlers import RotatingFileHandler
import os
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
root = logging.getLogger()
fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=2)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(message)s"))
root.addHandler(fh)
# allow INFO only for key modules
logging.getLogger("artibot.model").setLevel(logging.INFO)
logging.getLogger("artibot.ensemble").setLevel(logging.INFO)

from artibot.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_num_interop_threads(os.cpu_count() or 1)
    from artibot.bot_app import run_bot, CONFIG

    ans = input("Use LIVE API? [y/N]: ").strip().lower()
    CONFIG.setdefault("API", {})["LIVE_TRADING"] = ans.startswith("y")

    run_bot()
