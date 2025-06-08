"""Command line entry point for Artibot."""

from artibot.environment import ensure_dependencies
from artibot.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    ensure_dependencies()
    from artibot.bot_app import run_bot

    run_bot()
