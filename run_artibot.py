"""Command line entry point for the trading bot package."""

from artibot.environment import ensure_dependencies
from artibot.bot_app import run_bot

if __name__ == '__main__':
    ensure_dependencies()
    run_bot()
