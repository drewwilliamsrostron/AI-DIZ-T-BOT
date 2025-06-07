"""Command line entry point for the trading bot package."""

from __future__ import annotations

from artibot.environment import ensure_dependencies

if __name__ == "__main__":
    ensure_dependencies()
    from artibot.bot_app import run_bot

    run_bot()
