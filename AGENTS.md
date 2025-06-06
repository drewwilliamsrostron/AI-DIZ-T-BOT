# AGENTS

This repository contains a cryptocurrency trading bot with a GUI and optional reinforcement learning features. The code is divided between the modern `artibot` package and several legacy scripts. The main entry point is `run_artibot.py` which imports `artibot.bot_app:run_bot()`.

## Repository layout

```
run_artibot.py                - command line entry point
ARTIBOT.py                    - older all-in-one version of the bot
master_config.json            - example configuration with API keys (do not commit real keys)
Gemini_BTCUSD_1h.csv          - sample historical data
best_model_weights*.pth       - saved model checkpoints

artibot/                      - current modular bot implementation
    bot_app.py                - starts training, live polling and GUI
    dataset.py                - CSV loader and dataset definitions
    ensemble.py               - model ensemble controller
    training.py               - background training loop and exchange connector
    rl.py                     - meta‑reinforcement learning agent
    backtest.py               - evaluation utilities
    metrics.py                - performance statistics
    model.py                  - Transformer model definition
    gui.py                    - Tkinter user interface
    environment.py            - dependency installer and compatibility shim
    globals.py                - global variables shared across modules

Reinforcement Learning Test/  - experimental offline RL prototype
Last_working_bot_but_one_py/  - previous single-file bot version
Old bot versions/             - assorted historical scripts
```

## Working with the codebase

1. **Python version** – Python 3.9–3.13 are supported. Dependencies install automatically on first run via `environment.py`.
2. **Configuration** – Edit API credentials in `artibot/bot_app.py` or in `master_config.json` (if loading from there). Never commit real keys.
3. **Running the bot** – Execute `python run_artibot.py` from the repository root. The first run may take several minutes while packages install.
4. **Coding style** – Follow PEP 8 conventions and use f-strings. Each module should include a short docstring. Keep imports relative inside the `artibot` package.
5. **Version control** – Commit descriptive messages. Do not commit any API keys or large `best_model_weights*.pth` files.
6. **Testing** – No automated tests exist. To verify manually, run the bot and observe console output and GUI for errors.

The project is for demonstration purposes and does **not** constitute investment advice. Use at your own risk and test thoroughly before live trading.
