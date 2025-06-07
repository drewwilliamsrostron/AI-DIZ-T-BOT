# Artibot

[![CI](https://img.shields.io/badge/CI-none-lightgrey)](#)
[![Tests](https://img.shields.io/badge/tests-manual-orange)](#)
[![PyPI](https://img.shields.io/badge/PyPI-n/a-lightgrey)](#)
[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9--3.13-blue)](#)

A Python trading bot with an optional reinforcement learning component. The bot can train on historical data, backtest strategies and trade live using the Phemex exchange API. A minimal Tkinter GUI provides status information while the bot is running.

## Usage

1. Open `master_config.json` and replace the placeholder values with your Phemex
   and OpenAI credentials. The file is ignored by Git so your keys remain local.
2. *(Recommended)* create and activate a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
```

3. Start the bot with:

```bash
python run_artibot.py
```

4. On the very first run the program installs its Python dependencies automatically via `environment.ensure_dependencies()`. The GUI may sit on *Initializing…* for a few minutes while packages download—just let it finish.


## Project structure

```
run_artibot.py                - command line launcher
ARTIBOT.py                    - legacy all-in-one version
master_config.json            - example config (no secrets)
Gemini_BTCUSD_1h.csv          - sample historical data

artibot/                      - current modular implementation
    bot_app.py                - training loop, live polling and GUI
    dataset.py                - CSV loader and dataset
    ensemble.py               - model ensemble controller
    training.py               - background training thread
    rl.py                     - meta reinforcement learning agent
    backtest.py               - evaluation utilities
    metrics.py                - performance statistics
    model.py                  - Transformer model definition
    gui.py                    - Tkinter interface
    environment.py            - installs dependencies
    globals.py                - shared variables

Reinforcement Learning Test/  - experimental RL prototype
last_working_bot_but_only_using_one.py - previous working single-file bot
Old bot versions/             - assorted historical scripts
```

The GUI includes a status indicator at the bottom showing what the bot is currently doing (training, fetching data, etc.). When a background thread is waiting or sleeping, the status line displays a countdown so you can see the bot is still active.

## Running the test-suite

The repo includes lightweight tests that stub heavy dependencies so they run quickly. After installing requirements, execute:

```bash
pytest -q
```

For in-depth development conventions see [AGENTS.md](AGENTS.md).
