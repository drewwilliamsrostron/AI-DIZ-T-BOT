# AI-t-bot

A Python trading bot with an optional reinforcement learning component. The bot can train on historical data, backtest strategies and trade live using the Phemex exchange API. A minimal Tkinter GUI provides status information while the bot is running.

## Usage

1. Copy `.env.example` to `.env` and add your credentials. The bot will load it automatically. Required variables are:

   - `OPENAI_API_KEY` – OpenAI access key
   - `PHEMEX_API_KEY_LIVE` and `PHEMEX_API_SECRET_LIVE`
   - `PHEMEX_API_KEY_TEST` and `PHEMEX_API_SECRET_TEST`

   The provided `master_config.json` file uses the same placeholder names so you can override other settings without storing secrets in git.
2. Start the bot with:

```bash
python run_artibot.py
```

When importing `run_bot()` directly you may pass `max_epochs=<N>` to limit
training epochs for quick tests or demos.

3. On the very first run the program installs its Python dependencies via `environment.py`. This may take several minutes while the GUI remains on *Initializing…* and the console prints thread messages. Simply wait for installation to finish; the bot will continue automatically.

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
Last_working_bot_but_one_py/  - previous working single-file bot
Old bot versions/             - assorted historical scripts
```

The GUI includes a status indicator at the bottom showing what the bot is currently doing (training, fetching data, etc.). When a background thread is waiting or sleeping, the status line displays a countdown so you can see the bot is still active.
