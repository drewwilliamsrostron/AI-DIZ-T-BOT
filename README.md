# AI-t-bot

A Python trading bot with an optional reinforcement learning component. The bot can train on historical data, backtest strategies and trade live using the Phemex exchange API. A minimal Tkinter GUI provides status information while the bot is running.

## Usage

1. Edit `artibot/bot_app.py` and place your API credentials inside the `CONFIG` dictionary at the top of the file. The `API` section holds exchange keys while `CHATGPT` contains your OpenAI key. The `CSV_PATH` entry is resolved relative to the project directory so the data file is found even when you launch the bot from elsewhere.
2. Start the bot with:

```bash
python run_artibot.py
```

3. On the very first run the program installs its Python dependencies via `environment.py`. This may take several minutes while the GUI remains on *Initializing…* and the console prints thread messages. Simply wait for installation to finish; the bot will continue automatically.

## Project structure

```
run_artibot.py                - command line launcher
ARTIBOT.py                    - legacy all-in-one version
master_config.json            - example config with API keys
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
