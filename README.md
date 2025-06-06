# AI-t-bot
AI Trading Bot

## Usage

1. Edit `artibot/bot_app.py` and place your API credentials inside the
   `CONFIG` dictionary at the top of the file. The `API` section holds your
   exchange keys while `CHATGPT` contains your OpenAI key.
   The `USE_PREV_WEIGHTS` flag controls whether an existing
   `best_model_weights.pth` file is automatically loaded.
2. Start the bot with:

```bash
python run_artibot.py
```

The package is organised into a few self‑contained modules:

| File | Purpose |
|------|---------|
| `bot_app.py` | launches training, live polling and the GUI |
| `backtest.py` | evaluation utilities used during training |
| `dataset.py` | loads and preprocesses historical data |
| `ensemble.py` | neural‑network ensemble controller |
| `metrics.py` | computes performance statistics |
| `training.py` | background training loop |
| `rl.py` | meta‑learning agent |
| `gui.py` | simple Tkinter interface |
| `environment.py` | installs dependencies and compatibility shims |

The GUI now includes a single-line status indicator at the bottom showing
what the bot is currently doing (training, fetching data, etc.).  When a
background thread is waiting or sleeping, the status line displays a
countdown so you can see the bot is still active.
