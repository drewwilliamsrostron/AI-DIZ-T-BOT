# Artibot

[![CI](https://img.shields.io/badge/CI-none-lightgrey)](#)
[![Tests](https://img.shields.io/badge/tests-manual-orange)](#)
[![PyPI](https://img.shields.io/badge/PyPI-n/a-lightgrey)](#)
[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9--3.13-blue)](#)

Artibot is a modular Python trading bot. It trains a Transformer on hourly
OHLCV data, backtests every epoch and can trade live on the Phemex exchange.
A Tkinter dashboard visualises the training progress while a small
reinforcement learning agent tweaks hyper‑parameters over time.

## Architecture

```
run_artibot.py ──> bot_app.run_bot ──────────────┐
                           │                     │
                           ├─ csv_training_thread
                           │
                           ├─ phemex_live_thread
                           │
                           └─ meta_control_loop (RL)
                                 │
                                 ▼
                              EnsembleModel
                                   │
                                   ▼
                                TradingGUI
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
# either let the auto installer run on first launch …
python run_artibot.py
# …or install requirements yourself
pip install torch openai ccxt pandas numpy matplotlib scikit-learn TA-Lib pytest
```

The environment sets `NUMEXPR_MAX_THREADS` to the CPU count when the variable is
not defined and loads a small NumPy 2.x compatibility shim. Set
`ARTIBOT_SKIP_INSTALL=1` to disable the automatic installer.

## Configuration

Create `master_config.json` with your credentials. The bot currently trades
`BTCUSD` only so the symbol option was removed. Important keys:

```json
{
  "API": {
    "LIVE_TRADING": true,
    "SIMULATED_TRADING": false,
    "API_KEY_LIVE": "...",
    "API_SECRET_LIVE": "...",
    "API_KEY_TEST": "...",
    "API_SECRET_TEST": "...",
    "DEFAULT_TYPE": "swap",
    "API_URL_LIVE": "https://api.phemex.com",
    "API_URL_TEST": "https://testnet-api.phemex.com"
  },
  "CSV_PATH": "Gemini_BTCUSD_1h.csv",
  "NUM_WORKERS": 4,
  "ADAPT_TO_LIVE": true,
  "LIVE_POLL_INTERVAL": 900,
  "MIN_HOLD_SECONDS": 1800,
  "MIN_SHARPE": 1.0,
  "MAX_DRAWDOWN": -0.3,
  "MIN_ENTROPY": 1.0,
  "MIN_PROFIT_FACTOR": 1.0,
  "SHOW_WEIGHT_SELECTOR": false,
  "USE_PREV_WEIGHTS": true,
  "WEIGHTS_DIR": "weights",
  "CHATGPT": {"API_KEY": "..."}
}
```

## Usage

Start the bot and follow the prompt to choose live or sandbox mode:

```bash
python run_artibot.py
```

Run a short training session without the GUI:

```bash
python - <<'PY'
import artibot
artibot.run_bot(max_epochs=5)
PY
```

A quick smoke test is provided for CI:

```bash
python scripts/smoke.py --summary
```

### Live vs. Sandbox

Answer `n` at the startup prompt or set `LIVE_TRADING` to `false` in the config
to route orders to the Phemex sandbox while still running the full trading
cycle. Respond `y` for real trades on the live exchange.

## Testing

Run the linters and unit tests before committing:

```bash
pre-commit run --all-files
pytest -q
```

See [AGENTS.md](AGENTS.md) for development conventions and troubleshooting tips.
