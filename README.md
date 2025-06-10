# Artibot

[![CI](https://img.shields.io/badge/CI-none-lightgrey)](#)
[![Tests](https://img.shields.io/badge/tests-manual-orange)](#)
[![PyPI](https://img.shields.io/badge/PyPI-n/a-lightgrey)](#)
[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9--3.13-blue)](#)

A Python trading bot with an optional reinforcement learning component. The bot can train on historical data, backtest strategies and trade live using the Phemex exchange API. A minimal Tkinter GUI provides status information while the bot is running.

## Quick start

Legacy CSV feeds store prices scaled by `1e5` or `1e3`. The loader now detects these multipliers so everything is expressed in dollars. After indicator creation all features are normalised with a rolling z-score and clipped to ±50. Verify your setup with the bundled smoke test:

```bash
python scripts/smoke.py
```

It runs a quick 10‑epoch training session and prints the final composite reward and average candle range.

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

The launcher prints your Phemex account balance on start so you can verify the
API connection.

4. On the very first run the program installs its Python dependencies automatically via `environment.ensure_dependencies()`. The GUI may sit on *Initializing…* for a few minutes while packages download—just let it finish.
5. The DataLoader worker count now defaults to all available CPU cores
   (`os.cpu_count()` with no 8‑worker cap). Define `"NUM_WORKERS"` in
   `master_config.json` to set a specific value or add `"FORCE_ZERO_WORKERS":
   true` to disable multiprocessing. If NumExpr prints a warning about thread
   limits, the environment module sets `NUMEXPR_MAX_THREADS` to the detected CPU
   count when the variable is not already defined.
6. PyTorch is configured on startup to use all CPU cores for intra- and
   inter‑op parallelism via `torch.set_num_threads(os.cpu_count() or 1)` and
   `torch.set_num_interop_threads(os.cpu_count() or 1)`.
7. When running on PyTorch ≥ 2.2 the bot sets
   `torch.set_float32_matmul_precision("high")` and compiles each model with
   `torch.compile`. Both calls are guarded with `hasattr` so older versions work
   unchanged.


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

All threads call `set_status(primary, secondary)` and you can read both via `get_status_full()`.
## Running the test-suite

The repo includes lightweight tests that stub heavy dependencies so they run quickly. After installing requirements, execute:

```bash
pytest -q
```

For in-depth development conventions see [AGENTS.md](AGENTS.md).

## Development setup

Install [pre-commit](https://pre-commit.com/) and install the hooks once:

```bash
pip install pre-commit
pre-commit install
```

Run the checks locally before pushing:

```bash
pre-commit run --all-files
pytest -q
```

## Advanced usage

The helper modules in `artibot.utils` auto-select a CUDA device when
Orders are sent through `execution.submit_order`, which adds a short random delay and jitter to emulate slippage.
available and configure JSON logging.  `scripts/sweep.py` performs a small
grid search over learning rates and TP/SL multipliers, writing the reward for
each combination to `sweeps/results.csv`.

## Risk filters & attention diagnostics

Every epoch logs the Transformer attention mean and entropy. By default a model
is rejected when Sharpe drops below **1.0**, drawdown exceeds **-30 %**, or the
attention entropy falls under **1.0**. These limits can be overridden via
`MIN_SHARPE`, `MAX_DRAWDOWN` and `MIN_ENTROPY` in
`master_config.json` to suit
different risk appetites. The sweep script outputs the metrics and skips
rejected rows. Visualise one batch of weights with:

```bash
python scripts/plot_attention.py weights.npy
```
Entropy should stay between 1 and 2.5; zero means collapse.


## Nuclear Key workflow

Live trades are gated by the *Nuclear Key*.  Trading only proceeds when the
latest Sharpe ratio is at least **1.5**, drawdown is above **-20 %** and the
profit factor exceeds **1.5**.  When any metric falls outside these limits the
button labelled "Nuclear Key" turns grey and orders are blocked.

The dashboard includes buttons to close or edit the active trade and to enable
live mode.  "Close Active Trade" cancels all orders and exits the current
position.  "Edit Trade" lets you tweak SL/TP multipliers for the next order.
"Nuclear Key" becomes available once the gate conditions pass and requires a
manual click to start live trading.

A lightweight validation pipeline runs periodically (via
`schedule_monthly_validation`) to recalculate the Sharpe distribution and update
the gate state automatically.
