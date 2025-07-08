# Artibot

[![CI](https://img.shields.io/badge/CI-none-lightgrey)](#)
[![Tests](https://img.shields.io/badge/tests-manual-orange)](#)
[![PyPI](https://img.shields.io/badge/PyPI-n/a-lightgrey)](#)
[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9--3.13-blue)](#)

Artibot is a modular Python trading bot. It trains a Transformer on hourly
OHLCV data, backtests every epoch and can trade live on the Phemex exchange.
A Tkinter dashboard visualises the training progress while a small
reinforcement learning agent tweaks hyper‑parameters over time.

Recent updates added a policy‑gradient RL loop, optional technical indicators
(SMA, RSI, MACD, ATR, Vortex, CMF, EMA, Donchian, Kijun, Tenkan) and a
`HyperParams` dataclass loaded from ``master_config.json``.  The bot can now
hedge long/short exposure and dynamically enable or disable indicators during
training.

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

## Trading Logic

Long and short exposure are controlled by `long_frac` and `short_frac` in
`HyperParams`. The meta agent adjusts them between 0–10 % of account equity.
Gross exposure is capped at 12 % so the bot never risks more capital than
configured.

Indicator usage can be toggled on a per‑bar basis. New flags include `USE_EMA`,
`USE_DONCHIAN`, `USE_KIJUN`, `USE_TENKAN` and `USE_DISPLACEMENT` with matching
period settings. All fields live in `IndicatorHyperparams` and can be changed
during training.

| Flag | Period field |
|------|--------------|
| `USE_EMA` | `EMA_PERIOD` |
| `USE_DONCHIAN` | `DONCHIAN_PERIOD` |
| `USE_KIJUN` | `KIJUN_PERIOD` |
| `USE_TENKAN` | `TENKAN_PERIOD` |
| `USE_DISPLACEMENT` | `DISPLACEMENT` |

## Trading Logic - Feature Set

In addition to price-based indicators the agent now ingests **three contextual
features** on every hourly bar:

| Feature | Source | Purpose |
|---------|--------|---------|
| `sent_24h` | 24-hour mean FinBERT score of BTC headlines | Captures crowd sentiment |
| `macro_z`  | Z-score of latest macro surprise (CPI, NFP …) | Detects macro regime shifts |
| `rvol_7d`  | 7-day realised volatility | Allows position sizing relative to risk |

These vectors are appended to the model’s input and are automatically
populated by `artibot/feature_store.py`.

These vectors are appended to the model’s input **and are updated live** by
`artibot/feature_ingest.py`, which runs automatically when you start
`run_artibot.py`.  The job fetches:

* CryptoPanic headlines → FinBERT sentiment (+1 = bullish, -1 = bearish)  
* Economic-surprise numbers (e.g. CPI) → Z-scores  
* BTC 7-day realised volatility via the bundled CSV plus live Phemex data

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
# either let the auto installer run on first launch …
python run_artibot.py
# Windows users can double‑click ``run_artibot.bat`` instead
# …or install requirements yourself
pip install torch openai ccxt pandas numpy matplotlib scikit-learn TA-Lib pytest
```

Users on Python 3.11 should install a CUDA 12.1 build of PyTorch 2.x to enable
GPU acceleration. The bot falls back to CPU when no compatible wheel is found.

### GPU quick-start

```bash
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio --force-reinstall --no-cache-dir
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

Windows and Linux both require an NVIDIA driver version 545.23 or newer for CUDA 12.1.

The environment sets `NUMEXPR_MAX_THREADS` to the CPU count when the variable is
not defined, downgrades NumPy to the latest 1.x release for legacy packages and
chooses the correct CPU or CUDA build of PyTorch automatically.  Set
`ARTIBOT_SKIP_INSTALL=1` to disable the automatic installer.

### FlashAttention / SDP kernels

Install a **PyTorch nightly** wheel built with `USE_FLASH_ATTENTION=1` or compile
from source.  Call `torch.backends.cuda.enable_flash_sdp(True)` before training
to enable FlashAttention‑v2.  Benchmarks show 1.5–3× speed‑ups on sequences of
≤128 tokens and up to 40 % shorter epochs when profiling with
`torch.profiler.schedule(wait=1, warmup=1, active=3)`.

### FlashAttention Auto-Install

To automatically install a FlashAttention-capable PyTorch nightly on CUDA 11.8:

```bash
export FLASH_SDP_AUTO_INSTALL=1
python run_artibot.py
```

The wheels are hosted at <https://download.pytorch.org/whl/nightly/cu118>.

## Configuration

Create `master_config.json` with your credentials. The bot currently trades
`BTCUSD` only so the symbol option was removed. Important keys include the new
stop‑loss/take‑profit parameters and ATR threshold:

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
  "MIN_REWARD": 1.0,
  "MAX_DRAWDOWN": -0.3,
  "MIN_ENTROPY": 1.0,
  "MIN_PROFIT_FACTOR": 1.0,
  "SL": 5.0,
  "TP": 5.0,
  "ATR_THRESHOLD_K": 1.5,
  "SHOW_WEIGHT_SELECTOR": false,
  "USE_PREV_WEIGHTS": true,
  "WEIGHTS_DIR": "weights",
  "CHATGPT": {"API_KEY": "..."}
}
```

### Hyperparameter Sweep

Define `experiment_axes` in your config YAML and run:

```bash
export FLASH_SDP_AUTO_INSTALL=1
python scripts/sweep.py --config config/hyperparams.yaml --early_stop_epochs 3 --top_k 3
```

## Usage

Start the bot and follow the prompt to choose live or sandbox mode:

```bash
python run_artibot.py
```
The Tkinter dashboard opens automatically. The bot performs a quick
backtest on startup. A nuclear key gates trading until the mean Sharpe
and profit factor exceed the configured thresholds.  On recent
versions you'll also be prompted to **skip the historical sentiment
pull** if GDELT is slow. Answer `yes` to set the environment variable
`NO_HEAVY=1` and continue without downloading several gigabytes of
data.

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

### Dashboard

The Tkinter GUI displays several live metrics. The **Attention Weights** tab
shows a 3D surface highlighting how the Transformer focuses on previous price
bars. Higher peaks indicate the timesteps receiving the most attention and the
surface updates live as training progresses.  Current hyper‑parameters and
indicator toggles are shown in real time so the RL agent's decisions are easy to
track.

## Testing

Run the linters and unit tests before committing:

```bash
pre-commit run --all-files
pytest -q
```

See [AGENTS.md](AGENTS.md) for development conventions and troubleshooting tips.

### Back-filling contextual features (2015-present)

```bash
make backfill        # runs tools/backfill_gdelt.py (≈ 1–3 h first run)
```
This script streams GDELT hourly archives [1] and TradingEconomics CPI/NFP releases [2],
computes FinBERT sentiment and realised BTC volatility, and populates DuckDB with
one row per hour so back-tests from 2015 onward see realistic context data.

Live sentiment now comes from the GDELT DOC 2.0 API (15-minute latency).

[1] GDELT hourly CSVs update every 15-minutes
gdeltproject.org

[2] TradingEconomics guest API provides actual vs consensus values
docs.tradingeconomics.com

### Bootstrap & Data Layers

Artibot installs missing wheels the first time you run it. Packages like
`schedule` and `finbert-embedding` are fetched on demand unless the
environment variable `CI` is set. Historical context is loaded via
``tools/backfill_gdelt.py`` which stitches ``Gemini_BTCUSD_1h.csv`` with live
Phemex bars and retrieves GDELT articles with retry logic. FinBERT weights download through the
Hugging Face hub and are cached under ``~/.cache/artibot/finbert`` so sentiment
lookups stay fast offline.
Set ``NO_HEAVY=1`` to skip the bulky GDELT download entirely.
