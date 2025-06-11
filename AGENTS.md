# Artibot – Python Trading-Bot Development Guide

## Project Overview

Artibot trains a small Transformer on hourly market data and places orders on
Phemex.  Training, live polling, the meta reinforcement agent and the Tkinter
GUI all run in separate threads.  Dependencies are installed lazily on the first
launch via `environment.ensure_dependencies()`.

## Repository Structure

```
run_artibot.py                - CLI launcher
exchanges.py                  - simplified ccxt wrapper
master_config.json            - sample configuration (never commit real keys)
Gemini_BTCUSD_1h.csv          - example dataset

artibot/                      - production code
│  __init__.py               - exposes run_bot()
│  bot_app.py                - orchestration logic
│  dataset.py                - CSV loader and dataset classes
│  model.py                  - Transformer definition
│  ensemble.py               - ensemble controller
│  training.py               - background training threads
│  rl.py                     - meta-reinforcement agent
│  backtest.py               - evaluation utilities
│  metrics.py                - performance statistics
│  validation.py             - monthly walk-forward checks
│  live_risk.py              - live auto-pause helpers
│  risk.py                   - position sizing functions
│  position.py               - open/close position helpers
│  execution.py              - order jitter utilities
│  gui.py                    - Tkinter dashboard
│  environment.py            - first-run installer
│  globals.py                - shared state
```

## Agents and Scripts

| Name / path | Role | Parameters / config | Example |
|-------------|------|--------------------|---------|
| **run_artibot.py** | Command line entry for live trading. Prompts for live vs testnet and launches the GUI automatically. Trading stays disabled until backtest metrics enable the nuclear key. | reads `master_config.json` | `python run_artibot.py` |
| **run_bot** (`artibot.bot_app`) | Start training loop, live polling and GUI. | `max_epochs=None` | `import artibot; artibot.run_bot()` |
| **csv_training_thread** (`artibot.training`) | Train on CSV data in a worker thread. | `ensemble`, `data`, `stop_event`, `config`, `max_epochs` | used inside `run_bot` |
| **phemex_live_thread** (`artibot.training`) | Fetch recent bars from Phemex. | `connector`, `stop_event`, `poll_interval` | used inside `run_bot` |
| **EnsembleModel** (`artibot.ensemble`) | Container for Transformer models. | `device`, `n_models=2`, `lr=3e-4`, `weight_decay=1e-4` | `ens = EnsembleModel(device)` |
| **MetaTransformerRL** (`artibot.rl`) | Reinforcement learner that tweaks LR, WD and indicator periods. | `ensemble`, `lr=1e-3` | `agent = MetaTransformerRL(ens)` |
| **validate_and_gate** (`artibot.validation`) | Walk‑forward analysis and nuclear‑key gating. | `csv_path`, `config` | `validate_and_gate('data.csv', CONFIG)` |
| **update_auto_pause** (`artibot.live_risk`) | Pause trading when metrics fall below limits. | `sharpe`, `drawdown`, `ts=None` | `update_auto_pause(1.2, -0.05)` |
| **scripts/smoke.py** | Ten‑epoch smoke test. | `--summary` writes key metrics | `python scripts/smoke.py --summary` |
| **scripts/sweep.py** | Grid search over LR and TP/SL. | none | `python scripts/sweep.py` |

## Development Guidelines

* **One branch = one idea** – keep `main` runnable.
* Format with **Black** and lint with **Ruff** (`pre-commit run --all-files`).
* Tests live in `tests/` and are run with `pytest -q`.
* Keep new modules inside `artibot/` and include a short docstring and type hints.
* Never store secrets in the repo – use `master_config.json` locally.
* Use `queue.Queue` for cross‑thread communication; avoid new globals.

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # optional – auto install happens on first run
python run_artibot.py            # downloads packages and starts the GUI
```

## Testing Strategy

```bash
pre-commit run --all-files
pytest -q
```

A single passing test is enough for a pull request.  Green output means
merge‑ready.

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `RuntimeError: main thread is not in main loop` | Tkinter called from a worker thread | use `root.after` to schedule UI updates |
| GPU not found warnings | No CUDA device | PyTorch falls back to CPU |
| Hang during training | Waiting on an empty queue | ensure the producer thread is alive |
| Linter flood after commit | forgot `pre-commit install` | run it once then commit |

## Reference Resources

* Python `queue` docs – <https://docs.python.org/3/library/queue.html>
* Tkinter thread safety FAQ – <https://stackoverflow.com/questions/58118723>
* PyTorch CPU/GPU tips – <https://stackoverflow.com/questions/53266350>
* Black – <https://github.com/psf/black>
* Ruff – <https://docs.astral.sh/ruff/linter/>
* pytest quick-start – <https://docs.pytest.org/en/stable/getting-started.html>

