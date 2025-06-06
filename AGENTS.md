Below is a **ready-to-save `agents.md`** that merges your latest Artibot notes **with the structured, example-heavy style** of the Django guide you liked.
The first half explains *what* the bot is and *where* things live; the second half gives Codex (and you) exact development, testing, and style rules—using the same headings and Markdown flow as the Django sample.

---

# Artibot – Python Trading-Bot Development Guide

## Project Overview

Artibot is a multithreaded Python 3 trading bot that

* ingests hourly OHLCV CSV (or live WebSocket) data,
* trains a small Transformer (PyTorch ≥ 2.2) to predict **BUY / SELL / HOLD**,
* back-tests after each epoch, and
* can stream live prices to a Tkinter dashboard while sending real orders.

All heavy lifting lives in the `artibot/` package; the older single-file bots are kept only for reference.

---

## Tech Stack

| Layer / Concern | Tool / Version                                                                                       | Why                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Language**    | Python 3.10 – 3.13                                                                                   | Tested; 3.9 still works                                                    |
| **DL / ML**     | PyTorch ≥ 2.2                                                                                        | CPU fallback if no CUDA ([stackoverflow.com][1], [discuss.pytorch.org][2]) |
| **Data**        | pandas 2.x                                                                                           | Fast CSV/Parquet IO ([github.com][3])                                      |
| **Concurrency** | `threading` + `queue.Queue` (thread-safe) ([docs.python.org][4], [stackoverflow.com][5])             |                                                                            |
| **GUI**         | Tkinter (`root.after` for main-thread calls only) ([stackoverflow.com][6], [reddit.com][7])          |                                                                            |
| **Formatter**   | Black – opinionated, PEP 8-compliant ([github.com][8], [reddit.com][9])                              |                                                                            |
| **Linter**      | Ruff – 10-100× faster than flake8, autofix support ([docs.astral.sh][10], [blog.jerrycodes.com][11]) |                                                                            |
| **Tests**       | pytest – asserts + rich failure reports ([docs.pytest.org][12], [docs.pytest.org][13])               |                                                                            |
| **VCS flow**    | Git feature-branch workflow ([atlassian.com][14], [docs.aws.amazon.com][15])                         |                                                                            |

---

## Repository Structure

```
run_artibot.py                 - command-line entry point
ARTIBOT.py                     - legacy monolith (read-only)
master_config.json             - sample config (never commit real keys)
Gemini_BTCUSD_1h.csv           - example dataset
best_model_weights*.pth        - saved checkpoints (large; keep out of git)

artibot/                       - ***edit code here only***
│  bot_app.py      - spawns training, live polling, GUI threads
│  dataset.py      - CSV loader, dataset classes
│  model.py        - Transformer definition
│  ensemble.py     - ensemble controller
│  training.py     - background training loop + exchange I/O
│  rl.py           - meta-reinforcement agent
│  backtest.py     - evaluation utilities
│  metrics.py      - performance statistics
│  gui.py          - Tkinter dashboard
│  environment.py  - first-run pip installer
│  globals.py      - shared state (to be refactored!)
│
Reinforcement Learning Test/   - offline RL prototype (read-only)
Last_working_bot_but_one_py/   - last-known-good single-file bot (read-only)
Old bot versions/              - assorted history (read-only)
```

---

## Development Guidelines

### Key Principles

* **One branch = one idea** – keep `main` always runnable.
* Let **Black** format; never hand-tweak line breaks ([github.com][8]).
* Run **Ruff** with `--fix` to auto-remove unused imports, etc. ([docs.astral.sh][10])
* Keep business logic in `artibot/` modules; GUI files only draw & schedule events.
* Use **type hints** and a short **module docstring** in every new file.
* No hard-coded secrets; load from env vars or `master_config.json`.
* Prefer `queue.Queue` for cross-thread data; avoid new global vars. ([docs.python.org][4])

### Python Best Practices (Artibot-specific)

#### Dataset helpers

```python
# dataset.py
import pandas as pd

def load_csv_hourly(path: str) -> pd.DataFrame:
    """Load 1-hour OHLCV CSV; returns UTC-indexed DataFrame."""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")  # :contentReference[oaicite:11]{index=11}
    return df.sort_index()
```

#### Training loop threading pattern

```python
# training.py (excerpt)
import threading, queue

orders_q: queue.Queue[str] = queue.Queue()

def trainer():
    while True:
        batch = data_q.get()
        loss = model.train_on(batch)
        metrics_q.put(loss)

threading.Thread(target=trainer, daemon=True).start()
```

* `queue.Queue` handles its own locks – safe for put/get across threads. ([docs.python.org][4])
* UI updates must be posted back with `root.after(ms, callback)`; calling widgets from a worker thread can crash Tkinter. ([stackoverflow.com][6])

---

## Quick-Start Workflow (5 Commands)

```bash
# 1 Create feature branch
git checkout -b feat/my-idea           # isolates changes :contentReference[oaicite:14]{index=14}

# 2 Hack in artibot/
#    (dataset.py, model.py, etc.)

# 3 Format + lint fast
pre-commit run --all-files             # Black + Ruff hooks :contentReference[oaicite:15]{index=15}

# 4 Run tests (even 1 assert is fine)
pytest -q                              # install: pip install -U pytest :contentReference[oaicite:16]{index=16}

# 5 Push & open PR
git push origin feat/my-idea
```

---

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate   # std-lib venv ​:contentReference[oaicite:17]{index=17}
pip install -r requirements.txt                     # first run auto-installs via environment.py
python run_artibot.py                               # may take minutes on first install
```

### Required Environment Variables

The bot expects the following variables to be set before launching. Copy `.env.example` to `.env` and fill in your keys:

- `OPENAI_API_KEY`
- `PHEMEX_API_KEY_LIVE` and `PHEMEX_API_SECRET_LIVE`
- `PHEMEX_API_KEY_TEST` and `PHEMEX_API_SECRET_TEST`

`master_config.json` contains placeholders for these values so secrets never
live in the repository.

---

## Testing Strategy

* Locate tests in `tests/`, name them `test_*.py`.
* Use plain `assert`—pytest rewrites to rich reports. ([docs.pytest.org][13])
* Aim for one short test per new function; e.g.:

```python
# tests/test_metrics.py
from artibot.metrics import sharpe_ratio

def test_sharpe_ratio_basic():
    pnl = [1, -1, 1, -1]
    assert sharpe_ratio(pnl) == 0
```

Run with `pytest -q`; green output means merge-ready.

---

## Common Issues

| Symptom                                         | Likely Cause                        | Fix                                                                                                  |
| ----------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `RuntimeError: main thread is not in main loop` | Updating Tkinter from worker thread | Use `root.after` callback ([reddit.com][7])                                                          |
| GPU not found warnings                          | No CUDA device                      | PyTorch silently falls back to CPU; ignore or set `CUDA_VISIBLE_DEVICES=""` ([stackoverflow.com][1]) |
| Hang during training                            | Infinite `queue.get()`              | Call with timeout or ensure producer thread still alive                                              |
| Linter flood after commit                       | Forgot `pre-commit install`         | Run once: `pre-commit install`                                                                       |

---

## Reference Resources

* Python `queue` docs – thread-safe queues ([docs.python.org][4])
* Tkinter thread safety FAQ ([stackoverflow.com][6])
* PyTorch CPU/GPU tips ([stackoverflow.com][1])
* Black code style guide ([github.com][8])
* Ruff documentation ([docs.astral.sh][10])
* pytest quick-start ([docs.pytest.org][12])
* Atlassian Git workflow tutorial ([atlassian.com][14])

---

**Save this file as `agents.md`; commit once, and future Codex runs will follow these rails automatically.**

[1]: https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu?utm_source=chatgpt.com "How to tell PyTorch to not use the GPU? - Stack Overflow"
[2]: https://discuss.pytorch.org/t/trying-to-run-cpu-instead-of-cuda/175331?utm_source=chatgpt.com "Trying to run cpu instead of cuda - PyTorch Forums"
[3]: https://github.com/astral-sh/ruff-pre-commit?utm_source=chatgpt.com "astral-sh/ruff-pre-commit: A pre-commit hook for Ruff. - GitHub"
[4]: https://docs.python.org/3/library/queue.html?utm_source=chatgpt.com "queue — A synchronized queue class — Python 3.13.4 documentation"
[5]: https://stackoverflow.com/questions/14053102/is-python-queue-queue-get-and-put-thread-safe?utm_source=chatgpt.com "Is python Queue.queue get and put thread safe? - Stack Overflow"
[6]: https://stackoverflow.com/questions/58118723/is-tkinters-after-method-thread-safe?utm_source=chatgpt.com "Is tkinter's `after` method thread-safe? - python - Stack Overflow"
[7]: https://www.reddit.com/r/learnpython/comments/10qzto6/how_does_tkinter_multithreading_work_and_why/?utm_source=chatgpt.com "How does tkinter multithreading work, and why? : r/learnpython"
[8]: https://github.com/psf/black?utm_source=chatgpt.com "psf/black: The uncompromising Python code formatter - GitHub"
[9]: https://www.reddit.com/r/Python/comments/exrtgn/my_unpopular_opinion_about_black_code_formatter/?utm_source=chatgpt.com "My unpopular opinion about black code formatter : r/Python - Reddit"
[10]: https://docs.astral.sh/ruff/linter/?utm_source=chatgpt.com "The Ruff Linter - Astral Docs"
[11]: https://blog.jerrycodes.com/ruff-the-python-linter/?utm_source=chatgpt.com "Ruff: one Python linter to rule them all - Jerry Codes"
[12]: https://docs.pytest.org/en/stable/how-to/assert.html?utm_source=chatgpt.com "How to write and report assertions in tests - pytest documentation"
[13]: https://docs.pytest.org/en/stable/getting-started.html?utm_source=chatgpt.com "Get Started - pytest documentation"
[14]: https://www.atlassian.com/git/tutorials/comparing-workflows?utm_source=chatgpt.com "Git Workflow | Atlassian Git Tutorial"
[15]: https://docs.aws.amazon.com/prescriptive-guidance/latest/choosing-git-branch-approach/advantages-and-disadvantages-of-the-gitflow-strategy.html?utm_source=chatgpt.com "Advantages and disadvantages of the Gitflow strategy"
