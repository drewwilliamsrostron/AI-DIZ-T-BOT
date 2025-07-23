"""Simple entry point for the Optuna optimiser."""

from __future__ import annotations

import argparse
from pathlib import Path

from artibot.dataset import load_csv_hourly
from artibot.optuna_opt import run_search


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Gemini_BTCUSD_1h.csv")
    p.add_argument("--n_trials", type=int, default=20)
    args = p.parse_args()

    data = load_csv_hourly(args.data)
    hp, lr, wd = run_search(data, n_trials=args.n_trials)
    out = Path("best_params.txt")
    out.write_text(
        f"lr={lr}\nwd={wd}\n" + "\n".join(f"{k}={v}" for k, v in hp.__dict__.items())
    )
    print(f"Best params written to {out}")
