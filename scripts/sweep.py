import yaml
from pathlib import Path
from artibot.optuna_opt import run_bohb


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()

    yaml.safe_load(Path(args.config).read_text())
    hp, params = run_bohb(n_trials=args.trials)
    out = Path("best")
    out.mkdir(exist_ok=True)
    result_path = out / "bohb_result.yaml"
    result_path.write_text(yaml.safe_dump({"indicator_hp": vars(hp), **params}))
    print("Best parameters saved to", result_path)
