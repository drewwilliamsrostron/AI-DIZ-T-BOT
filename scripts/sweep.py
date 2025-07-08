import yaml
import math
import random
import shutil
from pathlib import Path
from artibot.run_artibot import main_train


def sample_params(axes):
    params = {}
    for name, spec in axes.items():
        if spec.get("sampling") == "log_uniform":
            params[name] = 10 ** random.uniform(
                math.log10(spec["min"]), math.log10(spec["max"])
            )
        elif spec.get("sampling") == "dirichlet":
            import numpy as np

            params[name] = np.random.dirichlet([1] * len(spec.get("weights", [1])))
        else:
            params[name] = random.choice(spec["choices"])
    return params


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--early_stop_epochs", type=int, default=3)
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    axes = yaml.safe_load(Path(args.config).read_text())["experiment_axes"]
    results = []
    for _ in range(20):
        params = sample_params(axes)
        res = main_train(**params, early_stop=args.early_stop_epochs)
        results.append((res["sharpe"], res["checkpoint_path"]))
    for _, ckpt in sorted(results, key=lambda x: -x[0])[: args.top_k]:
        shutil.copy(ckpt, Path("best") / Path(ckpt).name)
