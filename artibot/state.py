import json
import artibot.globals as G


def load(path: str) -> dict:
    """Load state from ``path`` and update globals."""
    try:
        with open(path, "r") as f:
            state = json.load(f)
    except Exception:
        state = {}
    G.global_best_composite_reward = state.get("best_reward", float("-inf"))
    return state
