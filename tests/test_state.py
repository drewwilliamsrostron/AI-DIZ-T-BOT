import json
import os

from artibot import globals as G, state


def test_best_reward_never_none_after_state_load(dummy_checkpoint):
    state.load(dummy_checkpoint)
    assert G.global_best_composite_reward is not None


def test_checkpoint_roundtrip_best_reward(tmp_path):
    G.global_best_composite_reward = 7.5
    os.chdir(tmp_path)
    from artibot.training import save_checkpoint

    save_checkpoint()
    data = json.load(open("checkpoint.json"))
    assert data["best_reward"] == 7.5
    G.global_best_composite_reward = float("-inf")
    state.load("checkpoint.json")
    assert G.global_best_composite_reward == 7.5
