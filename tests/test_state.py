from artibot import globals as G, state


def test_best_reward_never_none_after_state_load(dummy_checkpoint):
    state.load(dummy_checkpoint)
    assert G.global_best_composite_reward is not None
