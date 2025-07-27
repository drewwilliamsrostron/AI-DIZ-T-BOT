from artibot.meta_controller import MetaTransformerRL


def test_replay_buffer_capped():
    cfg = {"META": {"buffer": 5, "batch": 2}}
    agent = MetaTransformerRL(state_dim=2, action_dim=2, cfg=cfg)
    for _ in range(10):
        agent.store_transition([0.0, 0.0], 0, 1.0, [0.0, 0.0], False)
    assert len(agent.replay) <= cfg["META"]["buffer"]

