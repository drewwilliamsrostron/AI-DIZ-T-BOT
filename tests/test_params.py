from artibot.rl import MetaTransformerRL

class DummyEnsemble:
    pass

def test_rl_params():
    agent = MetaTransformerRL(DummyEnsemble())
    assert agent.eps_start == 0.20
    assert agent.eps_end == 0.05
    assert agent.eps_decay == 0.99
    assert agent.entropy_beta == 0.005
