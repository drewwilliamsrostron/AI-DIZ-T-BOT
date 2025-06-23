import types
import artibot.globals as G
from artibot.hyperparams import HyperParams, IndicatorHyperparams, WARMUP_STEPS
from artibot.rl import MetaTransformerRL


def test_toggles_ignored_after_warmup(monkeypatch):
    monkeypatch.setattr(G, "get_warmup_step", lambda: WARMUP_STEPS)

    class DummyOpt:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.01, "weight_decay": 0.0}]

    class DummyEnsemble:
        def __init__(self) -> None:
            self.optimizers = [DummyOpt()]
            self.indicator_hparams = IndicatorHyperparams()

    ens = DummyEnsemble()
    agent = MetaTransformerRL(ens)
    hp = HyperParams(indicator_hp=ens.indicator_hparams)
    before = ens.indicator_hparams.use_rsi

    agent.apply_action(hp, ens.indicator_hparams, {"toggle_rsi": 1})

    assert ens.indicator_hparams.use_rsi == before
