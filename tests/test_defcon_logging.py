import logging
import types

import artibot.training as training


def test_defcon5_logging(monkeypatch, caplog):
    def dummy_objective(trial):
        return 0.0

    class DummyStudy:
        def __init__(self):
            self.trials = []
            self.params = {"lr": 0.001, "entropy_beta": 0.001}

        def optimize(self, func, n_trials=1, timeout=None):
            trial = types.SimpleNamespace(params=self.params)
            func(trial)
            self.trials.append(trial)

        @property
        def best_params(self):
            return self.params

    monkeypatch.setattr(training, "objective", dummy_objective)
    monkeypatch.setattr(training.optuna, "create_study", lambda direction: DummyStudy())

    caplog.set_level(logging.INFO)
    training.run_hpo(n_trials=1)

    assert any("ENTERING DEFCON 5" in r.message for r in caplog.records)
