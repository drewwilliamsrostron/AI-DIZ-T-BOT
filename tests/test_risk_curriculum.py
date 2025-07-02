import importlib

import artibot.hyperparams as hyperparams
import artibot.training as training


def test_apply_risk_curriculum():
    importlib.reload(hyperparams)
    importlib.reload(training)

    training.apply_risk_curriculum(10)
    assert hyperparams.RISK_FILTER["MIN_SHARPE"] == -2.0
    assert hyperparams.RISK_FILTER["MAX_DRAWDOWN"] == -0.80

    training.apply_risk_curriculum(30)
    assert hyperparams.RISK_FILTER["MIN_SHARPE"] == -1.0
    assert hyperparams.RISK_FILTER["MAX_DRAWDOWN"] == -0.50

    training.apply_risk_curriculum(50)
    assert hyperparams.RISK_FILTER["MIN_SHARPE"] == 0.0
    assert hyperparams.RISK_FILTER["MAX_DRAWDOWN"] == -0.25
