# ruff: noqa: E402
import sys
import types
from importlib.machinery import ModuleSpec

for name in ["openai", "ccxt", "tkinter", "tkinter.ttk"]:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    sys.modules.setdefault(name, mod)

matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *a, **k: None
matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
sys.modules.setdefault("matplotlib", matplotlib)
pyplot = types.ModuleType("pyplot")
pyplot.__spec__ = ModuleSpec("pyplot", loader=None)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
backend = types.ModuleType("matplotlib.backends.backend_tkagg")
backend.FigureCanvasTkAgg = object
backend.__spec__ = ModuleSpec("matplotlib.backends.backend_tkagg", loader=None)
sys.modules.setdefault("matplotlib.backends.backend_tkagg", backend)

import torch
import artibot.globals as G
from artibot.ensemble import EnsembleModel
from artibot.hyperparams import IndicatorHyperparams
import pickle


def test_load_best_weights_updates_stats(tmp_path, monkeypatch):
    ckpt = {
        "best_composite_reward": 1.0,
        "state_dicts": [{}],
        "indicator_hparams": {"atr_period": 42},
    }
    path = tmp_path / "best.pth"
    torch.save = lambda obj, p: pickle.dump(obj, open(p, "wb"))
    torch.load = lambda p, map_location=None: pickle.load(open(p, "rb"))
    torch.save(ckpt, path)

    def dummy_backtest(*_a, **_k):
        return {
            "equity_curve": [(0, 1.0), (1, 2.0)],
            "net_pct": 123.0,
            "trades": 4,
            "sharpe": 1.5,
            "max_drawdown": -0.2,
            "trade_details": [],
            "composite_reward": 5.0,
            "inactivity_penalty": 0.0,
            "days_without_trading": 0,
            "days_in_profit": 1.0,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "avg_trade_duration": 0.0,
            "avg_win": 0.1,
            "avg_loss": -0.05,
        }

    monkeypatch.setattr("artibot.ensemble.robust_backtest", dummy_backtest)
    monkeypatch.setattr(
        "artibot.ensemble.compute_yearly_stats", lambda *a, **k: (None, "")
    )
    monkeypatch.setattr(
        "artibot.ensemble.compute_monthly_stats", lambda *a, **k: (None, "")
    )

    ens = EnsembleModel.__new__(EnsembleModel)
    ens.device = torch.device("cpu")
    ens.weights_path = str(path)
    ens.models = [types.SimpleNamespace(load_state_dict=lambda *a, **k: None)]
    ens.optimizers = [
        types.SimpleNamespace(param_groups=[{"lr": 1e-3, "weight_decay": 0.0}])
    ]
    ens._indicator_hparams = IndicatorHyperparams()
    ens.hp = types.SimpleNamespace(indicator_hp=ens._indicator_hparams)
    ens.load_best_weights(str(path), data_full=[[0, 0, 0, 0, 0, 0]] * 30)

    assert G.global_net_pct == 123.0
    assert G.global_best_net_pct == 123.0
    assert G.global_num_trades == 4
    assert G.global_win_rate == 0.5
    assert G.global_profit_factor == 1.2
    assert ens.indicator_hparams.atr_period == 42
