import torch
import types
import sys
from tests.test_rl import load_rl_module


def test_pick_action_moves_state():
    duckdb = types.ModuleType("duckdb")
    duckdb.connect = lambda *a, **k: types.SimpleNamespace(
        execute=lambda *a, **k: types.SimpleNamespace(fetchone=lambda: (0,))
    )
    duckdb.DuckDBPyConnection = object
    duckdb.InvalidInputException = Exception
    duckdb.BinderException = Exception
    sys.modules.setdefault("duckdb", duckdb)

    rl = load_rl_module()
    rl.globals.get_warmup_step = lambda: 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = rl.MetaTransformerRL(ensemble=None, device=device)
    agent.model.to(device)
    state = torch.randint(0, 10, (agent.model.state_dim,))
    agent.pick_action(state)
