import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ruff: noqa: E402
os.environ["ARTIBOT_SKIP_INSTALL"] = "1"
os.environ.setdefault("NUCLEAR_KEY", "0")
os.environ.setdefault("USE_WEIGHT_DIALOG", "0")
import artibot.environment

import types
import contextlib
import numpy as np

sys.modules.setdefault("openai", types.SimpleNamespace())


def noop() -> None:
    pass


artibot.environment.ensure_dependencies = noop


def pytest_addoption(parser):
    parser.addoption("--no-heavy", action="store_true", help="use lightweight mocks")


def pytest_configure(config):
    if not config.getoption("--no-heavy"):
        return

    # torch stub
    if "torch" not in sys.modules:

        class FakeModule:
            def parameters(self):
                return []

        class FakeTensor(np.ndarray):
            def mean(self, *a, **k):
                return np.mean(self, *a, **k)

            def std(self, *a, **k):
                return np.std(self, *a, **k)

            def softmax(self, axis=-1):
                e = np.exp(self - np.max(self, axis=axis, keepdims=True))
                return e / np.sum(e, axis=axis, keepdims=True)

            def sum(self, *a, **k):
                return np.sum(self, *a, **k)

            def max(self, *a, **k):
                return np.max(self, *a, **k)

            def clamp(self, min=None, max=None):
                out = self
                if min is not None:
                    out = np.maximum(out, min)
                if max is not None:
                    out = np.minimum(out, max)
                return out

        def tensor(data, **k):
            return np.array(data)

        torch_ns = types.SimpleNamespace(
            tensor=tensor,
            zeros=lambda *s, **k: np.zeros(s if len(s) > 1 else s[0]),
            zeros_like=lambda x: np.zeros_like(x),
            randn=lambda *s: np.random.randn(*s),
            manual_seed=np.random.seed,
            is_tensor=lambda x: isinstance(x, np.ndarray),
            float32=np.float32,
            nn=types.SimpleNamespace(Module=FakeModule, Parameter=FakeTensor),
            optim=types.SimpleNamespace(
                SGD=lambda *a, **k: types.SimpleNamespace(
                    param_groups=[{"lr": 0.0, "weight_decay": 0.0}]
                ),
                Adam=lambda *a, **k: types.SimpleNamespace(
                    param_groups=[{"lr": 0.0, "weight_decay": 0.0}]
                ),
                AdamW=lambda *a, **k: types.SimpleNamespace(
                    param_groups=[{"lr": 0.0, "weight_decay": 0.0}]
                ),
            ),
            no_grad=contextlib.nullcontext,
            device=lambda *a, **k: None,
            Tensor=FakeTensor,
            utils=types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)),
            inf=float("inf"),
            isfinite=np.isfinite,
        )
        sys.modules["torch"] = torch_ns

    # schedule stub
    if "schedule" not in sys.modules:
        sched = types.ModuleType("schedule")

        def every(*a, **k):
            return types.SimpleNamespace(do=lambda f: None)

        sched.every = every
        sched.run_pending = lambda: None
        sys.modules["schedule"] = sched
