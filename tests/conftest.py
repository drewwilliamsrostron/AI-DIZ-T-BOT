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


if "--no-heavy" in sys.argv and "torch" not in sys.modules:
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

    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    optim_mod = types.ModuleType("torch.optim")

    nn_mod.Module = FakeModule
    nn_mod.Parameter = FakeTensor

    optim_mod.SGD = lambda *a, **k: types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}])
    optim_mod.Adam = lambda *a, **k: types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}])
    optim_mod.AdamW = lambda *a, **k: types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}])
    lr_sched_mod = types.ModuleType("torch.optim.lr_scheduler")
    lr_sched_mod.ReduceLROnPlateau = object
    lr_sched_mod.CosineAnnealingLR = object
    optim_mod.lr_scheduler = lr_sched_mod

    torch_mod.tensor = tensor
    torch_mod.zeros = lambda *s, **k: np.zeros(s if len(s) > 1 else s[0])
    torch_mod.zeros_like = lambda x: np.zeros_like(x)
    torch_mod.randn = lambda *s: np.random.randn(*s)
    torch_mod.manual_seed = np.random.seed
    torch_mod.is_tensor = lambda x: isinstance(x, np.ndarray)
    torch_mod.float32 = np.float32
    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    amp_mod = types.SimpleNamespace(GradScaler=lambda *a, **k: None, autocast=contextlib.nullcontext)
    cuda_mod = types.ModuleType("torch.cuda")
    amp_sub = types.ModuleType("torch.cuda.amp")
    amp_sub.GradScaler = lambda *a, **k: None
    amp_sub.autocast = contextlib.nullcontext
    cuda_mod.amp = amp_sub
    torch_mod.cuda = cuda_mod
    torch_mod.nn.functional = types.SimpleNamespace()
    torch_mod.no_grad = contextlib.nullcontext
    class FakeDevice:
        pass

    torch_mod.device = FakeDevice
    torch_mod.Tensor = FakeTensor
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    data_mod.DataLoader = object
    data_mod.TensorDataset = object
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    torch_mod.inf = float("inf")
    torch_mod.isfinite = np.isfinite
    torch_mod.set_num_threads = lambda n: None
    torch_mod.set_num_interop_threads = lambda n: None

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.optim"] = optim_mod
    sys.modules["torch.optim.lr_scheduler"] = lr_sched_mod
    sys.modules["torch.cuda"] = cuda_mod
    sys.modules["torch.cuda.amp"] = amp_sub
    sys.modules["torch.nn.functional"] = torch_mod.nn.functional
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod

    if "schedule" not in sys.modules:
        sched = types.ModuleType("schedule")

        def every(*a, **k):
            return types.SimpleNamespace(do=lambda f: None)

        sched.every = every
        sched.run_pending = lambda: None
        sys.modules["schedule"] = sched


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
