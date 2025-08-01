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
from importlib.machinery import ModuleSpec
import numpy as np
import pytest
import sklearn as _sklearn  # ensure real sklearn is loaded for hmmlearn

_ = _sklearn.__version__

sys.modules.setdefault("openai", types.SimpleNamespace())


def noop() -> None:
    pass


artibot.environment.ensure_dependencies = noop


def pytest_addoption(parser):
    parser.addoption("--no-heavy", action="store_true", help="use lightweight mocks")


def pytest_configure(config):
    if not (config.getoption("--no-heavy") or os.environ.get("NO_HEAVY") == "1"):
        return
    os.environ["NO_HEAVY"] = "1"

    # torch stub
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.__spec__ = ModuleSpec("torch", loader=None)

        class FakeModule:
            def __init__(self, *a, **k):
                pass

            def __call__(self, *a, **k):
                return FakeTensor(np.zeros((len(a[0]), 3))) if a else None

            def parameters(self):
                return []

            def to(self, *a, **k):
                return self

        class FakeTensor(np.ndarray):
            def __new__(cls, data):
                return np.asarray(data).view(cls)

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

            def argmax(self, axis=None, **k):
                if "dim" in k:
                    axis = k.pop("dim")
                return np.ndarray.argmax(self, axis=axis, **k)

            def unsqueeze(self, axis):
                return np.expand_dims(self, axis).view(FakeTensor)

            def size(self, dim=None):
                return self.shape if dim is None else self.shape[dim]

            def to(self, *a, **k):
                return self

        def tensor(data, **k):
            return FakeTensor(data)

        torch_mod.tensor = tensor
        torch_mod.zeros = lambda *s, **k: FakeTensor(
            np.zeros(s if len(s) > 1 else s[0])
        )
        torch_mod.arange = lambda *a, **k: FakeTensor(np.arange(*a, **k))
        torch_mod.zeros_like = lambda x: FakeTensor(np.zeros_like(x))
        torch_mod.randn = lambda *s: FakeTensor(np.random.randn(*s))
        torch_mod.manual_seed = np.random.seed
        torch_mod.is_tensor = lambda x: isinstance(x, np.ndarray)
        torch_mod.float32 = np.float32
        torch_mod.float = np.float32
        nn_mod = types.ModuleType("torch.nn")
        nn_mod.Module = FakeModule
        nn_mod.Linear = FakeModule
        nn_mod.Parameter = FakeTensor
        fn_mod = types.ModuleType("torch.nn.functional")
        fn_mod.relu = lambda x: x
        fn_mod.cross_entropy = lambda logits, target: FakeTensor(0.0)
        fn_mod.__spec__ = ModuleSpec("torch.nn.functional", loader=None)
        nn_mod.functional = fn_mod
        nn_mod.__spec__ = ModuleSpec("torch.nn", loader=None)
        torch_mod.nn = nn_mod
        optim_mod = types.ModuleType("torch.optim")
        optim_mod.SGD = lambda *a, **k: types.SimpleNamespace(
            param_groups=[{"lr": 0.0, "weight_decay": 0.0}]
        )
        optim_mod.Adam = lambda *a, **k: types.SimpleNamespace(
            param_groups=[{"lr": 0.0, "weight_decay": 0.0}]
        )
        optim_mod.AdamW = lambda *a, **k: types.SimpleNamespace(
            param_groups=[{"lr": 0.0, "weight_decay": 0.0}]
        )
        optim_mod.__spec__ = ModuleSpec("torch.optim", loader=None)
        lr_sched = types.ModuleType("torch.optim.lr_scheduler")
        lr_sched.__spec__ = ModuleSpec("torch.optim.lr_scheduler", loader=None)
        lr_sched.ReduceLROnPlateau = object
        lr_sched.StepLR = object
        lr_sched.OneCycleLR = object
        lr_sched.CosineAnnealingLR = object
        optim_mod.lr_scheduler = lr_sched
        torch_mod.optim = optim_mod
        torch_mod.no_grad = contextlib.nullcontext

        class Device:
            def __init__(self, typ="cpu", *a, **k):
                self.type = typ

        torch_mod.device = Device
        torch_mod.set_num_threads = lambda n: None
        torch_mod.set_num_interop_threads = lambda n: None
        torch_mod.Tensor = FakeTensor
        utils_mod = types.ModuleType("torch.utils")
        data_mod = types.ModuleType("torch.utils.data")
        data_mod.Dataset = object
        data_mod.DataLoader = object
        data_mod.TensorDataset = object
        data_mod.random_split = lambda ds, lens: (ds, ds)
        tb_mod = types.ModuleType("torch.utils.tensorboard")

        class DummyWriter:
            def __init__(self, *a, **k):
                pass

            def add_scalar(self, *a, **k):
                pass

            def add_scalars(self, *a, **k):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        tb_mod.SummaryWriter = DummyWriter
        utils_mod.data = data_mod
        utils_mod.tensorboard = tb_mod
        utils_mod.__spec__ = ModuleSpec("torch.utils", loader=None)
        data_mod.__spec__ = ModuleSpec("torch.utils.data", loader=None)
        tb_mod.__spec__ = ModuleSpec("torch.utils.tensorboard", loader=None)
        torch_mod.utils = utils_mod
        torch_mod.inf = float("inf")
        torch_mod.isfinite = np.isfinite
        cuda_mod = types.ModuleType("torch.cuda")
        amp_mod = types.ModuleType("torch.cuda.amp")
        amp_mod.GradScaler = object
        amp_mod.autocast = contextlib.nullcontext
        cuda_mod.amp = amp_mod
        cuda_mod.is_available = lambda: False
        back_cuda = types.ModuleType("torch.backends.cuda")
        back_cuda.enable_flash_sdp = lambda *a, **k: None
        back_cuda.is_flash_attention_available = lambda: False
        back_cuda.flash_sdp_enabled = lambda: False
        backends_mod = types.ModuleType("torch.backends")
        backends_mod.cuda = back_cuda
        cuda_mod.__spec__ = ModuleSpec("torch.cuda", loader=None)
        amp_mod.__spec__ = ModuleSpec("torch.cuda.amp", loader=None)
        back_cuda.__spec__ = ModuleSpec("torch.backends.cuda", loader=None)
        backends_mod.__spec__ = ModuleSpec("torch.backends", loader=None)
        torch_mod.cuda = cuda_mod
        torch_mod.backends = backends_mod
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = nn_mod
        sys.modules["torch.nn.functional"] = fn_mod
        sys.modules["torch.optim"] = optim_mod
        sys.modules["torch.optim.lr_scheduler"] = lr_sched
        sys.modules["torch.utils"] = utils_mod
        sys.modules["torch.utils.data"] = data_mod
        sys.modules["torch.utils.tensorboard"] = tb_mod
        sys.modules["torch.cuda"] = cuda_mod
        sys.modules["torch.cuda.amp"] = amp_mod

    if "pandas" not in sys.modules:
        pd = types.ModuleType("pandas")
        pd.Series = lambda *a, **k: object()
        pd.DataFrame = lambda *a, **k: object()
        sys.modules["pandas"] = pd

    if "transformers" not in sys.modules:
        tfm = types.ModuleType("transformers")
        tfm.pipeline = lambda *a, **k: lambda x: [{"score": 0.0}]
        sys.modules["transformers"] = tfm

    if "duckdb" not in sys.modules:
        db = types.ModuleType("duckdb")
        db.connect = lambda *a, **k: types.SimpleNamespace(
            execute=lambda *a, **k: types.SimpleNamespace(fetchone=lambda: (0,))
        )
        db.DuckDBPyConnection = object
        db.InvalidInputException = Exception
        db.BinderException = Exception
        sys.modules["duckdb"] = db

    if "sklearn" not in sys.modules:
        sk = types.ModuleType("sklearn")
        impute_mod = types.ModuleType("sklearn.impute")
        impute_mod.KNNImputer = object
        cluster_mod = types.ModuleType("sklearn.cluster")
        cluster_mod.KMeans = object
        utils_mod = types.ModuleType("sklearn.utils")
        utils_mod.check_random_state = lambda seed=None: None
        sk.impute = impute_mod
        sk.cluster = cluster_mod
        sk.utils = utils_mod
        sys.modules["sklearn"] = sk
        sys.modules["sklearn.impute"] = impute_mod
        sys.modules["sklearn.cluster"] = cluster_mod
        sys.modules["sklearn.utils"] = utils_mod

    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        trial_mod = types.ModuleType("optuna.trial")
        trial_mod.Trial = object
        optuna.trial = trial_mod
        optuna.create_study = lambda *a, **k: types.SimpleNamespace(
            optimize=lambda *a, **k: None, best_params={}
        )
        optuna.__spec__ = ModuleSpec("optuna", loader=None)
        trial_mod.__spec__ = ModuleSpec("optuna.trial", loader=None)
        sys.modules["optuna"] = optuna
        sys.modules["optuna.trial"] = trial_mod

    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")
        psutil.cpu_percent = lambda *a, **k: 0.0
        psutil.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
        sys.modules["psutil"] = psutil

    if "requests" not in sys.modules:
        req = types.ModuleType("requests")
        req.RequestException = Exception
        req.get = lambda *a, **k: types.SimpleNamespace(json=lambda: {})

        class _DummySession:
            def __init__(self, *a, **k):
                pass

            def get(self, *a, **k):
                return types.SimpleNamespace(json=lambda: {})

        req.Session = _DummySession
        util_mod = types.SimpleNamespace(default_user_agent=lambda *a, **k: "")
        req.utils = util_mod
        exc_mod = types.SimpleNamespace(
            HTTPError=Exception,
            Timeout=Exception,
            TooManyRedirects=Exception,
            RequestException=Exception,
            ConnectionError=Exception,
        )
        req.exceptions = exc_mod
        sys.modules["requests"] = req
        sys.modules["requests.utils"] = util_mod
        sys.modules["requests.exceptions"] = exc_mod

    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.use = lambda *a, **k: None
        matplotlib.__spec__ = ModuleSpec("matplotlib", loader=None)
        plt = types.ModuleType("matplotlib.pyplot")
        plt.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)
        plt.figure = lambda *a, **k: types.SimpleNamespace(
            add_subplot=lambda *a, **k: object()
        )
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = plt
        backend = types.ModuleType("matplotlib.backends.backend_tkagg")
        backend.FigureCanvasTkAgg = lambda *a, **k: types.SimpleNamespace(
            get_tk_widget=lambda: object()
        )
        sys.modules["matplotlib.backends.backend_tkagg"] = backend

    # schedule stub
    if "schedule" not in sys.modules:
        sched = types.ModuleType("schedule")

        def every(*a, **k):
            return types.SimpleNamespace(do=lambda f: None)

        sched.every = every
        sched.run_pending = lambda: None
        sys.modules["schedule"] = sched


@pytest.fixture
def dummy_checkpoint(tmp_path):
    path = tmp_path / "ckpt.json"
    path.write_text("{}")
    return path
