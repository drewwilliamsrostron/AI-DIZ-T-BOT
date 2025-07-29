#!/usr/bin/env python3
"""
Complex AI Trading + Continuous Training + Robust Backtest Each Epoch + Live Phemex + Tkinter GUI
+ Meta–Control via Neural Network (Policy Gradient w/ Transformer) for Hyperparameter Adjustment
(Now includes all 9 improvements requested)
"""
# ruff: noqa: E402
###############################################################################
# NumPy 2.x compatibility shim – restores removed constants for old libraries
# numpy <2 wheels stop at Python 3.12 (NumPy 2.1 adds Py 3.13 support)
# numpy>=2.1,<3 ; python_version >= "3.13"
# numpy<2       ; python_version <  "3.13"
###############################################################################
import sys
import os
import warnings
import subprocess
import importlib.metadata as imd


def _ensure_numpy_lt2() -> None:
    """Install the appropriate NumPy for this Python."""
    if os.environ.get("NUMPY_PIN_DONE"):
        return
    try:
        ver = imd.version("numpy")
    except Exception:
        ver = ""

    if sys.version_info >= (3, 13):
        if not ver or tuple(int(x) for x in ver.split(".")[:2]) < (2, 1):
            print("[env] Installing NumPy>=2.1 for Python 3.13+")
            pkg = "numpy>=2.1,<3"
        else:
            pkg = None
    else:
        if not ver.startswith("1."):
            print("[env] Downgrading incompatible NumPy", ver, "→ <2.0")
            pkg = "numpy<2"
        else:
            pkg = None

    if pkg:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-input",
                    "--force-reinstall",
                    pkg,
                ]
            )
        except Exception as e:  # pragma: no cover - network errors
            print("[env] NumPy pin failed:", e)

    os.environ["NUMPY_PIN_DONE"] = "1"


_ensure_numpy_lt2()

# This prevents torch from emitting experimental API warnings under NumPy 2
os.environ.setdefault("NUMPY_EXPERIMENTAL_ARRAY_API", "1")
os.environ["_ARRAY_API"] = "numpy"
warnings.filterwarnings(
    "ignore",
    message=r"Failed to initialize NumPy.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"A module that was compiled using NumPy 1.x.*",
    module=r"torch.nn.modules.transformer",
    category=UserWarning,
)

try:
    import numpy as _np
except ModuleNotFoundError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as _np

# Re-export the aliases deleted in NumPy 2
for _name, _value in {
    "NaN": _np.nan,
    "Inf": _np.inf,
    "PINF": _np.inf,
    "NINF": -_np.inf,
}.items():
    setattr(_np, _name, _value)
    sys.modules["numpy"].__dict__[_name] = _value

###############################################################################
# Smart installer – works on Python 3.9 → 3.13, CPU or CUDA
###############################################################################
import sys
import subprocess
import platform
import logging


os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count()))


def _install_pytorch_for_env() -> None:
    """
    Install torch + torchvision + torchaudio that **exist** for the
    interpreter that is running this script.

    • 3.9 – 3.12 → stable 2.2.1 wheels (+cu118 or +cpu)
    • 3.13       → current nightly wheels (>= 2.6.0.dev…)
    """
    major, minor = sys.version_info[:2]

    # crude CUDA check: we’ll try GPU wheels first only when a NVIDIA adapter
    # visible to Windows. Fall back to CPU if the install fails later.
    cuda_ok = platform.system() == "Windows" and "NVIDIA" in subprocess.getoutput(
        "wmic path win32_VideoController get name"
    )

    if (major, minor) >= (3, 13):
        # 🟡 Nightly wheels already publish cp313 tags
        index = (
            "https://download.pytorch.org/whl/nightly/cu118"
            if cuda_ok
            else "https://download.pytorch.org/whl/nightly/cpu"
        )
        pkg_line = ["torch", "torchvision", "torchaudio", "--pre", "-f", index]
    else:
        # 🟢 Stable LTS wheels (2.2.x)
        suffix = "+cu118" if cuda_ok else "+cpu"
        index = (
            "https://download.pytorch.org/whl/cu118"
            if cuda_ok
            else "https://download.pytorch.org/whl/cpu"
        )
        pkg_line = [
            f"torch==2.2.1{suffix}",
            f"torchvision==0.17.1{suffix}",
            f"torchaudio==2.2.1{suffix}",
            "--extra-index-url",
            index,
        ]

    logging.info("Installing PyTorch trio for this environment …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkg_line])


def install_dependencies() -> None:
    """Install PyTorch (if missing) plus the rest of the requirements."""
    try:
        import torch  # noqa: F401
    except ImportError:
        _install_pytorch_for_env()

    # ---------------- other pure-Python dependencies ----------------
    pkgs = {
        "openai": "openai",
        "ccxt": "ccxt",
        "pandas": "pandas",
        "numpy": ("numpy>=2.1,<3" if sys.version_info >= (3, 13) else "numpy<2"),
        "matplotlib": "matplotlib",
        # 'scikit-learn' installs as 'sklearn'
        "sklearn": "scikit-learn",
        "TA-Lib": "TA-Lib",  # imported as `talib`
        "screeninfo": "screeninfo",  # monitor DPI detection
        "duckdb": "duckdb",  # feature store backend
        # pytest is required to run the bundled tests
        "pytest": "pytest",
        "transformers": "transformers==4.52.4",  # FinBERT sentiment model
        "schedule": "schedule",  # cron-like job scheduler
        "psutil": "psutil",
        "hmmlearn": "hmmlearn",
    }
    for import_name, pip_name in pkgs.items():
        try:
            __import__("talib" if import_name == "TA-Lib" else import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


###############################################################################
#  ❒  TA-Lib fallback for Python 3.13 (no binary wheels yet)
###############################################################################
try:
    import talib as _talib  # noqa: F401  ← works on 3.9-3.12 if wheels exist
except ModuleNotFoundError:
    # no wheel → install pandas-ta once and build a tiny shim
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-ta"])
    import pandas as pd
    import pandas_ta as pta

    class _TaShim:  # exposes *only* the funcs you use
        @staticmethod
        def RSI(arr, timeperiod=14):
            return pta.rsi(pd.Series(arr), length=timeperiod).values

        @staticmethod
        def MACD(arr, fastperiod=12, slowperiod=26, signalperiod=9):
            series = pd.Series(arr)
            try:
                res = pta.macd(
                    series, fast=fastperiod, slow=slowperiod, signal=signalperiod
                )
                if res is not None:
                    return (
                        res[f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"].values,
                        res[f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"].values,
                        res[f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"].values,
                    )
            except Exception:
                pass

            # Fallback manual calculation when pandas_ta returns None or errors
            ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
            ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signalperiod, adjust=False).mean()
            hist = macd - signal
            return macd.values, signal.values, hist.values

    import sys

    sys.modules["talib"] = _TaShim()  # ✅ calls like talib.RSI(...) keep working


def ensure_dependencies():
    """Install required packages if they are missing."""
    if os.environ.get("ARTIBOT_SKIP_INSTALL") == "1":
        return
    if os.environ.get("NUMPY_PIN_DONE") != "1":
        _ensure_numpy_lt2()
        os.environ["NUMPY_PIN_DONE"] = "1"
    install_dependencies()
