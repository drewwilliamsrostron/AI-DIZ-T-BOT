"""Runtime dependency checker and installer."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from typing import Dict


# Mapping of required packages and wheel specifiers for GPU or CPU installs
dependencies: Dict[str, Dict[str, str]] = {
    "torch": {"gpu": "torch==2.2.1+cu118", "cpu": "torch==2.2.1"},
    "torchvision": {"gpu": "torchvision==0.17.1+cu118", "cpu": "torchvision==0.17.1"},
    "torchaudio": {"gpu": "torchaudio==2.2.1+cu118", "cpu": "torchaudio==2.2.1"},
    "finbert-embedding": {"any": "finbert-embedding==0.1.6"},
}


_LOG = logging.getLogger(__name__)


def _driver_major() -> int:
    """Return the major NVIDIA driver version or 0 if undetectable."""

    cmd = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=os.name == "nt",
            check=True,
        )
        first = proc.stdout.strip().splitlines()[0]
        return int(first.split(".")[0])
    except Exception:
        return 0


def _install(pkg: str, extra_args: list[str] | None = None) -> None:
    """Install ``pkg`` via pip with optional ``extra_args``."""

    args = [sys.executable, "-m", "pip", "install", pkg]
    if extra_args:
        args.extend(extra_args)
    _LOG.warning("Installing missing package: %s", pkg)
    subprocess.run(args, check=True)


def ensure_dependencies() -> None:
    """Ensure core dependencies are installed."""

    if sys.modules.get("_deps_checked"):
        return
    sys.modules["_deps_checked"] = True

    gpu = _driver_major() >= 516
    wheel_key = "gpu" if gpu else "cpu"
    extra_index = ["--extra-index-url", "https://download.pytorch.org/whl/cu118"] if gpu else None

    for name, options in dependencies.items():
        import_name = name.replace("-", "_")
        if importlib.util.find_spec(import_name) is not None:
            continue
        pkg = options.get(wheel_key) or options.get("any")
        if not pkg:
            continue
        extra = extra_index if wheel_key == "gpu" and "gpu" in options else None
        _install(pkg, extra)

    import torch  # noqa: WPS433

    cuda_ok = torch.cuda.is_available()
    path = "GPU" if gpu and cuda_ok else "CPU"
    _LOG.info("PyTorch initialized using %s path", path)
