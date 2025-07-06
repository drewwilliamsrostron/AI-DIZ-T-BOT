import importlib
import logging
import subprocess
import sys

import torch

log = logging.getLogger("device")


CUDA_TAG = "+cu121"
TORCH_VER = "2.2.1"
TV_VER = "0.17.1"
TA_VER = "2.2.1"
CUDA_IDX = "https://download.pytorch.org/whl/cu121"


def _have_cuda() -> bool:
    return torch.version.cuda is not None


def _install_cuda() -> None:
    if _have_cuda():
        return
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        CUDA_IDX,
        f"torch=={TORCH_VER}{CUDA_TAG}",
        f"torchvision=={TV_VER}{CUDA_TAG}",
        f"torchaudio=={TA_VER}{CUDA_TAG}",
        "--extra-index-url",
        "https://pypi.org/simple",
    ]
    log.warning("CUDA wheel missing – installing… (%s)", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        importlib.reload(sys.modules["torch"])
        log.info("CUDA wheel installed and torch reloaded")
    except subprocess.CalledProcessError as e:
        log.error("CUDA wheel install failed: %s", e)


try:
    _install_cuda()
except Exception:
    log.exception("unexpected CUDA-install failure")


def get_device() -> torch.device:
    """Return the globally selected device."""

    return DEVICE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_cuda() -> None:
    """Warn when CUDA support is missing."""

    has_cuda = torch.version.cuda is not None
    try:
        ret = subprocess.call(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        ret = 1
    if not (has_cuda and ret == 0):
        msg = (
            "torch-cu121 wheel not installed or driver < 545; "
            "see README 'CUDA 12.1' section"
        )
        print(f"\033[91m{msg}\033[0m")
