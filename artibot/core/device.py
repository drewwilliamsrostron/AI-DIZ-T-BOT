import importlib
import logging
import subprocess

import torch

log = logging.getLogger("device")


def select_device() -> torch.device:
    """Return detected device, installing the GPU wheel when possible."""

    if torch.version.cuda is None:
        log.warning("CUDA wheel missing; attempting install")
        try:
            log.info("Installing torch %s+cu121", torch.__version__)
            subprocess.check_call(
                [
                    "pip",
                    "install",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu121",
                    "torch==" + torch.__version__ + "+cu121",
                ]
            )
            importlib.reload(torch)
        except Exception as exc:  # pragma: no cover - network errors
            log.warning("GPU wheel install failed: %s", exc)

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        log.info("Using CUDA device %s: %s", idx, name)
        return torch.device("cuda", idx)

    log.info("Using CPU")
    return torch.device("cpu")


def get_device() -> torch.device:
    """Return the globally selected device."""

    return select_device()


DEVICE = get_device()


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
