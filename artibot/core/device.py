import logging
import subprocess

import torch

log = logging.getLogger("device")


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        log.info("Using CUDA device %s: %s", idx, name)
        return torch.device("cuda", idx)
    log.warning("CUDA not available – falling back to CPU")
    return torch.device("cpu")


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
