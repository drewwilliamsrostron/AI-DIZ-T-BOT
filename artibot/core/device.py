import os
import sys
import subprocess
import importlib
import logging

import torch


def ensure_flash_sdp() -> None:
    """Auto-install FlashAttention builds and enable kernels."""
    try:
        from torch.backends.cuda import is_flash_attention_available
    except Exception:  # pragma: no cover - CPU-only environments
        return

    if (
        not is_flash_attention_available()
        and os.getenv("FLASH_SDP_AUTO_INSTALL") == "1"
    ):
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--pre",
            "torch",
            "torchvision",
            "--extra-index-url",
            "https://download.pytorch.org/whl/nightly/cu118",
        ]
        print(
            f"Installing FlashAttention-enabled PyTorch nightly: {' '.join(install_cmd)}"
        )
        subprocess.check_call(install_cmd)
        import importlib as _importlib

        _importlib.reload(torch)
    torch.backends.cuda.enable_flash_sdp(True)


ensure_flash_sdp()

log = logging.getLogger("device")


def enable_flash_sdp(flag: bool = True) -> bool:
    """Enable FlashAttention/SDP kernels when available."""

    try:
        torch.backends.cuda.enable_flash_sdp(flag)
    except Exception as exc:  # pragma: no cover - optional feature
        log.debug("Flash SDP not enabled: %s", exc)
        return False
    else:
        if flag:
            log.info("Flash SDP kernels enabled")
        return True


def is_flash_sdp_enabled() -> bool:
    """Return ``True`` when FlashAttention is available and active."""

    try:
        from torch.backends.cuda import is_flash_attention_available, flash_sdp_enabled

        return is_flash_attention_available() and flash_sdp_enabled()
    except Exception:  # pragma: no cover - optional feature
        return False


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


if os.environ.get("ARTIBOT_SKIP_INSTALL") != "1":
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
