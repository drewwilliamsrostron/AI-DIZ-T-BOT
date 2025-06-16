"""Auto-install missing PyPI packages on first import."""

from importlib import import_module
import importlib.util
import logging
import os
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

LOG = logging.getLogger("auto_install")


def install(pkg: str, import_as: str | None = None) -> None:
    """Install ``pkg`` via pip unless ``import_as`` is already importable."""
    mod_name = import_as or pkg
    if importlib.util.find_spec(mod_name) is not None:
        return
    if os.environ.get("CI") == "true":
        raise ModuleNotFoundError(mod_name)
    LOG.warning("Package %s missing – installing…", pkg)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg],
            stdout=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        LOG.error("Pip install failed for %s: %s", pkg, exc)
        raise
    import_module(mod_name)


def ensure_pkg(pkg: str) -> None:
    """Install ``pkg`` if missing (no-op on CI)."""

    try:
        __import__(pkg.split("==")[0])
        return
    except ModuleNotFoundError:
        pass
    if os.environ.get("CI") == "true":
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def require(
    pkg: str, import_name: str | None = None, min_version: str | None = None
) -> None:
    """Import ``import_name`` or install ``pkg`` interactively."""

    mod_name = import_name or pkg.split("==")[0].split("[")[0]
    try:
        mod = import_module(mod_name)
        if min_version:
            from packaging.version import Version

            if Version(getattr(mod, "__version__", "0")) < Version(min_version):
                raise ModuleNotFoundError(mod_name)
        return
    except ModuleNotFoundError:
        if os.environ.get("CI") == "true":
            raise
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
