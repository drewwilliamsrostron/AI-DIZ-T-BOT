"""Auto-install missing PyPI packages on first import."""

from importlib import import_module
import importlib.util
import logging
import os
import subprocess
import sys

LOG = logging.getLogger("auto_install")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


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


def ensure_pkg(pkg: str, import_as: str | None = None) -> None:
    """Install ``pkg`` if missing unless running under CI."""
    mod = import_as or pkg
    if importlib.util.find_spec(mod) is not None:
        return
    if os.environ.get("CI") == "true":
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def ensure_finbert() -> None:
    """Ensure the ``finbert`` package is installed."""
    ensure_pkg("finbert")


def ensure_schedule() -> None:
    """Ensure the ``schedule`` package is installed."""
    ensure_pkg("schedule")
