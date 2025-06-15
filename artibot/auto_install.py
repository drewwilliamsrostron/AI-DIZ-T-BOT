"""Auto-install missing PyPI packages on first import."""

from importlib import import_module
import importlib.util
import logging
import subprocess
import sys

LOG = logging.getLogger("auto_install")


def install(pkg: str, import_as: str | None = None) -> None:
    """Install ``pkg`` via pip unless ``import_as`` is already importable."""
    mod_name = import_as or pkg
    if importlib.util.find_spec(mod_name) is not None:
        return
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
