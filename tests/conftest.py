import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ruff: noqa: E402
os.environ["ARTIBOT_SKIP_INSTALL"] = "1"
import artibot.environment


def noop() -> None:
    pass


artibot.environment.ensure_dependencies = noop
