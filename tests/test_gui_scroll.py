import importlib
import sys


def test_scrollable_imports_on_headless():
    sys.modules["tkinter"] = None  # simulate head-less
    sys.modules["tkinter.ttk"] = None
    sys.modules["torch"] = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("torch", None)
    )
    sys.modules["openai"] = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("openai", None)
    )
    sys.modules["pandas"] = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("pandas", None)
    )
    import artibot.gui as g

    assert hasattr(g, "build_scrollable")
