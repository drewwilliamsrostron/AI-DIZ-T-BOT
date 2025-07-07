import importlib
import importlib.machinery
import subprocess
import sys
import types


def test_enable_flash_sdp_auto_install(monkeypatch):
    monkeypatch.setenv("FLASH_SDP_AUTO_INSTALL", "1")
    monkeypatch.setenv("ARTIBOT_SKIP_INSTALL", "1")
    install_calls = []
    monkeypatch.setattr(subprocess, "check_call", lambda cmd: install_calls.append(cmd))

    cuda_mod = types.ModuleType("torch.backends.cuda")
    cuda_mod.enable_flash_sdp = lambda flag=True: None
    cuda_mod.is_flash_attention_available = lambda: False
    cuda_mod.flash_sdp_enabled = lambda: True
    backends_mod = types.ModuleType("torch.backends")
    backends_mod.cuda = cuda_mod

    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.backends = backends_mod
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch_stub.version = types.SimpleNamespace(cuda="11.8")
    torch_stub.device = lambda x: x

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.backends", backends_mod)
    monkeypatch.setitem(sys.modules, "torch.backends.cuda", cuda_mod)

    def fake_reload(module):
        cuda_mod.is_flash_attention_available = lambda: True
        return module

    monkeypatch.setattr(importlib, "reload", fake_reload)

    import artibot.core.device as dev
    importlib.reload(dev)

    assert dev.is_flash_sdp_enabled()
