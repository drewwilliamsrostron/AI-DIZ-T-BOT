import sys
import types


def test_heartbeat_gpu(monkeypatch):
    cuda_mod = types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda idx: "GPU",
    )
    torch_mod = types.SimpleNamespace(cuda=cuda_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    nvml_mod = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda idx: object(),
        nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=55),
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=5, total=10),
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "nvidia", types.SimpleNamespace(nvml=nvml_mod))

    from artibot.utils.heartbeat import sample

    res = sample()
    assert "gpu" in res and "vram" in res
