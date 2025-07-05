import logging

try:  # torch may be mocked or missing during tests
    import torch

    try:
        _has_cuda = torch.cuda.is_available()
    except AttributeError:
        _has_cuda = False
    DEVICE = torch.device("cuda" if _has_cuda else "cpu")
except Exception:  # pragma: no cover - torch absent

    class _CPU:
        type = "cpu"

    DEVICE = _CPU()

logging.info("Using device: %s", getattr(DEVICE, "type", DEVICE))
