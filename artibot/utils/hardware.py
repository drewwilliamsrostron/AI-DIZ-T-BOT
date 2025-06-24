import torch

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # pragma: no cover - mocked torch
    device = "cpu"
