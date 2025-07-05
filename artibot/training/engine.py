"""Training engine helpers."""

from __future__ import annotations

from pathlib import Path
import logging
import torch

CHECKPOINTS_DIR = Path("models/checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def save_epoch_checkpoint(model: torch.nn.Module, epoch: int) -> None:
    """Save ``model`` state and keep only the last 5 checkpoints."""

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / f"epoch_{epoch}.pt"
    torch.save(model.state_dict(), path)
    files = sorted(CHECKPOINTS_DIR.glob("epoch_*.pt"))
    if len(files) > 5:
        for f in files[:-5]:
            try:
                f.unlink()
            except OSError:
                logging.warning("Failed to remove %s", f)
