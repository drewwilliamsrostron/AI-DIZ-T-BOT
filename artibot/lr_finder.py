"""Learning rate range test utilities."""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader


def find_optimal_lr(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    hp,
    *,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    num_iter: int = 100,
    beta: float = 0.98,
    device: torch.device | str = "cpu",
) -> tuple[float, list[tuple[float, float]]]:
    """Run the Leslie Smith LR-range test and return a suggested LR."""

    device = torch.device(device)
    bs = getattr(hp, "batch_size", 64)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    lr_mult = (max_lr / min_lr) ** (1 / num_iter)
    lr = min_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    lrs: List[float] = []
    losses: List[float] = []
    avg_loss = 0.0
    best_loss = float("inf")

    data_iter = iter(loader)
    model.train()
    for step in range(num_iter):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = loss_fn(out, y)
        loss_value = float(loss.item())

        avg_loss = beta * avg_loss + (1 - beta) * loss_value
        smoothed = avg_loss / (1 - beta ** (step + 1))

        lrs.append(lr)
        losses.append(smoothed)

        if smoothed < best_loss or step == 0:
            best_loss = smoothed
        if smoothed > 4 * best_loss:
            break

        loss.backward()
        optimizer.step()

        lr *= lr_mult
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    if not losses:
        return min_lr, []
    idx = int(min(range(len(losses)), key=losses.__getitem__))
    return lrs[idx], list(zip(lrs, losses))
