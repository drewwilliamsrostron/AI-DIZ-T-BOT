import torch, math
from torch.utils.data import DataLoader
from collections import deque

@torch.no_grad()
def find_optimal_lr(model, loss_fn, dataset, hp,
                    min_lr=1e-7, max_lr=1e-1, num_iter=100,
                    beta=0.98, device="cpu"):
    """
    Returns lr_suggestion, loss_curve (list of (lr, smoothed_loss))
    Implements Leslie Smith LR-range test.
    """
    dl = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
    dl_iter = iter(dl)

    lr_mult = (max_lr / min_lr) ** (1 / num_iter)
    lr = min_lr
    for p in model.parameters():
        p.grad = None
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    avg_loss, best_loss = 0., float("inf")
    losses, lrs = [], []

    for step in range(num_iter):
        try:
            x, y = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl); x, y = next(dl_iter)

        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed = avg_loss / (1 - beta**(step+1))

        losses.append(smoothed); lrs.append(lr)

        # Track best loss
        if smoothed < best_loss or step == 0:
            best_loss = smoothed
        # Stop if diverging
        if smoothed > 4 * best_loss:
            break

        loss.backward(); optim.step(); optim.zero_grad()

        lr *= lr_mult
        for pg in optim.param_groups:
            pg["lr"] = lr

    # pick lr slightly before the minimum
    min_idx = losses.index(min(losses))
    lr_suggestion = lrs[max(0, min_idx - 3)] * 0.35
    return lr_suggestion, list(zip(lrs, losses))
