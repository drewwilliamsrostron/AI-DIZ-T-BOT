import torch


def train_step(model, optimizer, scheduler, batch):
    """One training step with proper scheduler order."""
    model.train()
    x, y = batch
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    optimizer.step()  # Weight update FIRST
    scheduler.step()  # Then adjust learning rate
    return loss.item()
