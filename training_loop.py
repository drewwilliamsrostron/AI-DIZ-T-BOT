import torch


def train_step(model, optimizer, scheduler, batch):
    """Run a single training step.

    The learning rate scheduler must be stepped **after** the optimizer to
    avoid skipping the first value in the schedule (PyTorch >=1.1).  Passing
    ``scheduler`` as ``None`` disables LR updates.
    """

    model.train()
    x, y = batch
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    loss.backward()
    optimizer.step()  # update weights first
    if scheduler is not None:
        scheduler.step()  # then adjust learning rate
    return loss.item()
