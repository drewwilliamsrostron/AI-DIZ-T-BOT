import logging
import numpy as np
import torch

PATIENCE = 10


def policy_gradient_loss(
    log_probs: torch.Tensor, trade_pnl: torch.Tensor
) -> torch.Tensor:
    """Return policy gradient loss weighted by normalised P&L advantage."""

    advantage = trade_pnl - trade_pnl.mean()
    advantage = advantage / (trade_pnl.std() + 1e-6)
    return -(advantage.detach() * log_probs).mean()


def log_btc_correlation(
    strategy_returns: np.ndarray,
    btc_returns_last30d: np.ndarray,
    profit_factor: float,
) -> float:
    """Log correlation with BTC and issue tracking warning when high."""

    if strategy_returns.size == 0 or btc_returns_last30d.size == 0:
        return float("nan")
    corr = float(np.corrcoef(strategy_returns, btc_returns_last30d)[0, 1])
    logging.info("BTC_CORR %.3f", corr)
    if corr > 0.85 and profit_factor < 1.2:
        logging.warning("Tracking BTC?")
    return corr


def train_step(model, optimizer, scheduler, batch):
    """Run a single training step.

    The learning rate scheduler must be stepped **after** the optimizer to
    avoid skipping the first value in the schedule (PyTorch >=1.1).  Passing
    ``scheduler`` as ``None`` disables LR updates.
    """

    model.train()
    x, y = batch
    optimizer.zero_grad()
    logits = model(x)[0]
    y = y.view(-1).long()
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()  # update weights first
    if scheduler is not None:
        scheduler.step()  # then adjust learning rate
    return loss.item()


def run_training_loop(ensemble, dataloader, optimizer, scheduler=None, epochs=1):
    """Simple loop calling :func:`train_step` and pruning periodically."""

    wait = 0
    for _ in range(epochs):
        for batch in dataloader:
            train_step(ensemble, optimizer, scheduler, batch)
        wait += 1
        if wait >= PATIENCE:
            ensemble.prune()
            wait = 0
