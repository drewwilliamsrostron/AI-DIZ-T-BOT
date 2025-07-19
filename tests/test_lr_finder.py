import torch
from torch.utils.data import TensorDataset

from artibot.lr_finder import find_optimal_lr
from artibot.hyperparams import HyperParams


def test_lr_finder_returns_reasonable_lr():
    x = torch.randn(256, 10)
    y = torch.randn(256, 1)
    ds = TensorDataset(x, y)
    model = torch.nn.Sequential(torch.nn.Linear(10, 1))
    hp = HyperParams()
    hp.batch_size = 16
    lr, curve = find_optimal_lr(
        model,
        torch.nn.MSELoss(),
        ds,
        hp,
        min_lr=1e-5,
        max_lr=1e-2,
        num_iter=50,
        device="cpu",
    )
    assert curve
    assert 1e-5 < lr < 1e-2
