import torch
from torch import nn
from types import SimpleNamespace

from artibot.lr_finder import find_optimal_lr

# Define a simple synthetic dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=200, n_features=10):
        self.x = torch.randn(n_samples, n_features)
        # simple linear relation with noise
        weights = torch.randn(n_features, 1)
        self.y = self.x @ weights + 0.1 * torch.randn(n_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def test_find_optimal_lr_returns_reasonable_value():
    dataset = DummyDataset(n_samples=128, n_features=8)
    # simple 2-layer MLP
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    loss_fn = nn.MSELoss()
    # minimal hyperparams namespace with batch_size
    hp = SimpleNamespace(batch_size=16)
    suggested_lr, curve = find_optimal_lr(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        hp=hp,
        min_lr=1e-6,
        max_lr=1e-2,
        num_iter=50,
        device="cpu",
    )
    # check that the suggested LR is within a reasonable range
    assert suggested_lr > 1e-6
    assert suggested_lr < 1e-2
