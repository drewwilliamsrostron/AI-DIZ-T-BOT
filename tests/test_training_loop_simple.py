import torch
from training_loop import train_step


class DummyModel(torch.nn.Linear):
    def forward(self, x):
        logits = super().forward(x)
        return logits, None, None


def test_train_step_improves_loss():
    torch.manual_seed(0)
    model = DummyModel(2, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batch_x = torch.randn(32, 2)
    weights = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    with torch.no_grad():
        labels = (batch_x @ weights.T).argmax(dim=1)
    batch = (batch_x, labels)
    init = torch.nn.functional.cross_entropy(model(batch_x)[0], labels)
    loss = init.item()
    for _ in range(50):
        loss = train_step(model, opt, None, batch)
    assert loss < init.item()
