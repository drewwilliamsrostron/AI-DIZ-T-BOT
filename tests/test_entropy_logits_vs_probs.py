import torch
from artibot import utils


def test_entropy_logits_vs_probs():
    logits = torch.randn(1, 1, 8, 8)
    probs = logits.softmax(-1)
    diff = abs(utils.attention_entropy(logits) - utils.attention_entropy(probs))
    assert diff < 1e-4
