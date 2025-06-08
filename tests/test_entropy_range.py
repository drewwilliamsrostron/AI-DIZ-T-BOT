import torch
from artibot import utils


def test_entropy_range_logits():
    logits = torch.randn(2, 4, 32, 32)
    val = utils.attention_entropy(logits)
    assert 0.5 < val < 4.0
